# Decoder-Only Transformer — Implementation Assessment

Based on codebase audit (`include/`, `src/`, `test/`, `docs/`). Evaluates what exists, what needs adaptation, and what must be built from scratch.

---

## 1. Causal Self-Attention

### What exists

- `tensor_matmul(a, b)` — BLAS-accelerated (via Accelerate on macOS), contiguous fast path, fallback triple-loop. Autograd wired (saves nothing, recomputes backward grads from inputs).
- `tensor_softmax(t, dim)` — numerically stable (max-subtract), NEON SIMD for 2D path, general nD coord-decompose fallback. Autograd wired (saves softmax output + dim).
- `tensor_transpose(t, d1, d2)` — lightweight view, no data copy.
- `tensor_reshape(t, ndim, shape)` — contiguous fast path (view), non-contiguous fallback copies.
- `tensor_slice(t, dim, start, len)` — view.
- `tensor_div`, `tensor_mul`, `tensor_add` — all with broadcasting, autograd, contiguous fast paths.
- Scale factor (divide by sqrt(d_head)) is trivial: `tensor_mul` with a broadcast scalar `tensor_pow(d_head, -0.5)` or precomputed constant.

### What's missing

| Component | Status | Effort |
|-----------|--------|--------|
| Multi-head QKV projection (linear → reshape → split) | Build on `linear` + `tensor_reshape`/`tensor_transpose`. No new ops. | Low |
| Scaled dot-product: `scores = q @ k^T / sqrt(d)` | `tensor_matmul` + `tensor_transpose` + `tensor_mul` (scalar). Works but materializes full `[B, H, N, N]` scores — O(n²) memory. | Low |
| Causal mask | **Critical.** Must prevent attending to future tokens. Two approaches: (a) materialize additive mask and pass to softmax, (b) modify softmax kernel to skip future positions. Approach (a) wastes memory/bandwidth — bad for long sequences. Approach (b) needs a new `tensor_causal_softmax` op or fused attention kernel. | **Medium** |
| `tensor_causal_softmax` (fused) | Would combine softmax + triangular mask in one kernel, O(n²) compute but no mask materialization. Could be a standalone op with its own grad_fn, or a fused `tensor_attention` that does `softmax(Q@K^T/√d)@V` end-to-end with causal masking internally. | **High** |
| Attention output projection | Post-attention linear. `linear_forward` works. | Low |

### Recommendation

Start with approach (a) — materialize mask via `tensor_triu(..., diagonal=1)` then `tensor_fill(-inf)` — to get correct behavior fast. Then replace with fused causal softmax as P1 optimization. The mask-as-additive approach uses O(N²) scratch per head; for MNLM-size (seq=512, d_model=512, 8 heads) this is ~1MB, fine. For longer sequences it becomes the bottleneck.

**New ops needed:**
- `tensor_triu(t, diagonal)` — upper-triangular mask generator (returns `[N,N]` tensor of 0s on/below diagonal, -inf above). Or simpler: `tensor_causal_mask(N)`.
- Fused `tensor_causal_softmax(scores)` — softmax over last dim with -inf on upper triangle, no mask materialization.

### Autograd

Attention backward needs:
- dScores = dP @ V^T  (P = softmax output)
- dV = P^T @ dP
- dQ = dScores @ K
- dK = dScores^T @ Q

All expressible via existing `tensor_matmul` backward (which uses `cblas_sgemm`). No new gradient math needed — just wire the existing matmul backward chain.

If fused `tensor_causal_softmax` is implemented, its backward needs the standard softmax + causal-mask gradient (same as softmax backward but zeroing upper-triangle positions in the `smax * (g - dot)` formula).

---

## 2. Embedding

### What exists

- `tensor_zeros`, `tensor_randn`, `tensor_uniform` — weight init.
- `tensor_slice` — could extract rows but no direct embedding lookup.

### What's missing

| Component | Status | Effort |
|-----------|--------|--------|
| Embedding table (`embedding`) | 2D param tensor `[vocab_size, d_model]`. `tensor_uniform` init works. | Low |
| Embedding forward (lookup) | `y[i] = table[ids[i]]` — integer index into first dim. **No existing op.** Needs `tensor_embedding(input_ids, table)` that takes `[N]` int labels and returns `[N, d_model]` float tensor. | **Low-Medium** |
| Embedding backward | Gradient flows to the looked-up rows only. dTable[ids[i]] += dY[i]. Sparse scatter-add. Needs a new backward kernel. | Low-Medium |

### Recommendation

Straightforward op: `tensor_embedding` reads int IDs from the input tensor's data region (stored as `int*` cast from the float data, same pattern as cross_entropy target). Forward copies rows. Backward scatters grads into the table rows via atomic-add (or sequential if no duplicate IDs in batch — guaranteed in MNLM with next-token prediction on disjoint sentences). Allocate from scratch pool.

**New ops needed:**
- `tensor_embedding(table, ids)` — forward lookup + backward scatter.

---

## 3. RoPE (Rotary Position Embedding)

### What exists

- `tensor_pow(t, exp)` — for computing `theta = base^{-2k/d}` frequency terms.
- `tensor_mul`, `tensor_add`, `tensor_sub`, `tensor_neg` — complex multiplication expressed as real-vector ops.
- `tensor_slice` — for splitting into even/odd pairs.
- NEON SIMD in `simd.h` — `simd_expf_f32` for fast exp (useful for computing sin/cos frequency tables on init, not in hot path).

### What's missing

| Component | Status | Effort |
|-----------|--------|--------|
| RoPE frequency table init | `theta_k = base^{-2k/d}` for k=0..d/2-1. Implemented via `tensor_rope_freqs(d, base)` in `src/rope.c`. Tests in `test/test_rope.c` and `ref_rope.py` (PyTorch ref). | Low | DONE |
| RoPE forward: apply rotation to Q and K | Rotate pairs `(x_{2k}, x_{2k+1})` by `(cos mθ_k, sin mθ_k)`. Needs `cos`/`sin` on each position m. Two approaches: (a) compose via `tensor_mul` + `tensor_add` — works but slow (many passes over data). (b) dedicated `tensor_rope(q, k, freqs)` kernel — single pass, vectorizable. | **Medium-High** |
| RoPE backward | Gradients for rotated Q/K require rotating grad_output by the same angles. Could reuse forward's sin/cos table. | Medium |

### Recommendation

RoPE is deceptively expensive when composed from primitives: each position needs cos(mθ) × even-half + sin(mθ) × odd-half for both Q and K, plus the reverse for each backward. That's 4-8 extra passes over [B,H,N,d] per layer.

**Implement dedicated `tensor_rope_inplace(q, k, freqs_cos, freqs_sin)`** that:
1. Reads position index m from the sequence dim.
2. Applies `q_new[2k] = q[2k]*cos(mθk) - q[2k+1]*sin(mθk)` and the paired formula for `q_new[2k+1]`, plus same for k.
3. Writes in-place (modifies Q/K directly). RoPE is applied before the causal mask step, so no need to preserve original Q/K for backward — just rotate the gradient incoming gradient in the same pattern.

Alternative simpler approach: apply RoPE via `tensor_reshape` to interleave pairs, then `tensor_mul` by cos/sin of the right shape with broadcasting. This avoids new ops but costs multiple passes.

**New ops needed:**
- `tensor_rope(q, k, cos_table, sin_table)` — in-place rotation of Q/K. Or at minimum a standalone `tensor_rotate_pairs(x, cos, sin)` that applies the 2×2 rotation to adjacent pairs along last dim.

---

## 4. SwiGLU Activation

### What exists

Formula: `SwiGLU(x) = Swish(xW_gate) ⊗ (xW_up)` where `Swish(z) = z * σ(z)`.

The two projections `xW_gate` and `xW_up` are just `linear_forward` calls — no new code.

The element-wise gate `Swish(z) = z * sigmoid(z)` needs:
- `tensor_mul` — exists.
- `tensor_sigmoid` — **currently a stub** (`ops_activation.c` line ~65, returns NULL).

### What's missing

| Component | Status | Effort |
|-----------|--------|--------|
| `tensor_sigmoid(t)` | Stub — must implement forward + backward. Forward: `1/(1+exp(-x))`. Backward: `sig * (1-sig) * grad`. Both trivially vectorizable with NEON. | **Low** (but critical) |
| `tensor_silu(t)` or Swish | If no dedicated op, compose as `tensor_mul(x, tensor_sigmoid(x))` — two ops, fine for correctness. Dedicated `tensor_silu` fuses into one pass, saves one read/write. | Low |
| `tensor_swiglu(gate_out, up_out)` or compose from parts | Compose: `silu = tensor_silu(gate_out)` → `tensor_mul(silu, up_out)`. Works, 3 scratch tensors per call. Dedicated fusion halves memory traffic. | Low-Medium |

### Recommendation

**Do the stubs first** — implement `tensor_sigmoid` (forward + backward). Without it SwiGLU is blocked. Then compose SwiGLU from existing ops:

```c
tensor *gate = linear_forward(ffn_gate, x);  // [B, N, 4*d]
tensor *up   = linear_forward(ffn_up, x);    // [B, N, 4*d]
tensor *out  = tensor_mul(tensor_silu(gate), up);
tensor *out2 = linear_forward(ffn_down, out); // [B, N, d]
```

This is correct and autograd-wired automatically. The 3-scratch-tensor overhead (~12KB for batch=64, d=512) is negligible. Fuse later if profiling shows it.

**New ops needed:**
- `tensor_sigmoid(t)` — implement from stub. Affects both SwiGLU and any future gating.
- (optional) `tensor_silu(t)` — fused Swish.
- (optional) `tensor_swiglu(gate, up)` — fused SwiGLU gate.

---

## 5. Byte-Level Tokenizer

### What exists

- **Nothing.** No text processing, no vocabulary, no tokenizer in the codebase. MNIST loads raw image bytes, not text.

### What's needed

| Component | Status | Effort |
|-----------|--------|--------|
| UTF-8 byte encoding | Python has `.encode("utf-8")`. In C need a UTF-8 decoder or treat raw bytes (byte-level BPE works on byte values 0-255). Simplest: treat input as raw byte sequence (each byte = 1 token). Compatible with GPT-2's byte-level BPE base. | Low |
| BPE merge table | Pre-computed BPE merges (from training on some corpus). Need a file format: sorted merge list `(byte_pair, new_id)`. | **High** (needs training script) |
| Tokenizer encode: text → IDs | Walk byte sequence, apply BPE merges greedily. O(N) with hash table for merge lookup. | Medium |
| Tokenizer decode: IDs → text | Walk ID sequence, each ID maps to a byte sequence (from vocab). Concatenate. | Low-Medium |
| Vocabulary data structures | `char *vocab[size]` — each entry is a byte string. `merge_ranks[(b1,b2)]` — map from byte pair to priority/rank. | Medium |
| Integration: text → tensor | Encode to tokens → allocate `tensor_zeros_data(1, [N])` → copy IDs into data region as ints. This is the input to `tensor_embedding`. | Low |

### Recommendation

**Three-tier approach:**

**Tier 1 (MVP): Raw byte mode**
No BPE. Treat each UTF-8 byte as a token. Vocab size = 256 (+ special tokens = ~260). Embedding table is tiny. Works for character-level generation — poor quality but gets the pipeline running immediately.

**Tier 2 (BPE): Python-based training + C decoder**
Train BPE merges in Python (using HuggingFace `tokenizers` or `tiktoken`). Export as:
- `vocab.bin` — binary file: `[vocab_size][byte_len][bytes]`
- `merges.bin` — binary file: `[num_merges][b1][b2][new_id]`

Write C `tokenizer.h` / `tokenizer.c`:
```c
typedef struct {
    int   vocab_size;
    char **vocab;
    int   *merge_buf;  // flat merge table for O(1) pair lookup
} tokenizer;

int  *tokenizer_encode(tokenizer *tok, const char *text, int *len);
char *tokenizer_decode(tokenizer *tok, const int *ids, int len);
```

**Tier 3 (Optimized):** Byte-level BPE with regex pre-tokenization (GPT-2 pattern), Unicode normalization, special token handling. This matches "state of the art" (GPT-4 / Llama 3 level tokenization).

### Effort estimate

| Tier | Components | Effort | Quality |
|------|------------|--------|---------|
| 1 — Raw bytes | No new C code beyond embed lookup. Just wire ID→tensor pipeline. | Very low | Poor |
| 2 — Python BPE + C decode | Python script + C tokenizer (~300 lines) | Medium | Good |
| 3 — Full byte-level BPE | Regex pre-tokenization, unicode normalization, special tokens, pretokenize cache | High | SOTA |

---

## Summary: Implementation Order & Dependencies

### Dependency graph

```
sigmoid ──> silu ──> swiglu
                        │
embedding ──────────────┤
                        │
matmul ──> attention ───┤
softmax    causal_mask   │
ropo ───────────────────┤
                         │
tokenizer ──> embedding ─┤
                         │
                   transformer_block
                   (attn + swiglu ffn + pre-norm + residual)
```

### Phase 1 — Foundation (no new ops, just compose)

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 1 | `tensor_sigmoid` forward + backward + test | — | ~60 | DONE |
| 2 | `tensor_silu` (compose from sigmoid+mul) | (1) | ~10 / ~40 | DONE |
| 3 | SwiGLU FFN block (compose from existing linear + silu + mul) | (1,2) | ~30 | DONE |
| 4 | Embedding lookup + backward | — | ~80 | DONE |
| 5 | Causal mask via `tensor_triu` + `tensor_fill(-inf)` | — | ~40 | DONE |
| 6 | Scaled dot-product attention (compose from matmul + softmax + mask) | (5) | ~60 | DONE |
| 7 | Multi-head split/merge wrappers | — | ~80 | DONE |

### Phase 2 — Activation Fusion & Position Encoding

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 8  | Fused `tensor_silu(x)` — single-pass Swish (no intermediate sigmoid tensor) | (1) | ~40 | DONE |
| 9  | Fused `tensor_swiglu(gate, up)` — single-pass gated activation | (8) | ~40 | DONE |
| 10 | RoPE frequency table init | — | ~40 | DONE |
| 11 | `tensor_rope` — dedicated in-place Q/K rotation | (10) | ~100 | DONE |
| 12 | Fused `tensor_causal_softmax(scores)` — no mask materialization | — | ~120 | DONE |
| 13 | Fused `tensor_attention(Q,K,V, mask)` — end-to-end | (12) | ~80 |

### Phase 3 — Tokenizer

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 14 | Byte-level vocab file format + Python training script | — | ~150 (py) |
| 15 | C tokenizer: encode (BPE greedy) + decode | (14) | ~300 |
| 16 | Tokenizer → tensor pipeline (text → IDs → embedding) | (4, 15) | ~30 |

### Phase 4 — Full Model

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 17 | `transformer_block` (pre-norm attn + pre-norm swiglu-ffn + residual) | (3,6,7,9,11) | ~60 |
| 18 | Decoder-only LM (embed → N×block → norm → lm_head) | (4,17) | ~60 |
| 19 | Training loop (next-token prediction, teacher forcing, cross-entropy) | (18) | ~100 |
| 20 | Generation loop (autoregressive, kv-cache optional) | (18) | ~80 |

### Total estimated new code

- **C source:** ~1400–1600 lines across `src/`, `include/
- **Python:** ~150 lines for BPE trainer script
- **Tests:** ~400 lines (one test per new op, plus integration test for training loop)
- **No new dependencies** — Accelerate BLAS + libm + zlib already in the Makefile. BPE trainer needs Python + `tokenizers` or `tiktoken` (dev-only, not linked into C binary).

### Architectural considerations

- **KV cache** — essential for efficient autoregressive generation. Store K and V tensors per layer, append new tokens' K/V at each step. Needs `tensor_slice` for write (or a `tensor_cat` op along seq dim). `tensor_cat` would be a new op (concatenate along dim, with autograd). Alternatively, pre-allocate max-seq-length cache and slice into it — no new op, just pointer arithmetic.
- **Pre-norm vs post-norm** — `tensor_layer_norm` exists and supports arbitrary ndim. Pre-norm (layer norm before each sublayer, as used in Llama/GPT-NeoX) is slightly simpler to wire: `x = x + sublayer(norm(x))`.
- **Weight tying** — embedding weight = lm_head weight. Easy: reuse the same `tensor*` pointer in both `linear_create` (for lm_head) and the embedding table. Autograd accumulates grads to the same buffer.
- **Scratch pool sizing** — transformer activations per layer: attn ~4×[B,N,d] (Q,K,V,output) + FFN ~3×[B,N,4d] (gate, up, down). For batch=64, N=512, d=512: ~30MB per layer × N layers. With 12 layers this is ~360MB. Current scratch pool is 192MB — **needs bump to 512MB–1GB** for a full transformer.
