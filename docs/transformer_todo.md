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

### Scope

Byte-level only — each byte value (0–255) maps directly to a token ID. No BPE, no subword merging, no external training data. Vocab size = 256 + special tokens (BOS, EOS, PAD, UNK) = ~260.

### What's needed

| Component | Status | Effort |
|-----------|--------|--------|
| Tokenizer struct | Holds byte-to-ID lookup (identity map: byte[n] = n) and special token IDs. ~12 bytes. | Trivial |
| Encode: text → IDs | Walk UTF-8 byte string, emit one ID per byte. If byte is valid UTF-8 continuation, treat same as any other byte. | Low |
| Decode: IDs → text | Walk ID sequence, map each ID back to byte value. If ID >= 256, skip special tokens or emit replacement char. | Low |
| Special tokens | BOS (257), EOS (258), PAD (259), UNK (260). Encode prepends BOS/appends EOS. Decode strips specials. | Low |
| Integration: text → tensor | Encode to IDs → allocate `tensor_zeros_data(1, [N])` → copy IDs into data region as ints. Input to `tensor_embedding`. | Low |

### Recommendation

Simple flat tokenizer, no hash tables, no merge files. Encode = byte-for-ID copy. Decode = ID-for-byte copy (clamp to 0–255).

```c
typedef struct {
    int bos_id, eos_id, pad_id, unk_id;
} tokenizer;

int  *tokenizer_encode(tokenizer *tok, const char *text, int *len);
char *tokenizer_decode(tokenizer *tok, const int *ids, int len);
```

Embedding table vocab size = 261 (256 bytes + BOS/EOS/PAD/UNK). No lookup table needed for tokenizer — byte value is the ID.

Allocates from data pool. Encode/decode are O(N) linear walks.

### Effort estimate

| Component | Lines |
|-----------|-------|
| Tokenizer struct + init | ~10 |
| Encode (byte walk) | ~15 |
| Decode (ID walk) | ~15 |
| Special token handling | ~10 |
| Tensor pipeline (text→tensor) | ~20 |
| **Total** | **~60** |

---

## Summary: Implementation Order & Dependencies

### Dependency graph

```
sigmoid ──> silu ──> swiglu ──────────────────┐
embedding ────────────────────────────────────┤
                                              │
causal_softmax ──> attention ─────────────────┤
rope ──────────────────────────────────────┐  │
tokenizer ──> embedding ───────────────────┤  │
                                           │  │
matmul_2d ──> [15a] batched_matmul ✅ ──────┤  │
                                            │  │
[15b] tensor_cat ──> [15c] kv_cache ────────┤  │
                                            │  │
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
| 13 | Fused `tensor_attention(Q,K,V, mask)` — end-to-end | (12) | ~80 | DONE |

### Phase 3 — Tokenizer

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 14 | Tokenizer: byte-level encode + decode + special tokens | — | ~60 | DONE |

### Phase 3b — Transformer Primitives

Gaps found during code audit before assembling the full model.

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 15a | `tensor_matmul` batched n-dim support (3D+) | — | ~80 | ✅ **DONE** |
| 15b | `tensor_cat` along dim with autograd | — | ~80 |
| 15c | KV-cache struct + append helper | (15b) | ~50 |
| 15d | Bump scratch pool (192MB → 512MB) | — | ~1 |

#### 15a — Batched `tensor_matmul` ✅ **DONE**

**What was implemented:**

Extended `tensor_matmul` in `src/ops_matrix.c` to handle N ≥ 2 dims with
NumPy-style broadcasting:

- **Forward:** Computes broadcast batch shape from all dims except the last 2.
  Loops over batch elements calling `cblas_sgemm` per slice (or manual triple
  loop when slices are non-contiguous). Output is a contiguous N-D tensor
  with shape `[broadcast(leading_dims), M, N]`.
- **Backward:** Same batch loop with `cblas_sgemm` (or manual loop) for both
  `da = gd @ B^T` and `db = A^T @ gd`. Handles self-matmul (`a == b`)
  through all batch dims. Gradient accumulation correctly sums over
  broadcast dims.
- **Strided fallback:** When inner 2D slices are non-contiguous (column
  stride != 1), uses full stride-based element-wise loops.
- **2D fast path preserved:** No change to the existing optimized 2D code path.
- **Autograd:** `grad_fn` wired; backward function discriminates 2D vs ND
  at runtime.

**Tests added:**
- `test/ref_batched_matmul.py` — PyTorch reference: 3D batched, broadcast
  both ways, 4D broadcast, self-matmul, finite-diff check, single-batch vs 2D.
- `test/test_batched_matmul.c` — C tests: 3D, 4D, broadcast, self-matmul,
  single-batch vs 2D equivalence, numerical gradient check, PyTorch reference
  values match, no-grad mode, regression checks for 2D still works.

**Files changed:**
- `src/ops_matrix.c` — batched matmul forward + backward
- `test/ref_batched_matmul.py` — new PyTorch reference
- `test/test_batched_matmul.c` — new C test suite

---

#### 15b — `tensor_cat` along dim with autograd

**Why:** KV-cache appends new K/V tokens along the sequence dim each step.
`tensor_slice` can write but needs pre-allocated max-size buffer. `tensor_cat`
is more natural and enables dynamic growth.

**What to do:**

```c
tensor *tensor_cat(const tensor *a, const tensor *b, int dim);
```

- Concatenate `a` and `b` along `dim`. All other dims must match.
- Allocate output from scratch pool, copy both inputs.
- Autograd backward: split `grad_output` along `dim` at the boundary
  (size of `a->shape[dim]`), scatter each half to the corresponding input.
- No BLAS needed — pure memcpy / strided copy.

**Effort:** ~80 lines + test.

---

#### 15c — KV-cache struct + append

**Why:** Autoregressive generation feeds one token at a time. Without caching,
each step recomputes K/V for all past tokens — O(N²) per step vs O(N) with cache.

**What to do:**

```c
typedef struct {
    tensor *k_cache;   // [B, H, max_seq, d_k], params pool
    tensor *v_cache;   // [B, H, max_seq, d_k], params pool
    int     seq_len;   // current valid length
    int     max_seq;
} kv_cache;

kv_cache *kv_cache_create(int B, int H, int max_seq, int d_k);
```

- Pre-allocate full-size buffers in params pool.
- `kv_cache_append(kvc, K_new, V_new)` — writes new K/V slices at
  position `seq_len`, increments `seq_len`. Uses `tensor_slice` on the
  cache to get the writable view (no alloc, no autograd — inference only).
- `kv_cache_get(kvc)` — returns `(K, V)` views of shape `[B, H, seq_len, d_k]`
  via `tensor_slice` along dim 2.
- No autograd needed — generation is eval-only (`dnn_no_grad`).

**KV-cache is NOT used during training** (teacher forcing computes full
sequence in one shot). So no backward pass required.

**Effort:** ~50 lines + test.

---

#### 15d — Bump scratch pool

Current: 192MB in `main.c`. Transformer with B=64, N=512, d=512, 12 layers
needs ~30MB/layer × 12 = 360MB activations, plus temp buffers.

Change to 512MB in `main.c` (and document in the transformer training entry
point). For smaller configs (B=16, N=256, d=256, 6 layers) ~30MB is fine,
but keeping a single safe default avoids cryptic OOM.

```c
mem_pool scratch = mem_pool_create(512 * 1024 * 1024);
```

---

### Phase 4 — Full Model

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 16 | `transformer_block` (pre-norm attn + pre-norm swiglu-ffn + residual) transformer.c | (3,6,7,9,11,15a) | ~60 |
| 17 | Decoder-only LM (embed → N×block → norm → lm_head) | (4,16) | ~60 |
| 18 | Training loop (next-token prediction, teacher forcing, cross-entropy) | (17) | ~100 |
| 19 | Generation loop (autoregressive, kv-cache optional) | (17,15b,15c) | ~80 |

### Total estimated new code

- **C source:** ~1300–1500 lines across `src/`, `include/
- **Python:** none needed
- **Tests:** ~400 lines (one test per new op, plus integration test for training loop)
- **No new dependencies** — Accelerate BLAS + libm + zlib already in the Makefile.

### Architectural considerations

- **KV cache** — essential for efficient autoregressive generation. Store K and V tensors per layer, append new tokens' K/V at each step. Needs `tensor_slice` for write (or a `tensor_cat` op along seq dim). `tensor_cat` would be a new op (concatenate along dim, with autograd). Alternatively, pre-allocate max-seq-length cache and slice into it — no new op, just pointer arithmetic.
- **Pre-norm vs post-norm** — `tensor_layer_norm` exists and supports arbitrary ndim. Pre-norm (layer norm before each sublayer, as used in Llama/GPT-NeoX) is slightly simpler to wire: `x = x + sublayer(norm(x))`.
- **Weight tying** — embedding weight = lm_head weight. Easy: reuse the same `tensor*` pointer in both `linear_create` (for lm_head) and the embedding table. Autograd accumulates grads to the same buffer.
- **Scratch pool sizing** — transformer activations per layer: attn ~4×[B,N,d] (Q,K,V,output) + FFN ~3×[B,N,4d] (gate, up, down). For batch=64, N=512, d=512: ~30MB per layer × N layers. With 12 layers this is ~360MB. Current scratch pool is 192MB — **needs bump to 512MB–1GB** for a full transformer.
