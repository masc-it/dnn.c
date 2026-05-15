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
| RoPE forward: apply rotation to Q and K | Rotate pairs `(x_{2k}, x_{2k+1})` by `(cos mθ_k, sin mθ_k)`. Needs `cos`/`sin` on each position m. Two approaches: (a) compose via `tensor_mul` + `tensor_add` — works but slow (many passes over data). (b) dedicated `tensor_rope(q, k, freqs)` kernel — single pass, vectorizable. | **Medium-High** | DONE |
| RoPE backward | Gradients for rotated Q/K require rotating grad_output by the same angles. Could reuse forward's sin/cos table. | Medium | DONE |

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
[15b] tensor_cat ✅ ──> [15c] kv_cache ✅ ──┤  │
                                            │  │
                                       transformer_block ✅
                                       (attn + swiglu ffn + pre-norm + residual)
                                            │
                                       decoder_lm ✅
                                       (embed + n×block + norm + lm_head)
                                            │
                                       train_step ✅
                                       (teacher forcing + cross-entropy + optimizer)
                                            │
                                       gen ✅
                                       (autoregressive generation, kv-cache optional)
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
| 15b | `tensor_cat` along dim with autograd | — | ~80 | **DONE** |
| 15c | KV-cache struct + append helper | (15b) | ~50 | ✅ **DONE**
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

#### 15b — `tensor_cat` along dim with autograd ✅ **DONE**

**What was implemented:**

Extended `src/ops_elem.c` with `tensor_cat` and a `_cat_copy` helper, plus
`cat_backward`:

- **Forward:** Concatenates `a` and `b` along `dim`. All non-`dim` dimensions
  must match. Supports negative dim indexing. Output is contiguous, allocated
  from scratch pool.
- **Copy:** `_cat_copy` has a contiguous fast path (row-by-row `memcpy`) and a
  general coord-decompose fallback for strided inputs (e.g. after transpose).
- **Backward:** Splits `grad_output` along `dim` — elements with
  `coord[dim] < a->shape[dim]` flow to `a`'s grad buffer, rest to `b`.
  Has contiguous fast path (nested `for` loops, no coord-decompose) and a
  general coord-decompose fallback. Handles `a == b` (self-concatenation)
  correctly by accumulating both halves into the same buffer.
- **Autograd:** `grad_fn` wired; saves `dim` for backward. Grad mode
  respected — no tape when either input doesn't require grad or
  `dnn_no_grad` is active.

**Tests added:**
- `test/ref_cat.py` — PyTorch reference: 1D, 2D dim=0/dim=1, 3D dim=1,
  partial grad, self-concatenation, chain with matmul, exact value match.
- `test/test_cat.c` — C tests: 1D/2D/3D forward, dim normalization,
  backward (simple, 2D dim0/dim1, partial grad, self, sum loss, no-grad,
  chain with matmul).

**Files changed:**
- `include/ops.h` — added `tensor_cat` declaration
- `src/ops_elem.c` — added `cat_backward`, `_cat_copy`, `tensor_cat`
- `test/ref_cat.py` — new PyTorch reference
- `test/test_cat.c` — new C test suite
- `docs/transformer_todo.md` — marked 15b done

---

#### 15c — KV-cache struct + append helper ✅ **DONE**

**What was implemented:**

- **`kv_cache` struct** in `include/transformer.h` — holds `k_cache`, `v_cache`
  tensors (params pool), `seq_len`, `max_seq`.
- **`kv_cache_create(B, H, max_seq, d_k)`** — allocates zero-filled `[B, H, max_seq, d_k]`
  tensors from params pool via `tensor_zeros`.  seq_len starts at 0.
- **`kv_cache_append(kvc, K_new, V_new)`** — copies N_new tokens into the cache
  at position `seq_len` along dim 2.  Uses direct pointer arithmetic on the
  cache buffer with per-(b,h) memcpy — no scratch allocs during append.
  Asserts K/V are 4D, batch/head/d_k match, and cache has room.
- **`kv_cache_get_K(kvc)` / `kv_cache_get_V(kvc)`** — returns `tensor_slice`
  views of valid portion `[B, H, seq_len, d_k]` (lightweight, shares cache
  buffer).  Asserts `seq_len > 0`.
- **No autograd** — all cache ops are eval-only (params pool tensors created
  with `requires_grad=0`, no `grad_fn` wired).  KV-cache is NOT used during
  training.

**Design decision:** Append avoids `tensor_slice` scratch allocs in the hot
loop by computing cache buffer offsets directly.  A single `kv_cache_append`
call does zero scratch allocations — just `memcpy` into pre-allocated
param-pool buffers.  Get uses `tensor_slice` (one small scratch alloc per
view) since it's called once per forward pass during generation.

**Tests added:**
- `test/ref_kv_cache.py` — PyTorch reference: create cache, append one/multiple
  tokens, multi-batch/multi-head, fill to max seq, empty slice, value matching.
- `test/test_kv_cache.c` — C tests: create/zeroed, append one token (verify data
  in cache buffer via coord-based `tget`), append multiple tokens cumulatively,
  multi-batch multi-head exact position checks, fill to capacity, seq_len
  growth tracking.  All pass.

**Files changed:**
- `include/transformer.h` — added `kv_cache` struct + function declarations
- `src/transformer.c` — replaced placeholder with `kv_cache_*` implementation
- `test/ref_kv_cache.py` — new PyTorch reference
- `test/test_kv_cache.c` — new C test suite
- `docs/transformer_todo.md` — marked 15c done

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

#### 18 — Training loop ✅ **DONE**

**What was implemented:**

- **`decoder_lm_train_step(lm, input_ids, opt)`** in `src/transformer.c` — one
  training step with teacher forcing:

      1. `decoder_lm_forward` → logits [B, N, vocab]
      2. Slice logits[:, :-1, :] → [B, N-1, vocab]
      3. Build target tensor from input_ids[:, 1:] (int IDs, same int-in-float trick)
      4. `tensor_cross_entropy` over vocab dim (3D logits, 2D targets)
      5. `dnn_backward(loss)` — flows through slice view back to all params
      6. `adamw_step` + `adamw_zero_grad` — parameter update

  The slice view is autograd-wired: `tensor_slice` registers a `slice_backward`
  that scatters gradients from the view back to the parent. `cross_entropy_backward`
  writes through `_grad_ensure` (which walks to the root) directly into the
  original logits grad, and the autograd traversal continues through the
  decoder LM graph back to all leaf params.

**Tests added:**
- `test/ref_decoder_lm_training.py` — PyTorch reference: 3 steps training with
  `--small` config, outputs loss values and grad reference arrays
- `test/test_decoder_lm_training.c` — C tests:
  - Train step runs with finite loss
  - All parameter groups get grad buffers allocated
  - Parameters change after update (embedding max diff > 0)
  - Loss decreases over 3 steps (monotonic decrease)
  - Reference loss values match PyTorch within tolerance (random init differs)
  - Various batch/sequence shapes work (B=1/2/4, N=2/3/4/5)

**Files changed:**
- `include/transformer.h` — added `decoder_lm_train_step` declaration
- `src/transformer.c` — added `decoder_lm_train_step` implementation
- `test/ref_decoder_lm_training.py` — new PyTorch reference
- `test/test_decoder_lm_training.c` — new C test suite
- `docs/transformer_todo.md` — marked 18 done

---

#### 16 — `transformer_block` ✅ **DONE**

**What was implemented:**

- **`transformer_block` struct** in `include/transformer.h` — holds Q/K/V/output
  projections (`linear`), `n_heads`, `d_k`, `d_model`, pre-norm params for both
  attention and FFN sublayers (`attn_norm_weight/bias`, `ffn_norm_weight/bias`),
  and a `swiglu_ffn`.
- **`transformer_block_create(d_model, n_heads, d_k, intermediate_size)`** —
  allocates all params from params pool.  Asserts `d_model == n_heads * d_k`.
  Norm γ init to 1, β to 0.
- **`transformer_block_forward(block, x)`** — implements the pre-norm
  decoder-only architecture:

      1. Layer norm → Q/K/V projections → split heads
      2. Fused causal attention (`tensor_attention` with implicit masking)
      3. Merge heads → output projection → residual add
      4. Layer norm → SwiGLU FFN → residual add

  Input is `[B, N, d_model]` (3D), output same shape.  All sub-operations
  have autograd wired — backward flows gradients through the full graph.
  No extra mask needed (causal masking is implicit in `tensor_attention`).

**Tests added:**
- `test/ref_transformer.py` — PyTorch reference: full block forward + backward
  with seed=42.  Supports `--small` flag for compact test config.
- `test/test_transformer.c` — C tests: block creation shapes, all 19 param
  groups get non-finite grads, numerical gradient check (finite diff on input),
  no-grad mode, 2-block autograd chain, batch B=2, seq len N=1 and N=7,
  forward+backward structural validation, approximate forward match with
  weight-synced PyTorch reference.

**Files changed:**
- `include/transformer.h` — added `transformer_block` struct + function decls
- `src/transformer.c` — added `transformer_block_create`, `transformer_block_forward`
- `test/ref_transformer.py` — new PyTorch reference
- `test/test_transformer.c` — new C test suite

---

#### 17 — Decoder-only LM ✅ **DONE**

**What was implemented:**

- **`decoder_lm` struct** in `include/transformer.h` — holds embedding table
  (`[vocab_size, d_model]`), array of `transformer_block` pointers, final
  layer norm params, and `lm_head` linear projection (`d_model → vocab_size`).
- **`decoder_lm_create(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)`**
  — allocates all parameters from params pool.  Embedding and lm_head are
  separate tensors (no weight tying by default).  Norm γ init to 1, β to 0.
- **`decoder_lm_forward(lm, input_ids)`** — full autoregressive LM forward:

      1. Flatten [B, N] → [B*N], embed → [B*N, d_model], reshape → [B, N, d_model]
      2. Pass through all `n_layers` transformer blocks
      3. Final layer norm
      4. LM head: linear([B, N, d_model]) → [B, N, vocab_size]

  Input `input_ids` is a 2D `[B, N]` int tensor (data stored as `int*` in the
  float region, same pattern as cross_entropy target).  Must be contiguous.
  Returns `[B, N, vocab_size]` float logits.  Autograd wired through the
  entire graph — backward flows gradients to all parameters.

**Tests added:**
- `test/ref_decoder_lm.py` — PyTorch reference: full decoder LM forward +
  backward with seed=42.  Supports `--small` flag.
- `test/test_decoder_lm.c` — C tests: create shapes, forward shape + all 21
  param groups get finite grads, embedding gradient sparsity (only looked-up
  rows get non-zero grad), numerical gradient check (finite diff on lm_head
  weight), no-grad mode, batch B=2, seq len N=1 and N=7, weight tying
  verification (separate tensors), duplicate ID accumulation.

**Files changed:**
- `include/transformer.h` — added `decoder_lm` struct + `decoder_lm_create`/
  `decoder_lm_forward` declarations
- `src/transformer.c` — added `decoder_lm_create` and `decoder_lm_forward`
- `test/ref_decoder_lm.py` — new PyTorch reference
- `test/test_decoder_lm.c` — new C test suite

---

### Phase 4 — Full Model

| # | Item | Depends on | Lines |
|---|------|------------|-------|
| 16 | `transformer_block` (pre-norm attn + pre-norm swiglu-ffn + residual) transformer.c | (3,6,7,9,11,15a) | ~60 | ✅ **DONE** |
| 17 | Decoder-only LM (embed → N×block → norm → lm_head) | (4,16) | ~60 | ✅ **DONE** |
| 18 | Training loop (next-token prediction, teacher forcing, cross-entropy) | (17) | ~100 | ✅ **DONE** |
| 19 | Generation loop (autoregressive, kv-cache optional) | (17,15b,15c) | ~80 | ✅ **DONE** |
| 20 | Gradient clipping (L2 norm + value) | (18) | ~80 | ✅ **DONE** |

### Total estimated new code

- **C source:** ~1300–1500 lines across `src/`, `include/
- **Python:** none needed
- **Tests:** ~400 lines (one test per new op, plus integration test for training loop)
- **No new dependencies** — Accelerate BLAS + libm + zlib already in the Makefile.

#### 19 — Generation loop (autoregressive, kv-cache optional) ✅ **DONE**

**What was implemented:**

- **`transformer_block_forward_cached(block, x, cache)`** — forward one token
  through a single block using KV-cache:

  1. Pre-norm + QKV projection + split heads (`Qh`, `Kh`, `Vh` all `[B, H, N_new, d_k]`)
  2. `kv_cache_append(cache, Kh, Vh)` — store new K/V in the pre-allocated cache
  3. `kv_cache_get_K` / `kv_cache_get_V` — slice views of full cached `[B, H, S, d_k]`
  4. Inline scaled dot-product attention: `scores = Q @ K_full^T * scale` with
     cblas_sgemm, softmax over last dim (no causal mask — single new token
     attends to all past tokens), then `O = P @ V_full`
  5. Merge heads + output projection + residual
  6. Pre-norm + SwiGLU FFN + residual

  No autograd wired (generation runs in `dnn_no_grad` mode).
  Supports batch > 1 (though generation currently only uses batch=1).
  Uses BLAS (cblas_sgemm) with NO_CBLAS fallback.

- **`_argmax(logits, vocab_size)`** — picks token with highest logit
- **`_sample_with_temp(logits, vocab_size, temp)`** — categorical sampling
  with temperature: softmax(logits / temp), then sample from CDF via `rand()`

- **`decoder_lm_generate(lm, prompt_ids, max_new_tokens, temperature,
   use_cache, &n_out)`** — full autoregressive generation:

  **No-cache path:**
  - Loop: build tensor from accumulated output, call `decoder_lm_forward` to
    get logits for all positions, extract last token's logits, sample
  - O(N²) per step — full forward pass every iteration

  **KV-cache path:**
  - Create `kv_cache` per layer (pre-allocated to max_seq = prompt_len +
    max_new_tokens)
  - Process prompt tokens one-by-one to populate cache (embed → cached block
    forward for each layer)
  - On last prompt token: sample next token from logits
  - Then loop: sample, embed, cached block forward, final norm + lm_head
  - Each step is O(1) in sequence length — only processes the single new token

  Generation stops on EOS (ID 258) or max_new_tokens.
  Runs in `dnn_no_grad` — no autograd tape created.
  Returns int array from data pool (caller resets data pool to free).

**Scratch pool management:**
- Both cached and non-cached paths now reset `_mem_pool_scratch()` between
  iterations — no O(N²) scratch accumulation during generation.
- Each generation step: forward pass fills scratch, logits row copied to
  data pool buffer, scratch reset, then next step begins fresh.
- Non-last prompt tokens in cached path also reset scratch between steps.

**RoPE support:**
- `decoder_lm_generate` works correctly with RoPE enabled.
- Cached path applies RoPE at the correct position offset (`cache->seq_len`)
  for each token via `tensor_slice` of freq tables.  Non-cached path applies
  RoPE from position 0 (full sequence).  Both produce identical results.

**Tests added:**
- `test/ref_generation.py` — PyTorch reference: creates decoder LM, trains 5
  steps, generates with argmax + temperature, with/without cache, short prompt.
  Verifies cached == non-cached outputs match exactly.
- `test/test_generation.c` — C tests (8 tests):
  - Argmax no-cache: finite tokens in vocab range, prompt prefix preserved
  - Cached vs non-cached equivalence (multiple seeds/configs)
  - Short prompt (N=1): both paths match
  - Max new tokens limit respected
  - Deterministic: same seed produces same output
  - Temperature sampling: structurally valid (may match argmax)
  - Various model configs: cached/non-cached match across architectures
  - No-grad mode: generation doesn't accumulate grads
- `test/ref_generation_prefix.py` — PyTorch reference (5 cases):
  - No RoPE, short prompt (N=3): cached == non-cached
  - RoPE, short prompt (N=3): cached == non-cached
  - RoPE, long prefix (N=8): cached == non-cached
  - RoPE, single token (N=1): cached == non-cached
  - No RoPE, long prefix (N=8): cached == non-cached
- `test/test_generation_prefix.c` — C tests (8 tests):
  - RoPE + short prompt (N=3): cached == non-cached
  - RoPE + long prefix (N=8): cached == non-cached
  - RoPE + single token (N=1): cached == non-cached
  - No RoPE + long prefix (N=8): cached == non-cached
  - RoPE + various model configs (4 configs)
  - RoPE + max new tokens limit
  - RoPE + EOS stopping (vocab includes EOS ID)
  - Prefix exact preservation (6-token prefix, RoPE)

**Files changed:**
- `include/transformer.h` — added `transformer_block_forward_cached`,
  `decoder_lm_generate` declarations
- `src/transformer.c` — added `transformer_block_forward_cached`, `_argmax`,
  `_sample_with_temp`, `decoder_lm_generate`.  Added `#include "pool_int.h"`
  for `_mem_pool_scratch()`.  Added scratch pool reset between iterations
  in both cached and non-cached paths.
- `test/ref_generation.py` — new PyTorch reference
- `test/test_generation.c` — new C test suite
- `test/ref_generation_prefix.py` — new PyTorch reference (RoPE + prefix)
- `test/test_generation_prefix.c` — new C test suite for prefixes + RoPE
- `docs/transformer_todo.md` — marked 19 done, then refined with scratch
  management + RoPE + prefix tests

---

#### 20 — Gradient clipping (L2 norm + value) ✅ **DONE**

**What was implemented:**

Two gradient clipping functions in `optim.h`/`optim.c`:

- **`clip_grad_norm(params, n_params, max_norm)`** — L2 norm clipping. Computes
  total L2 norm of all gradients across all params. If `total_norm > max_norm`,
  scales all gradients by `max_norm / total_norm`. Returns the total norm BEFORE
  clipping (for logging). No-op if `max_norm <= 0`. Uses `double` accumulator
  for the norm sum to improve precision.

- **`clip_grad_value(params, n_params, clip_value)`** — element-wise value
  clipping. Clamps each gradient element to `[-clip_value, clip_value]`.
  No-op if `clip_value <= 0`.

**Integration into training step:**

- `decoder_lm_train_step(lm, input_ids, opt, grad_clip)` now accepts a
  `grad_clip` float parameter. If `> 0`, calls `clip_grad_norm` on the
  optimizer's params after `dnn_backward` and before `adamw_step`.
- `main_lm.c` passes `grad_clip=1.0f` to the training step.
- All existing callers updated to pass `0.0f` (no change in behavior).

**Tests added:**
- `test/ref_grad_clip.py` — PyTorch reference: clip norm basic (large grads
  scaled), no-op when norm < max_norm, extreme clip (norm reduced to exactly
  max_norm), value clip, max_norm=0 no-op, training with/without clip comparison.
- `test/test_grad_clip.c` — C tests (10 tests):
  - Norm clip basic: norm reduced to <= max_norm
  - Norm clip no-op: norm < max_norm → unchanged
  - Norm clip zero/negative max_norm: no-op
  - Norm clip extreme: very small max_norm works
  - Norm clip return value: returns norm before clipping
  - Value clip basic: max abs grad reduced to <= clip_value
  - Value clip no-op: large clip_value doesn't change grads
  - Value clip zero/negative: no-op
  - Clip in training step: loss finite and positive
  - Multi-step training with clip: loss decreases monotonically

**Files changed:**
- `include/optim.h` — added `clip_grad_norm`, `clip_grad_value` declarations
- `src/optim.c` — added `clip_grad_norm`, `clip_grad_value` implementations
- `include/transformer.h` — added `grad_clip` param to `decoder_lm_train_step`
- `src/transformer.c` — added clipping call in `decoder_lm_train_step`
- `main_lm.c` — passes `grad_clip=1.0f`
- `test/test_grad_clip.c` — new C test suite
- `test/ref_grad_clip.py` — new PyTorch reference
- `docs/transformer_todo.md` — marked 20 done

### Architectural considerations

- **KV cache** — essential for efficient autoregressive generation. Store K and V tensors per layer, append new tokens' K/V at each step. Needs `tensor_slice` for write (or a `tensor_cat` op along seq dim). `tensor_cat` would be a new op (concatenate along dim, with autograd). Alternatively, pre-allocate max-seq-length cache and slice into it — no new op, just pointer arithmetic.
- **Pre-norm vs post-norm** — `tensor_layer_norm` exists and supports arbitrary ndim. Pre-norm (layer norm before each sublayer, as used in Llama/GPT-NeoX) is slightly simpler to wire: `x = x + sublayer(norm(x))`.
- **Weight tying** — embedding weight = lm_head weight. Easy: reuse the same `tensor*` pointer in both `linear_create` (for lm_head) and the embedding table. Autograd accumulates grads to the same buffer.
- **Scratch pool sizing** — transformer activations per layer: attn ~4×[B,N,d] (Q,K,V,output) + FFN ~3×[B,N,4d] (gate, up, down). For batch=64, N=512, d=512: ~30MB per layer × N layers. With 12 layers this is ~360MB. Current scratch pool is 192MB — **needs bump to 512MB–1GB** for a full transformer.
