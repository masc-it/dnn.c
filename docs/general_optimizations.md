# Under-Optimized Hot Paths

## 1. ReLU allocates new output — biggest low-hanging fruit

`tensor_relu()` (`ops_activation.c:40`) calls `_tensor_scratch_create` for output even when input doesn't need the original buffer preserved. For a 3-conv MNIST CNN, that's 3 extra tensors ~10.4MB written per batch. Fix: detect when input is non-leaf intermediate (no grad needed on original data) and write in-place into the input buffer. Already documented as "implemented: false" in `cnn_optim.md`.

**implemented: true -> reverted** — `tensor_relu` now modifies the input data buffer in-place and returns a lightweight view tensor (24 bytes on scratch pool) sharing that buffer. `relu_backward` derives the mask from the output values (`out[i] > 0` is equivalent to `in[i] > 0`), so no saved tensors are needed. The view preserves the prior op's `grad_fn` chain for correct topological sort traversal. REVERT REASON: `make run` batch/s went from 32/s -> 26/s.

## 2. Cross-entropy forward: 3 passes over logits (nD path only)

The 2D fast path (`ndim==2 && dim==1 && contiguous`) was already **2 row reads** (max via SIMD, sum_exp via SIMD) + 1 scalar access (`row[td[n]]`). The "Pass 3" comment was misleading — it's not a full pass.

The **general nD path** did 3 full coord-decompose passes (max, sum_exp, loss). Fused to 2 passes by saving the target logit value during the sum_exp pass and computing loss per-slice afterward. Saves ~33% of index decomposition work on the nD path.

**Status: done.** 2D path left as-is (already optimal). nD path passes reduced 3→2. No measurable impact on MNIST CNN benchmark (uses 2D path).

## 3. Softmax backward: coord decomposition done twice

In `softmax_backward()` (`ops_activation.c:84-150`), both loops independently decompose flat index → `coord[DNN_MAX_DIMS]` → `slice_idx`. This is 2× the div/mod work. Fuse into one loop.

## 4. Element-wise backward broadcast paths: coord decompose per element ×2

`add_backward`, `sub_backward`, `mul_backward`, `div_backward` — all have a broadcast fallback that decomposes flat index → coord for **each input tensor separately** via `_bcast_off`. This is the full coord decompose + stride multiply per element, per input. For non-broadcast but non-contiguous cases, this is pure overhead that strided loops would avoid.

## 5. `_bcast_off` fast-path too narrow

The fast path in `_bcast_off()` (`broadcast.h:38-46`) only fires when `t->contiguous && t->ndim == out_ndim && no dim has shape==1`. The common case of "contiguous, same ndim, no broadcast" is fast. But "contiguous, same ndim, shape-1 broadcast" falls through to the slow per-element stride loop even though the index computation is just `idx = flat_index % i` with zero-stride for 1-sized dims. Could add a second fast path.

## 6. Dropout forward/backward: coord decompose on contiguous tensors

`tensor_dropout` and `dropout_backward` decompose flat index → coord → `_bcast_off` for **every element** even when the input is contiguous. Dropout inputs are almost always contiguous (activations). Should use flat indexing: `float *tp = td + t->offset` and iterate linearly.

## 7. `tensor_sum` forward: coord decompose per output element

`tensor_sum()` decomposes each output index → coord, then runs an inner loop over the reduction dimension modifying `coord[dim]` and calling `_bcast_off`. For contiguous tensors, this should use strided pointers — a `memcpy`-style loop with stride `t->strides[dim]` skipping over the reduction dimension. The current approach is O(numel × dim_size) in div/mod operations.

## 8. Conv2d backward d_bias: nested scalar loops

Four nested loops (`oc, n, oh, ow`) summing `gd[...]` per output channel (`conv.c:170-178`). Could be a `cblas_sgemv` with a ones-vector: `d_bias = gd^T @ 1`. Or at minimum `#pragma omp simd reduction(+:acc)` on the inner loop instead of the outer-parallel + scalar reduction.

## 9. Layer norm backward: `dy*weight` computed twice per element

In `ln_backward()`, the expression `gd[s*d+j] * (wd ? wd[j] : 1.0f)` is computed once in the reduction loop (sum_dy, sum_dy_xmu) and again in the final dx loop (`norm.c:62` vs line 73). Could compute once and keep in a register / local array. With `#pragma omp simd` already there, the compiler may optimize this, but not guaranteed.

## 10. `dnn_backward` uses `realloc` — violates pool-only rule

The topological sort in `dnn_backward()` (`autograd.c:68`) grows via `realloc`, which calls heap alloc during training. Should use scratch pool with doubling strategy (`mem_scratch_alloc`) or pre-allocate based on graph depth estimate.

## 11. No vectorization on broadcast backward hot loops

The scalar fallback loops in `add_backward` / `sub_backward` / `mul_backward` / `div_backward` have no `#pragma omp simd`. The inner body has irregular memory access (via `_bcast_off`), so auto-vectorization fails and no hint is given to the compiler.

## 12. Softmax 2D+dim-1 fast path missing in backward

`softmax_backward` has **no** 2D contiguous fast path — only the general nD coord-decompose path. The forward does have one (and uses NEON). So softmax backward for a classifier (the most common use) is ~10× slower than it could be.
