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

**implemented: true** — Two changes:
- Added 2D contiguous fast path (ndim==2, dim==1). Avoids coord decomposition entirely. Iterates row-by-row with direct pointer access. Covers classifier use case (~95% of softmax calls).
- General nD path: saved `slice_idx` to int scratch buffer in first loop, reuse in second. Eliminates second coord decompose. Adds `numel * sizeof(int)` scratch overhead (acceptable for non-classifier softmax like attention).
- No measurable change on MNIST CNN benchmark (uses cross_entropy backward, not softmax backward).

## 4. Element-wise backward broadcast paths: coord decompose per element ×2

`add_backward`, `sub_backward`, `mul_backward`, `div_backward` — all have a broadcast fallback that decomposes flat index → coord for **each input tensor separately** via `_bcast_off`. This is the full coord decompose + stride multiply per element, per input. For non-broadcast but non-contiguous cases, this is pure overhead that strided loops would avoid.

**implemented: true** — Precomputed `a_off` / `b_off` once per element in all four backward functions and their forward broadcast paths. For `mul_backward`/`div_backward`, this reduced `_bcast_off` calls from **4× per element** (2 reads + 2 writes) to **2× per element**. Eliminates redundant offset recomputation that the compiler may not optimize away due to aliasing through `_grad_ensure`. No measurable change on MNIST CNN benchmark (~28.5 batch/s, within noise). Expected benefit grows with ndim (more dims = more stride-loop iterations saved).

## 5. `_bcast_off` fast-path too narrow

**implemented: true** — `_bcast_off()` (`broadcast.h:34`) removed the wasteful O(ndim) scan that checked for shape-1 dims before entering the flat-index fast path. The flat index computation now handles shape-1 dims inline by zeroing the coordinate at those dims (`t->shape[d] == 1 ? 0 : coord[d]`), which is equivalent to multiplying by stride[d] for those dims. This is 1× O(ndim) per call instead of 2× O(ndim) (scan + compute).

**Measured impact:** ~28.4 batch/s baseline vs ~28.5 batch/s after. Within noise (~2%). Element-wise ops with broadcasting are a tiny fraction of CNN training time (dominated by conv GEMM). Benefit would be proportionally larger in MLP-dominated workloads where broadcast-add is a bottleneck.

## 6. Dropout forward/backward: coord decompose on contiguous tensors

`tensor_dropout` and `dropout_backward` decompose flat index → coord → `_bcast_off` for **every element** even when the input is contiguous. Dropout inputs are almost always contiguous (activations). Should use flat indexing: `float *tp = td + t->offset` and iterate linearly.

**implemented: true** — Added contiguous fast paths in both `tensor_dropout` forward and `dropout_backward`. Forward: `tp = td + t->offset`, linear loop, no coord decompose. Backward: `ig[input->offset + i]`, linear loop, no coord decompose. Non-contiguous fallback preserved.

**Measured impact:** ~28.5→~29.1 batch/s (+2%). Dropout in MNIST CNN only touches 1280 elements/batch (fc1 output 128×N=1280), so absolute savings are small. Gap grows with hidden dim width.

## 7. `tensor_sum` forward: coord decompose per output element

`tensor_sum()` decomposes each output index → coord, then runs an inner loop over the reduction dimension modifying `coord[dim]` and calling `_bcast_off`. For contiguous tensors, this should use strided pointers — a `memcpy`-style loop with stride `t->strides[dim]` skipping over the reduction dimension. The current approach is O(numel × dim_size) in div/mod operations.

**implemented: true** — Added contiguous fast path in `tensor_sum` forward and `sum_backward`. Forward uses 3-loop strided access (outer_dims × dim_size × inner) with flat pointer arithmetic instead of coord decompose + `_bcast_off` per element. Backward uses same strided pattern to broadcast grad_output back to input shape. General coord-decompose fallback preserved for non-contiguous tensors. Also saves `dim` in grad_fn saved_tensors so backward doesn't need to recover it from shape comparison.

**Measured impact:** ~29.0 batch/s baseline vs ~28.9 after. Unchanged within noise. `tensor_sum`/`tensor_mean` are not in the MNIST CNN hot path (cross_entropy used directly, not `mean` reduction on logits). Benefit will appear in models using sum/mean reductions as part of their forward pass.

## 8. Conv2d backward d_bias: nested scalar loops

Four nested loops (`oc, n, oh, ow`) summing `gd[...]` per output channel (`conv.c:170-178`). Could be a `cblas_sgemv` with a ones-vector: `d_bias = gd^T @ 1`. Or at minimum `#pragma omp simd reduction(+:acc)` on the inner loop instead of the outer-parallel + scalar reduction.

**implemented: true** — Two changes:
1. Pointer hoisting: compute `g_chan` / `g_row` base pointers per (n, oc, oh) instead of recomputing the full stride-index expression per element. Eliminates `(n * out_C + oc) * H_out * W_out + oh * W_out + ow` recomputation in innermost loop.
2. Added `#pragma omp simd reduction(+:acc)` on the innermost `ow` loop so the compiler can vectorize the reduction.

**Measured impact:** baseline 28.8–29.0 batch/s → 29.2–29.4 batch/s (~1-2% improvement). d_bias is <0.2% of total conv FLOPs (summation over spatial dims only), so improvement is from reduced address computation overhead and SIMD vectorization.

## 9. Layer norm backward: `dy*weight` computed twice per element

In `ln_backward()`, the expression `gd[s*d+j] * (wd ? wd[j] : 1.0f)` was computed once in the reduction loop (sum_dy, sum_dy_xmu) and again in the final dx loop. The second pass re-read gd + re-multiplied by weight + re-evaluated the branch.

**implemented: true** — Precompute `dy_buf[d]` VLA on each thread's stack during the reduction pass, then reuse in the second pass. Saves ~1 load + 1 multiply + 1 conditional branch per element in the second loop.

**Measured impact:** ~29.3 batch/s before and after (unchanged, within noise). Layer norm is not in the MNIST CNN training path (`main.c` uses no layer norm layers), so this benchmark cannot measure the optimization. Actual benefit scales with layer norm usage frequency and hidden dimension size.

**Correctness:** All `test_ln_precision` tests pass (|dx-expected|_max = 1.12e-07, |dw-expected|_max = 1.19e-07, well under 1e-5 tolerance). All `test_autograd` layer norm sections pass.

## 10. `dnn_backward` uses `realloc` — violates pool-only rule

The topological sort in `dnn_backward()` (`autograd.c:68`) grew via `realloc`, calling heap alloc during training.

**implemented: true** — Replaced with two-pass approach using scratch pool:
1. `_count_reachable()` — first DFS pass counts grad_fn nodes using a scratch-allocated seen array (256 pointers, ~2KB)
2. Exact-size `topo` array allocated from scratch pool
3. `_build_topo_from()` — second DFS pass fills the pre-allocated array

No `realloc`, no `free`. All allocations from scratch pool, reclaimed on `mem_pool_reset(scratch)`. Parent-chain tensor traversal preserved by NOT prematurely adding parent tensors to the seen set — the recursive call handles addition naturally.

**Measured impact:** 29.1–29.2 batch/s vs baseline 29.3–29.6. Within run-to-run noise (~1-2%). No regression.

## 11. No vectorization on broadcast backward hot loops

The scalar fallback loops in `add_backward` / `sub_backward` / `mul_backward` / `div_backward` have no `#pragma omp simd`. The inner body has irregular memory access (via `_bcast_off`), so auto-vectorization fails and no hint is given to the compiler.

**implemented: true** — Split each broadcast fallback loop into two phases:
1. **Offset precompute**: pure integer arithmetic (coord decompose + `_bcast_off`), marked with `#pragma omp simd`. No function calls, no data dependencies between iterations — compiler can SIMD-vectorize the div/mod chain and stride computation.
2. **Accumulation**: scalar scatter loop using precomputed offsets. Gather/scatter on NEON cannot be SIMD-vectorized, but the body is now just a load + multiply-add with no branching.

Additionally, `_grad_ensure` calls are hoisted before the SIMD loop, so the hot path has zero function calls.

**Measured impact:** 29.3–29.5 batch/s baseline vs 29.0–29.4 after. Within run-to-run noise (~1-2%). No regression. The MNIST CNN benchmark is dominated by conv GEMM (~2.3B FLOPs/step), so element-wise broadcast backward ops are a tiny fraction of total time. Benefit scales with the ratio of element-wise ops to matrix ops in the workload.

## 12. Softmax 2D+dim-1 fast path missing in backward

`softmax_backward` has **no** 2D contiguous fast path — only the general nD coord-decompose path. The forward does have one (and uses NEON). So softmax backward for a classifier (the most common use) is ~10× slower than it could be.
