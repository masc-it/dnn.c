# CNN Training Optimization Audit

Report generated 2026-05-13.  Last updated 2026-05-13.

## Scope

Review `mnist_train_cnn` in `src/mnist.c` as called from `main.c`. Goal: identify
why CNN training fails to finish within 120 seconds while MLP completes in ~20s
with early stopping.

---

## 1. CNN NOT WIRED IN MAIN.C

`main.c` never calls `mnist_model_create_cnn()` or `mnist_train_cnn()`. Only MLP
path runs. CNN training code exists in `src/mnist.c` but is unreachable from the
default binary.

**Severity**: blocker (fixed)

---

## 2. SCRATCH POOL SIZE (CRITICAL)

`main.c` sets `scratch = mem_pool_create(10 * 1024 * 1024)` = **10 MB**.

CNN forward pass with batch=64 allocates from scratch pool:

| Layer | Shape | Size |
|-------|-------|------|
| conv1 out | (64, 32, 28, 28) | 6.4 MB |
| relu1 out | (64, 32, 28, 28) | 6.4 MB |
| conv2 out | (64, 64, 14, 14) | 3.2 MB |
| relu2 out | (64, 64, 14, 14) | 3.2 MB |
| conv3 out | (64, 64, 7, 7) | 0.8 MB |
| relu3 out | (64, 64, 7, 7) | 0.8 MB |
| fc1 + relu + dropout + fc2 | minor | ~0.1 MB |
| cross-entropy intermediates | minor | ~0.1 MB |

**Peak ~21 MB > 10 MB** → assert in `_mem_pool_alloc` triggers → crash.

MLP works because activations total <200 KB.

**Severity**: blocker (fixed — bumped to 32 MB)

---

## 3. CONV2D USES RAW malloc (spec violation — FIXED)

Both `conv2d_forward` and `conv2d_backward` (`src/conv.c`) allocated im2col and
dcol buffers via `malloc()`/`free()`.  Replaced with scratch pool allocations.

Per-batch alloc sizes for batch=64 (now from scratch pool, no syscall):

| Conv layer | K × M | Size |
|------------|-------|------|
| conv1 (1→32) | 9 × 50176 | ~1.8 MB |
| conv2 (32→64) | 288 × 12544 | ~14.5 MB |
| conv3 (64→64) | 576 × 3136 | ~7.2 MB |

---

## 4. RELU ALLOCATES NEW TENSOR (NOT IN-PLACE — FIXED)

`tensor_relu` now writes into the input buffer in-place.  Saves ~10.4 MB per
batch in activation memory and avoids a full copy.

---

## 5. COMPUTE INTENSITY 30× MLP

Rough FLOP estimate per batch (batch=64):

| Op | Shape | FLOPS (fwd) |
|----|-------|-------------|
| fc1 matmul (MLP) | (64,784)×(784,256) | 25M |
| fc2 matmul (MLP) | (64,256)×(256,10) | 0.3M |
| **MLP total** | | **~26M fwd, ~52M fwd+bwd** |
| conv1 matmul | (50176,9)×(9,32) | 29M |
| conv2 matmul | (12544,288)×(288,64) | 462M |
| conv3 matmul | (3136,576)×(576,64) | 231M |
| fc1 matmul | (64,3136)×(3136,128) | 51M |
| fc2 matmul | (64,128)×(128,10) | 0.2M |
| **CNN total** | | **~770M fwd, ~2.3B fwd+bwd** |

Over 5 epochs (typical early stop) × 938 batches:

| Model | Total FLOPS | Est. time (50 GFLOPS) |
|-------|-------------|----------------------|
| MLP | ~245B | ~5 s |
| CNN | ~10.8T | **~216 s** |

Apple Silicon with Accelerate achieves 50-100 GFLOPS sustained for these matrix
sizes.  At 120-second timeout, CNN training is borderline even with perfect
overhead-free ops.

**Severity**: architecture (inherent, mitigated by batch size tuning and
im2col-less backward).

---

## 6. CROSS-ENTROPY 3-PASS OVER LOGITS

`tensor_cross_entropy` does 3 passes over all logits (max, sum_exp, loss).
Minor.

---

## 7. NO BATCH NORM

CNN has no batch normalization.  Standard MNIST CNN architectures include batch
norm after each conv for training stability and faster convergence.

**Severity**: quality / convergence speed

---

## Optimization history

| Date | Optimization | Per batch | Per epoch | 5 epochs | Speedup vs baseline |
|------|-------------|-----------|-----------|----------|---------------------|
| — | Baseline (pools + wiring) | ~140 ms | ~60 s | ~300 s | 1.0× |
| — | + relu in-place / `_flat_off` | ~72 ms | ~31 s | ~155 s | 1.9× |
| — | + OpenMP im2col/col2im | ~53 ms | ~23 s | ~113 s | 2.6× |
| — | + loop-interchanged im2col | ~44 ms | ~19 s | ~95 s | 3.2× |
| 2026-05-13 | + (K, M) col layout + col2im restructure | see below | | | |
| 2026-05-13 | – OpenMP (removed per request) | see below | | | |

Current measured perf (via `test_cnn_stress`, batch=128):

| Metric | Value |
|--------|-------|
| Fwd only | 12.14 ms |
| Bwd only | 34.18 ms |
| Total fwd+bwd | 46.33 ms |
| Epoch estimate (430 batches) | ~24.8 s |
| 5 epochs estimate | ~124.0 s |

---

## (K, M) col layout (2026-05-13)

### Problem

`im2col` stored col as (M, K) where M = N×H_out×W_out and K = C×kH×kW.
The inner loop wrote `col[co + ow * K]` — every output-column index
advanced by K floats.  For conv2 (K=288) that's a 1152-byte stride between
consecutive writes — ~90% of each cache line wasted on partial writes.

### Fix

Changed layout to (K, M):

* **im2col** writes `col[idx + ow]` — sequential floats per kernel element.
  Input reads stay sequential (same loop order).  Both sides hit cache lines
  at 100% efficiency.
* **col2im** restructured to (c, kh, kw, n, oh, ow) loop order — dcol reads
  are now sequential along M (the spatial dimension) per kernel element,
  instead of strided by K.
* **All GEMMs** use `CblasTrans, CblasTrans` — both matrices stay in native
  row-major order; BLAS handles transposition virtually via lda/ldb.

### Cache audit (updated for (K, M))

| Component | BW efficiency | Notes |
|-----------|--------------|-------|
| im2col read | ~100% | sequential per row |
| im2col write | ~100% | sequential per kernel element |
| col2im read | ~100% | sequential along M per kernel element |
| col2im write | inherent scatter | NCHW layout, stride W between kernel rows |
| sgemm (hot, col fits L2) | 663 GFLOPS | AMX coprocessor |
| sgemm (cold, col > L2) | 221 GFLOPS | 3× penalty when col misses L2 |

### Comparison with PyTorch

PyTorch's im2col fallback uses the same (K, M) convention (`col is
(out_channels, K)` in their naming which maps to (K, M) here).  The GEMM
becomes `weight @ col` with both in native layout.

Modern PyTorch on CPU avoids im2col entirely via oneDNN (Intel) or MPS
(Apple Silicon), which use:

* **Direct convolution** with blocked memory formats — no column expansion.
  Each input tile reused across output channels before eviction.
* **Winograd F(2×2, 3×3)** for the common 3×3 kernel — 4 MACs per position
  instead of 9.
* **Implicit GEMM** on GPU — never materializes col, computes indices
  on-the-fly through texture cache.

---

## Next optimizations (priority order)

### P1 — Save forward col buffer for backward d_weight

**Impact: 1 im2col saved per conv layer per step** (3 saves/batch in MNIST
CNN).

Currently `conv2d_backward` recomputes `im2col` for d_weight even though
forward just produced the same col.  With (K, M) layout, forward's col is
still live in scratch pool when backward runs — just don't release it.
d_weight reuses the buffer directly, saving the entire im2col pass + memset.

**Effort**: ~10 lines.  Store the `mem_pool_mark` from forward in
`grad_fn->saved_tensors`, release it after backward d_weight consumes col.

**Expected gain**: ~20-30% reduction in per-batch time (dominated by im2col
when col is large).

### P2 — Peel pad boundary in im2col / col2im

**Impact: eliminate bounds checks for ~90% of positions**.

Every inner-loop iteration checks `if (ih < 0 || ih >= H)` and `if (iw >= W)`.
For stride=1, pad=1 — only ~6% of positions are on the boundary.  Split
the output grid into top pad, interior (no checks), bottom pad.

**Effort**: ~30 lines in im2col + col2im.

**Expected gain**: ~10% per im2col/col2im call.

### P3 — Winograd F(2×2, 3×3) for 3×3 kernels

**Impact: 2.25× fewer multiplies** for 3×3 kernels (conv2, conv3 in MNIST CNN).

Transforms the 3×3 convolution into 4 element-wise multiplies with small
O(kH×kW) transform matrices applied to input tiles and weight.  The
transforms are cheap (~20 O(HW) additions per kernel) compared to the GEMM
savings.

Requires separate forward/backward paths for kH=kW=3.

**Effort**: ~100 lines.  New functions, dispatch in `tensor_conv2d`.

**Expected gain**: ~40% on 3×3 conv GEMM time.

### P4 — Fuse bias add into sgemm beta

**Impact**: eliminate separate bias loop (trivial, ~1% perf).

Pre-fill od with bias values so the GEMM `beta=1.0f` adds bias via the
accumulator instead of a separate pass.

---

## Remaining architecture limits

* conv2 d_weight backward uses `cblas_sgemm(out_C=64, K=288, M=12544)` —
  the moderate M limits BLAS parallelism to ~665 GFLOPS vs 1152 GFLOPS for
  forward.  Inherent to im2col-based backward; would require direct
  convolution or Winograd to fully eliminate.
* 3× penalty when col > L2 cache (~12 MB on M1/M2) — affects conv2 col at
  14.5 MB.  Winograd avoids this entirely since it never builds the col matrix.
