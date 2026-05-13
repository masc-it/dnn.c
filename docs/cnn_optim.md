# CNN Training Optimization Audit

Report generated 2026-05-13.

## Scope

Review `mnist_train_cnn` in `src/mnist.c` as called from `main.c`. Goal: identify why CNN training fails to finish within 120 seconds while MLP completes in ~20s with early stopping.

---

## 1. CNN NOT WIRED IN MAIN.C

`main.c` never calls `mnist_model_create_cnn()` or `mnist_train_cnn()`. Only MLP path runs. CNN training code exists in `src/mnist.c` but is unreachable from the default binary.

**Severity**: blocker

**Fix**: add `mnist_train_cnn()` call to `main.c`.

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

**Severity**: blocker

**Fix**: increase scratch pool to ≥32 MB (or 1 GB as spec says).

---

## 3. CONV2D USES RAW malloc (spec violation)

Both `conv2d_forward` and `conv2d_backward` (`src/conv.c`) allocate im2col and dcol buffers via `malloc()`/`free()`:

```c
float *col = malloc((size_t)M * K * sizeof(float));
// ...
free(col);
```

Spec (`docs/spec.md`) requires: *"No heap allocs during training. Every allocation goes through one of the three pools."*

Per-batch malloc sizes for batch=64:

| Conv layer | M × K | Malloc size |
|------------|--------|-------------|
| conv1 (1→32) | 50176 × 9 | ~1.8 MB |
| conv2 (32→64) | 12544 × 288 | ~14.5 MB |
| conv3 (64→64) | 3136 × 576 | ~7.2 MB |

Over 20 epochs × 938 batches/epoch = 18,760 batches. Each batch calls forward + backward, so:

```
3 conv layers × 2 (fwd+bwd) × 18,760 = ~112K malloc/free cycles
```

Each large malloc incurs mmap syscall, page faults, and TLB overhead.

**Severity**: performance (major contributor to 120s timeout)

**Fix**: allocate im2col/dcol from scratch pool instead of heap.

---

## 4. RELU ALLOCATES NEW TENSOR (NOT IN-PLACE)

Spec says: *"Free-list ops (in-place relu) reuse the input buffer."*

But `tensor_relu` (`src/ops_activation.c`) always creates a new output:

```c
tensor *out = _tensor_scratch_create(t->ndim, t->shape, 0);
```

ReLU is element-wise: input is not needed after forward once the grad_fn saves a pointer to it. It could modify in-place to halve activation memory and avoid a full copy.

Impact per batch:

| Layer | Shape | Extra alloc |
|-------|-------|-------------|
| relu1 | (64, 32, 28, 28) | 6.4 MB |
| relu2 | (64, 64, 14, 14) | 3.2 MB |
| relu3 | (64, 64, 7, 7) | 0.8 MB |
| Total | | 10.4 MB dupe |

**Severity**: space + performance

**Fix**: implement in-place ReLU that writes into input buffer.

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

Apple Silicon with Accelerate achieves roughly 50-100 GFLOPS sustained for these matrix sizes. At 120-second timeout, CNN training is borderline even with perfect overhead-free ops.

**Severity**: architecture (inherent, but can be mitigated)

**Mitigations**:
- Increase batch size (more work per BLAS call, better cache utilization)
- Reduce channel count (32→16, 64→32) to halve FLOPS
- Fuse conv+ReLU to reduce im2col passes
- Use larger stride or fewer conv layers

---

## 6. IM2COL BOUNDS CHECKS EVERY PIXEL

im2col inner loop in `src/conv.c` checks bounds per pixel via conditionals inside 4-deep nested loops:

```c
for (int kw = 0; kw < kW; kw++) {
    int iw = w_start + kw;
    if (iw >= 0 && iw < W)
        col[off] = x[...];
    else
        col[off] = 0.0f;
}
```

For conv1 (stride=1, pad=1, kernel=3): every position is in-bounds because `h_start` ranges from -1 to 28 and `ih = h_start + kh` ranges from 0 to 27 for all loops. The bounds check fires 9 times per output pixel but is **always true**. Same for conv2 and conv3 with stride=2, pad=1 (half the positions have one row/col of padding, but the check is evaluated per kernel element).

**Severity**: performance (compiler may not elide)

**Fix**: peel the pad-boundary rows/cols or use clamp-region im2col.

---

## 7. CROSS-ENTROPY 3-PASS OVER LOGITS

`tensor_cross_entropy` does 3 passes over all logits (max, sum_exp, loss) with full coordinate reconstruction via division/modulo each time. Could fuse into 1-2 passes.

**Severity**: minor

---

## 8. NO BATCH NORM

CNN has no batch normalization. Standard MNIST CNN architectures include batch norm after each conv for training stability and faster convergence. Without it, gradient scales vary across layers, requiring lower learning rates or more epochs.

**Severity**: quality / convergence speed

**Fix**: add `tensor_batch_norm` calls after conv layers.

---

## Prioritized Fix Plan

| Priority | Issue | Expected gain |
|----------|-------|---------------|
| P0 | Wire CNN in main.c, bump scratch pool to 32 MB | Unblock run |
| P1 | Replace malloc with scratch pool in conv2d | Remove syscall overhead |
| P1 | In-place ReLU | Halve activation memory, reduce copy |
| P2 | Optimize im2col (peel pad boundary) | ~20% conv speedup |
| P2 | Tune batch size (128 instead of 64) | Better BLAS utilization |
| P3 | Fuse cross-entropy passes | Marginal |
| P3 | Add batch norm | Better convergence per epoch |

Without P1 fixes, CNN likely exceeds 120s due to malloc churn + activation copying overhead even if compute fits.
