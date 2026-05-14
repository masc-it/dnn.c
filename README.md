# dnn.c — Deep Neural Network library in C

From-scratch DNN training framework in C11. Implements autograd, optimizers,
conv/linear layers, and MNIST digit classification. Zero heap allocations
during training — all memory managed via bump allocators.

## Architecture

### Memory Model

3 bump allocators (`mem_pool`), no `free` — one `free` per pool at destroy.

| Pool | Reset? | Contents |
|------|--------|----------|
| `params` | Never | Weights, biases, grads, optimizer state, tensor structs |
| `scratch` | Each step | Activations, saved tensors for backward, `grad_fn` nodes |
| `data` | Per batch | Raw file bytes, decoded images, batch float tensors |

### Tensor

Row-major N-D array, up to 8 dims. Supports views (slice/transpose/reshape),
strides, `requires_grad` flag, grad buffer, and `grad_fn` pointer for autograd tape.

### Autograd

Topological sort via DFS from loss backward. Each op registers a `grad_fn`
with a `backward` callback. `dnn_backward()` traverses in reverse topo order,
accumulates gradients into `t->grad` buffers. Thread-local grad mode
(`dnn_no_grad_enter`/`exit`) for eval.

## Ops

| Category | Functions |
|----------|-----------|
| Element-wise | `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, `tensor_neg`, `tensor_pow` |
| Matrix | `tensor_matmul` (BLAS via Accelerate on macOS, fallback manual) |
| Activation | `tensor_relu`, `tensor_softmax` |
| Regularization | `tensor_dropout` (inverted dropout) |
| Reduction | `tensor_sum`, `tensor_mean` |
| Loss | `tensor_cross_entropy` (fused softmax + NLL, numerically stabilized) |
| Normalization | `tensor_layer_norm` (last-dim, learnable γ/β) |
| Convolution | `tensor_conv2d` (im2col + GEMM, bias, stride, padding) |

All element-wise ops support NumPy-style broadcasting with fast-paths for
same-shape contiguous inputs.

## Layers / Models

- **`linear`** — `y = x @ W + b`, uniform init, autograd wired
- **`mnist_model`** — MLP: 784 → 256 → 10, ReLU + dropout(0.2)
- **`mnist_model_cnn`** — 3 conv layers (1→32→64→64, 3×3 kernels) + FC 3136→128→10, ReLU + dropout(0.5)

## Optimizers

- **SGD with momentum** — velocity buffers, momentum=0 → plain SGD
- **AdamW** — decoupled weight decay, bias-corrected moments, matches PyTorch exactly

## Training Pipeline (`mnist.c`)

- Downloads MNIST via `curl` + `gunzip` (4 IDX files)
- Loads images as float tensors [N,784] ∈ [0,1], labels as int tensors [N]
- Batched training loop with shuffling, early stopping (patience-based)
- AdamW optimizer, cross-entropy loss, batch=128, lr=0.001
- Shared `mnist_train_impl()` used by both MLP and CNN paths

## Optimizations

- NEON SIMD for ReLU fwd/bwd, softmax, cross-entropy, fast `expf` polynomial
- OpenMP parallel for im2col/col2im, layer norm, reductions
- (K,M) column layout in conv — sequential im2col writes, `CblasTrans` GEMM
- Forward col buffer reused in conv backward (saves 1 im2col per step)
- Bias fused into sgemm `beta=1.0f`
- Precomputed broadcast offsets in element-wise backward ops
- Contiguous fast-paths in `tensor_sum`, dropout, softmax, layer norm
- No `malloc`/`realloc` in hot path — all scratch pool allocations
- Bounds-peeled im2col/col2im (no boundary checks in hot loop)

## Project Structure

```
include/         — Public API headers (tensor.h, ops.h, nn.h, conv.h, ...)
src/             — Implementation (.c + internal headers)
test/            — C unit tests + Python reference scripts
docs/            — Design spec, optimization audit, coding rules
main.c           — Entry point: CNN training on MNIST
Makefile         — Build: static lib, tests, benchmarks
```

## Build

```sh
make            # builds libdnn.a + test binaries
make run        # builds + runs MNIST CNN training
make test       # runs all unit tests
make bench_all  # conv2d/matmul/ops benchmarks
```

Requires: Apple Accelerate (or cblas), libz, libomp.

## Design Principles

- `_` prefix = internal API
- Comments explain WHY, not WHAT
- Test behavior, not implementation
- Interfaces are deep — few powerful operations, not many trivial ones
- All pooled, no raw `malloc` during training
