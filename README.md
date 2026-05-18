# dnn.c — Deep Neural Network library in C

From-scratch DNN training framework in C11. ARM NEON + OMP.

Decided to finally give this a shot after ~2 years of studying and replicating from first principles what great Andrej Karpathy shared on the internet (sources at the bottom). Got tired of python and pytorch, was time to go deeper.

Thanks to this project, I truly realised how blessed we are to have pytorch and the people surroudning it.

The focus of this project for me was to put in practice the best low-level design practices I learnt about and instruct ds4-flash to implement them step-by-step (most of the time). 

I left ~0% room to the AI assistant for design choices, 90% freedom in implementation. As usual, pareto law applies here as well, the 10% implementation guidance brought the majority of longer-term benefits.

Up to now, the cost of this project is around $6. we average $1/day (opencode-go)

This model is truly amazing as a daily assistant in "pair-programming mode" (not yolo/vibes).

## Architecture

### Memory Model

3 bump allocators.

| Pool | Reset? | Contents |
|------|--------|----------|
| `params` | Never | Weights, biases, grads, optimizer state, tensor structs |
| `scratch` | Each step | Activations, saved tensors for backward, `grad_fn` nodes |
| `data` | Per batch | Raw file bytes, decoded images, batch tensors |

### Tensor

Row-major N-D array, up to 8 dims. Supports views (slice/transpose/reshape),
strides, `requires_grad` flag, grad buffer, and `grad_fn` pointer for autograd tape.

### Autograd

Topological sort via DFS from loss backward. Each op registers a `grad_fn`
with a `backward` callback. `dnn_backward()` traverses in reverse topo order,
accumulates gradients into `t->grad` buffers. Thread-local grad mode
(`dnn_no_grad_enter`/`exit`) for eval.

## Ops

- **Arithmetic** — `add`, `sub`, `mul`, `div` (element-wise, broadcasting)
- **Matrix** — `matmul`, `matmul_add` (fused matmul + bias, optional trans_b)
- **Activations** — `relu`, `sigmoid`, `tanh`, `silu`, `swiglu`, `softmax`, `causal_softmax`
- **Attention** — scaled dot-product attention with causal or prefix-LM masking, per-batch `seq_lens`
- **Normalization** — `layer_norm`, `rms_norm`
- **Convolution** — `conv2d` (im2col-based)
- **Pooling** — `avg_pool2d`
- **Embedding** — table lookup (`embedding`)
- **Reduction** — `sum`, `mean`
- **Loss** — `cross_entropy`, `cross_entropy_masked` (masked for variable-length batches)
- **Multi-head** — `split_heads`, `merge_heads`, `split_qkv_heads` (fused QKV split)
- **RoPE** — rotary position embedding, frequency table init
- **Regularization** — `dropout`
- **Tensor ops** — `cat`, `pow`, `neg`, `triu`

All element-wise ops support NumPy-style broadcasting with fast-paths for
same-shape contiguous inputs.

## Layers

- **Linear** — `x @ W + b` (fully-connected)
- **SwiGLU FFN** — SiLU-gated FFN with gate/up/down projections (Llama-style)
- **LayerNorm** — `γ * (x-μ)/√(σ²+ε) + β`
- **RMSNorm** — `x * rsqrt(mean(x²)+ε) * γ` (no bias)
- **Embedding** — trainable token/feature lookup table
- **Conv2D** — 2D cross-correlation with im2col/col2im
- **Transformer Block** — pre-norm decoder block: fused QKV, RoPE, causal/prefix-LM attention, SwiGLU FFN, residual connections, KV-cache for generation
- **Decoder LM (GPT)** — token embed → N×transformer_block → final RMSNorm → tied lm_head
- **Vision-LM (VLM)** — image patch embed + position embedding + decoder LM with prefix-LM attention (bidirectional image tokens, causal text tokens)

## Optimizers

- **SGD with momentum** — velocity buffers, momentum=0 → plain SGD
- **AdamW** — decoupled weight decay, bias-corrected moments, matches PyTorch exactly
- **Gradient clipping** — `clip_grad_norm` (L2), `clip_grad_value` (element-wise)
- **LR scheduler** — 6 schedule types:
  - `LR_SCHEDULE_CONSTANT` — fixed LR
  - `LR_SCHEDULE_LINEAR_WARMUP_COSINE` — linear warmup then cosine decay
  - `LR_SCHEDULE_LINEAR_WARMUP` — linear warmup then constant
  - `LR_SCHEDULE_COSINE` — cosine decay from base to min
  - `LR_SCHEDULE_STEP` — multiply by gamma every step_size iters
  - `LR_SCHEDULE_EXPONENTIAL` — multiply by gamma each step

## Training Pipelines

Under `examples/`:

- **MNIST MLP** — MLP classifier with AdamW, CE loss, early stopping, validation accuracy
- **MNIST CNN** — CNN classifier (conv+pool+linear) with AdamW, CE loss, early stopping
- **MNIST CNN+Pool** — CNN with avg_pool2d instead of max-pool, AdamW, CE loss
- **Promessi Sposi LM** — decoder-only transformer LM trained on Italian literature, AdamW, cosine LR with linear warmup, RoPE, periodic text generation eval
- **MNIST VLM** — tiny prefix-LM VLM classifying digits via caption generation, AdamW, gradient clipping, cosine LR with warmup, checkpointing, early stopping
- **ImageNet VLM** — VLM for ImageNet-1k classification, padded variable-length batches with masked CE, per-batch `seq_lens` for optimized prefix-LM attention

## Optimizations

### SIMD (ARM NEON)

- **`fast expf`** — range reduction + 6th-order Taylor polynomial on [-ln2/2, ln2/2], max rel error < 1e-6
- **Activations** — ReLU, sigmoid, SiLU, SwiGLU fwd/bwd (all fused, no intermediate tensors)
- **Softmax** — row-prefix online max+sum_exp with NEON `simd_expf_f32` for fwd/bwd
- **Causal softmax** — fused online max+sum_exp in single pass (both causal and prefix-LM)
- **Cross-entropy** — fwd/bwd with one-hot detection in NEON (`simd_ce_bwd_row_kernel`)
- **Pooling** — `avg_pool2d` fwd/bwd with k=2,s=2 and k=2,s=1 SIMD kernels
- **LayerNorm / RMSNorm** — vectorized mean, variance, and scale/shift with NEON
- **Reductions** — `simd_reduce_max_f32`, `simd_reduce_sum_f32` horizontal ops
- **Attention** — row-prefix softmax with online max+sum_exp rescaling in NEON

### OpenMP (parallelism)

- **im2col/col2im** — `#pragma omp parallel for` over batch dimension
- **LayerNorm / RMSNorm** — parallel over outer dims, `#pragma omp simd` reductions
- **Attention** — `collapse(2)` over batch×heads for tiled causal/prefix-LM fwd/bwd
- **Multi-head** — split/merge heads parallel over batch×heads (`collapse(2)`)
- **Element-wise ops** — parallel over outer dims with `#pragma omp simd` in inner loops
- **Matrix ops** — parallel over batch dim for matmul, bias scatter, etc.
- **Optimizer** — parallel gradient clipping (`reduction(+:total_norm_sq)`) and parameter update
- **Reductions** — parallel sum/mean with SIMD reduction clauses

### Convolution

- **(K, M) column layout** — im2col writes sequential floats (vs strided in M, K), GEMM uses `CblasTrans` on both operands for correct output
- **Winograd F(2×2, 3×3)** — replaces im2col+GEMM for 3×3 stride=1 pad=1 (most common config)
- **Forward col buffer reused in backward** — saves one im2col per training step
- **Bias fused into sgemm** `beta=1.0f` — avoids separate bias-add kernel
- **Bounds-peeled im2col/col2im** — no boundary checks in hot inner loops; `memset` zeroes padding once
- **Strided no-pad fast path** — when stride==kernel && pad==0, skips all bounds checks
- **Precomputed (oh, ow) ranges** per (kh, kw) — eliminates conditional branches in inner loop

### Memory & Runtime

- **3 bump allocators, No `malloc`/`realloc` in hot path** — all activations, saved tensors, and grad_fn nodes from scratch pool
- **Thread-local grad mode** — `dnn_no_grad_enter`/`exit` skips autograd tape entirely at eval
- **Contiguous fast-paths** — `tensor_sum`, dropout, softmax, layer norm, RMSNorm, embedding, CE all branch to simple loops when input is contiguous
- **Precomputed broadcast offsets** — element-wise backward ops compute index maps once per shape, not per element
- **Attention tiling** — `DNN_ATTENTION_TILE_ROWS` (default 64) controls row-block size for triangular causal attention
- **Causal softmax single-pass** — online max+sum_exp avoids storing full scores matrix for softmax

## Project Structure

```
include/         — Public API headers (tensor.h, ops.h, nn.h, autograd.h,
                   context.h, conv.h, optim.h, norm.h, pool.h,
                   attention.h, multihead.h, rope.h, tokenizer.h,
                   transformer.h, module.h, rng.h, gpt.h, vlm.h,
                   imagenet_vlm.h, dnn.h)
src/             — Implementation (.c + internal headers)
test/            — C unit tests + Python reference scripts
bench/           — Benchmarks (conv2d, matmul, ops, multihead, transformer,
                   attention, batched_matmul, cat, coord, vlm, vlm_debug,
                   conv2d_vlm)
docs/            — Design spec, optimization audit, coding rules
examples/mnist/          — MNIST: data loading, MLP/CNN/CNN+Pool models, training loops
examples/promessi_lm/    — Decoder-only transformer LM training & generation on Italian text
examples/mnist_vlm/      — VLM-based MNIST digit captioning (prefix-LM)
examples/imagenet_vlm/   — VLM for ImageNet-1k classification
main_prep_data.c — Data preparation utilities
Makefile         — Build: static lib, tests, benchmarks
```

## Build

```sh
make                          # builds libdnn.a + test binaries
make run_mnist_mlp            # build + run MNIST MLP classifier
make run_mnist_cnn            # build + run MNIST CNN classifier
make run_mnist_cnn_pool       # build + run MNIST CNN+Pool classifier
make run_promessi_lm          # build + run decoder-only LM on Italian text
make run_mnist_vlm            # build + run VLM-based MNIST digit captioning
make run_imagenet_vlm DATA_DIR=/path/to/imagenet-shards  # VLM ImageNet-1k
make main_prep_data           # build + run data preparation utilities
```

Requires: Apple Accelerate (or cblas), libz, libomp.

## Design Principles

- `_` prefix = internal API
- Comments explain WHY, not WHAT
- Test behavior, not implementation
- Interfaces are deep — few powerful operations, not many trivial ones
- All pooled, no raw `malloc` during training
- Executables go under `build/`
