# dnn.c — Deep Neural Network library in C

From-scratch DNN training framework in C11.

Examples include: 
- classic MNIST digit classification (MLP/CNN)
- full decoder-only transformer LM with RoPE, KV-cache, and byte-level tokenizer (docs/promessi_sposi.txt)

~Zero heap allocations during training.

Decided to finally give this a shot after ~2 years of studying and replicating from first principles what great Andrej Karpathy shared on the internet (sources at the bottom). Got tired of python and pytorch, was time to go deeper. 

Thanks to this project, I truly realised how blessed we are to have pytorch and the people surroudning it.

The focus of this project for me was to put in practice the best low-level design practices I learnt about and instruct ds4-flash to implement them step-by-step (most of the time). 

I left ~0% room to the AI assistant for design choices, 90% freedom in implementation. As usual, pareto law applies here as well, the 10% implementation guidance brought the majority of longer-term benefits.

Up to now, the cost of this project is around $6. we average $1/day (opencode-go)

This model is truly amazing as a daily assistant in "pair-programming mode" (not yolo/vibes).

## Architecture

### Memory Model

3 bump allocators (`mem_pool`), no `free` — one `free` per pool at destroy.

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

| Category | Functions |
|----------|-----------|
| Element-wise | `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, `tensor_neg`, `tensor_pow` |
| Matrix | `tensor_matmul` (BLAS via Accelerate on macOS, fallback manual) |
| Activation | `tensor_relu`, `tensor_sigmoid`, `tensor_tanh`, `tensor_silu`, `tensor_swiglu`, `tensor_softmax`, `tensor_causal_softmax` |
| Regularization | `tensor_dropout` (inverted dropout) |
| Reduction | `tensor_sum`, `tensor_mean` |
| Loss | `tensor_cross_entropy` (fused softmax + NLL, numerically stabilized) |
| Normalization | `tensor_layer_norm` (last-dim, learnable γ/β) |
| Convolution | `tensor_conv2d` (im2col + GEMM, bias, stride, padding) |
| Attention | `tensor_attention` (scaled dot-product, mask, 2/3/4D), `tensor_split_heads`, `tensor_merge_heads` |
| Position | `tensor_rope` (RoPE rotary encoding), `tensor_rope_freqs_init` |
| Embedding | `tensor_embedding` (table lookup by int IDs, autograd) |
| Utility | `tensor_cat` (concat along dim), `tensor_triu` (upper-triangular mask) |

All element-wise ops support NumPy-style broadcasting with fast-paths for
same-shape contiguous inputs.

## Layers / Models

- **`linear`** — `y = x @ W + b`, uniform init, autograd wired
- **`swiglu_ffn`** — SwiGLU FFN: SiLU(xW_g) ⊗ (xW_u) W_d, matches Llama/Mistral
- **`transformer_block`** — Decoder block (pre-norm): causal attn + SwiGLU FFN, residual, RoPE
- **`decoder_lm`** — Full autoregressive LM: embed → N×transformer → norm → lm_head
- **`mnist_model`** — MLP: 784 → 256 → 10, ReLU + dropout(0.2)
- **`mnist_model_cnn`** — 3 conv layers (1→32→64→64, 3×3 kernels) + FC 3136→128→10, stride-2 downsampling
- **`mnist_model_cnn_pool`** — 3 conv layers + avg_pool2d downsampling, comparable architecture

## KV-Cache

Pre-allocated K/V buffers for O(1) per-step autoregressive generation.

- `kv_cache_create` — allocates [B, H, max_seq, d_k] tensors in params pool
- `kv_cache_append` — copies new K/V into cache at `seq_len`, advances
- `kv_cache_get_K`/`get_V` — returns slice view of valid portion
- `transformer_block_forward_cached` — forward one new token with cached K/V
- `decoder_lm_generate` — autoregressive generation with sampling, EOS stop

No autograd — generation is eval-only. KV-cache not used during training
(teacher forcing computes full sequence in one shot).

## RoPE (Rotary Position Embedding)

Pair-wise 2D rotation applied to Q/K head dimensions. In-place on input buffer.

- `tensor_rope` — apply rotation, backward wired (inverse angle transpose)
- `tensor_rope_freqs_init` — compute cos/sin tables in params pool
- `decoder_lm_enable_rope` — assign freq tables to all blocks

## Tokenizer

Byte-level tokenizer mapping each byte (0–255) to token ID = byte value.
Special tokens occupy IDs 257–260.

- BOS=257, EOS=258, PAD=259, UNK=260
- Optional special-token strings (`<|im_start|>`, `<|im_end|>`, `<|pad|>`)
- `tokenizer_default()` — basic byte tokenizer
- `tokenizer_with_chat_template()` — preset with chat template special strings
- `tokenizer_with_specials()` — custom special string overrides
- Encode prepends BOS, appends EOS; decode strips special tokens
- Binary dataset format: magic header + flat int32 token IDs
- `tokenizer_text_to_tensor` / `tokenizer_tensor_to_text` for direct tensor I/O

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

### MNIST (`main.c`)

- Downloads MNIST via `curl` + `gunzip` (4 IDX files)
- Loads images as float tensors [N,784] ∈ [0,1], labels as int tensors [N]
- Batched training loop with shuffling, early stopping (patience-based)
- AdamW optimizer, cross-entropy loss, batch=128, lr=0.001
- Shared `mnist_train_impl()` used by both MLP and CNN paths
- CNN model: 3 conv layers (1→32→64→64) + FC 3136→128→10

### Decoder LM (`main_lm.c`)

- Loads binary tokenized dataset (`data/promessi.bin`) with header + int32 IDs
- Decoder-only transformer: d_model=256, 2 layers, 4 heads (d_k=64), SwiGLU FFN (int=512)
- RoPE position encoding, GPT-2 style weight init, AdamW, LR scheduler (warmup + cosine)
- High-level `decoder_lm_train_step` (forward + shift + loss + backward + grad clip + update)
- Teacher-forced next-token prediction with cross-entropy
- Periodic autoregressive generation samples during training

## Optimizations

- NEON SIMD for ReLU fwd/bwd, sigmoid, SiLU, SwiGLU fwd/bwd, avg_pool2d fwd/bwd, softmax, cross-entropy, fast `expf` polynomial
- OpenMP parallel for im2col/col2im, layer norm, reductions
- (K,M) column layout in conv — sequential im2col writes, `CblasTrans` GEMM
- Forward col buffer reused in conv backward (saves 1 im2col per step)
- Bias fused into sgemm `beta=1.0f`
- Precomputed broadcast offsets in element-wise backward ops
- Contiguous fast-paths in `tensor_sum`, dropout, softmax, layer norm, embedding
- No `malloc`/`realloc` in hot path — all scratch pool allocations
- Bounds-peeled im2col/col2im (no boundary checks in hot loop)

## Project Structure

```
include/         — Public API headers (tensor.h, ops.h, nn.h, autograd.h,
                   context.h, conv.h, optim.h, norm.h, pool.h,
                   attention.h, multihead.h, rope.h, tokenizer.h,
                   transformer.h, mnist.h, dnn.h)
src/             — Implementation (.c + internal headers)
test/            — C unit tests + Python reference scripts
bench/           — Benchmarks (conv2d, matmul, ops, multihead, transformer, coord, batched_matmul)
docs/            — Design spec, optimization audit, coding rules
main.c           — Entry point: CNN training on MNIST
main_lm.c        — Entry point: decoder-only LM training
main_prep_data.c — Entry point: data preparation utilities
Makefile         — Build: static lib, tests, benchmarks
```

## Build

```sh
make            # builds libdnn.a + test binaries
make run        # builds + runs MNIST CNN training
make run_lm     # builds + runs decoder LM training
make test       # runs all unit tests
make bench_all  # conv2d/matmul/ops/multihead/transformer benchmarks
make main       # build + run MNIST
make main_lm    # build + run decoder LM
```

Requires: Apple Accelerate (or cblas), libz, libomp.

## Design Principles

- `_` prefix = internal API
- Comments explain WHY, not WHAT
- Test behavior, not implementation
- Interfaces are deep — few powerful operations, not many trivial ones
- All pooled, no raw `malloc` during training
- Executables go under `build/`
