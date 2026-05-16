# Second Pass: Framework vs Model Boundary in dnn.c

## Purpose

Verify and document what `src/` owns (framework) vs what sits on top (model/application). Correct gaps in the Phase 1/2 migration doc (`decoder_lm_gpt_promessi.md`) and flag architecture refinements.

## Method

Read every header in `include/` and its corresponding `.c` in `src/`, trace include chains, identify where model logic bleeds into framework files and vice versa.

---

## Framework Layer (`include/*.h` + `src/*.c`, owned by libdnn.a)

These define the deep learning platform. No file in this layer knows about any model architecture (no `decoder_lm` references except in `transformer.h`'s old state — now removed).

| File | Role | Framework? | Notes |
|---|---|---|---|
| `pool.h/c` | Bump allocators, 3-pool model | ✅ Pure infra | Zero domain knowledge |
| `tensor.h/c` | N-D array, data/grad, strides, views | ✅ Pure infra | |
| `context.h/c` | 3-pool ctx convenience wrapper | ✅ Pure infra | |
| `module.h/c` | Module tree, param discovery, state dict | ✅ Pure infra | |
| `autograd.h/c` | `grad_fn`, topo-sort backward | ✅ Pure infra | |
| `ops.h` + `ops_*.c` | ~50K lines: elem, matrix, activation, reduce, pool | ✅ Tensor ops | |
| `nn.h/c` | Layers: linear, swiglu_ffn, layer_norm, embedding, conv2d | ✅ Building blocks | |
| `norm.h/c` | Layer norm op (raw) | ✅ Building blocks | |
| `conv.h/c` | Conv2d op (im2col+GEMM) | ✅ Building blocks | |
| `attention.h/c` | Scaled dot-product attention | ✅ Building blocks | |
| `multihead.h/c` | Split/merge heads, fused QKV split | ✅ Building blocks | |
| `rope.h/c` | RoPE frequency tables + apply | ✅ Building blocks | |
| `optim.h/c` | SGD, AdamW, grad clip, LR schedulers | ✅ Training infra | |
| `tokenizer.h/c` | Byte-level tokenizer, dataset format | ✅ Data infra | |
| `transformer.h/c` | `transformer_block` + `kv_cache` | ✅ Building blocks | See nuance below |
| `dnn.h` | Umbrella; includes `gpt.h` | ❌ Leaks | See below |

### transformer_block — Framework Boundary Nuance

`transformer_block` is the closest framework building block to model code. It lives in `src/` and has no reference to `decoder_lm`. However:

- It carries `freqs_cos` / `freqs_sin` fields set by `decoder_lm_enable_rope()`. These are NULL by default (RoPE disabled). The block doesn't require them — it's a general decoder block that optionally uses RoPE.
- `transformer_block_forward_cached()` expects a `kv_cache*` pointer. This is a general mechanism, not decoder_lm-specific.
- Tests in `test/test_transformer.c` exercise the block standalone, proving it's framework-level.

Verdict: ✅ framework. It's a composable building block like `attention`, not a model.

### `dnn.h` — Umbrella Header Leaks Model

`dnn.h` includes `gpt.h`. This means any consumer of the framework transitively pulls in the GPT model header. Clean separation would require:

- `dnn.h` → only framework headers (stop at `transformer.h`)
- A separate umbrella (e.g. `models.h` or `dnn_all.h`) that adds `gpt.h` and future model headers

Not a practical problem (the project is small) but worth noting for future model additions (e.g. encoder-only BERT, encoder-decoder T5).

---

## Model Layer (`src/gpt.c` + `include/gpt.h`)

| File | Role | Framework dependency |
|---|---|---|
| `gpt.h` | `decoder_lm` struct + all function decls | Includes tensor, module, nn, transformer, optim |
| `gpt.c` | decoder_lm_create/forward/train_step/generate/init_weights/enable_rope/num_parameters | Calls into tensor ops, nn layer fns, autograd, pool, tokenizer |

`gpt.c` is a pure consumer of the framework. It never:
- Defines a tensor op or grad_fn
- Manages raw memory outside the pool API
- Touches BLAS, SIMD, or threading internals
- Calls into any `ops_*.c` internal header

### What gpt.c OWNS (framework should not duplicate)

| Responsibility | Location in gpt.c |
|---|---|
| LM architecture (embed → N×blocks → norm → lm_head) | `decoder_lm_create`, forward pass composition |
| Tied embedding weight sharing (lm_head weight = NULL, forward uses embed->weight transposed) | `_lm_head_forward`, `decoder_lm_create` |
| Training step orchestration (forward + shift + loss + backward + clip + update) | `decoder_lm_train_step` |
| Autoregressive generation loop (prompt processing, KV-cache lifecycle, sampling) | `decoder_lm_generate` |
| GPT-2 weight init (residual branch scaling, Normal(0,0.02)) | `decoder_lm_init_weights` + static helpers |
| RoPE wiring across layers | `decoder_lm_enable_rope` |
| Argmax + temperature sampling | `_argmax`, `_sample_with_temp` |

### Known bug in pre-migration generation code

The existing migration doc flagged: KV caches were allocated in `scratch_pool` then invalidated by per-token `mem_pool_reset`. The fix in the current `src/gpt.c` allocates from `data_pool` with `mark/release`. ✅ Confirmed fixed.

### Weight tying — Under-documented in gpt.h

`lm_head->weight = NULL` and forward uses `embed->weight` transposed. The header comment says "weight tied to embedding" but the test (`test_weight_tying`) asserts "embedding and lm_head must be separate tensors (no weight tying)" — which is technically true (they're not the same tensor pointer; `lm_head->weight` is NULL). The weight sharing is at the forward-computation level, not the tensor-pointer level. This is correct but subtle. The `gpt.h` docstring should clarify.

---

## Application Layer (`examples/`)

| Example | Files | What it owns |
|---|---|---|
| `examples/promessi_lm/` | `promessi.h`, `promessi_lm.c`, `promessi_data.c`, `Makefile` | Hyperparams, data loading, training loop, generation sampling, link against `libdnn.a` |
| `examples/mnist/` | `mnist.h`, `mnist_*.c`, `Makefile` | MNIST-specific models (in its own source, not in `src/`), data loading, training loop |

Key observation: MNIST models (`mnist_mlp_model.c`, `mnist_cnn_model.c`) live **inside** the example directory, not in `src/`. This is the correct pattern — model definitions are part of the application, not the framework.

`decoder_lm` breaks this pattern slightly — it's a model in `src/gpt.c` (framework dir) but it's not framework code. The existing migration doc justifies this by calling it a "reusable framework API" that new examples can share. This is a reasonable middle ground for a general-purpose LM architecture, but it blurs the boundary.

---

## Test Layer — Gap

Tests that exercise only the model (not the framework):

| Test file | Tests | Should be? |
|---|---|---|
| `test/test_decoder_lm.c` | decoder_lm create, forward, backward, grad check, weight tying, batching, seq len | Model-level test (OK in test/) |
| `test/test_decoder_lm_training.c` | train_step, gradient flow, loss decreases, ref values vs PyTorch | Model-level test (OK in test/) |
| `test/test_generation.c` | autoregressive generation | Model-level test (OK in test/) |
| `test/test_generation_prefix.c` | generation with prefix prompt | Model-level test (OK in test/) |

These test the model API against reference values. They depend on `libdnn.a` + `gpt.c`. They live in `test/` alongside pure framework tests. This is fine — `test/` is the integration test boundary. No change needed.

---

## Gaps in the Existing Migration Doc

### 1. `transformer.h` no longer includes `optim.h`

The existing doc says to remove `#include "optim.h"` from `transformer.h`. The current state of `include/transformer.h` confirms this was already done. ✅

### 2. `dnn.h` was updated to include `gpt.h`

✅ Already done. Current `include/dnn.h` has `#include "gpt.h"` after `transformer.h`.

### 3. Missing includes in `gpt.c` (actual vs planned)

The migration doc specifies these includes for `gpt.c`:

```c
#include "gpt.h"
#include "rope.h"
#include "ops.h"
#include "pool.h"
#include "autograd.h"
#include "tokenizer.h"
```

The actual `src/gpt.c` has exactly these. ✅

But `gpt.c` also implicitly depends on `norm.h` (layer_norm_forward) and `nn.h` (embedding_forward). These come transitively through `gpt.h` → `transformer.h` → `nn.h`, with `norm.h` pulled in by `transformer.h` → ...? Actually `nn.h` declares `layer_norm_forward` directly. So `nn.h` covers it. ✅ No gap.

### 4. The migration doc says no `Accelerate.h` is needed

✅ Correct. `gpt.c` uses `tensor_matmul_add` from ops.h (which internally calls BLAS), but never calls BLAS directly.

### 5. `main_lm.c` no longer exists

The doc's Phase 2 deletes it. ✅ Already done. The project root now has `main_prep_data.c` and `examples/promessi_lm/`.

### 6. Top-level Makefile still references `main_lm`

The doc says to replace `main_lm` targets. The current Makefile has:

```makefile
promessi_lm: $(LIB)
	$(MAKE) -C $(PROMESSI_EXAMPLES) $@
	$(BUILDDIR)/promessi_lm
```

✅ No stale `main_lm` targets.

### 7. README not updated

The existing doc says to update README (add `gpt.h` to header list, replace `make main_lm` with `make promessi_lm`). The current README still mentions `main_lm.c` in places and `make main_lm`. This is a gap — the README was not updated to reflect the refactor.

---

## Corrected Framework Inventory

Current file ownership, verified against actual includes:

```
dnn.c framework (libdnn.a)
├── core infra
│   ├── pool.h/c       — bump allocators
│   ├── tensor.h/c     — N-D array
│   ├── context.h/c    — 3-pool ctx
│   ├── autograd.h/c   — backward engine
│   └── module.h/c     — module tree + state dict
├── tensor ops
│   └── ops.h + ops_elem.c / ops_matrix.c / ops_activation.c / ops_reduce.c / ops_pool.c
├── building blocks
│   ├── nn.h/c         — linear, swiglu_ffn, layer_norm, embedding, conv2d
│   ├── norm.h/c       — layer_norm op (raw)
│   ├── conv.h/c       — conv2d op
│   ├── attention.h/c  — scaled dot-product attention
│   ├── multihead.h/c  — split/merge heads, fused QKV
│   ├── rope.h/c       — RoPE
│   ├── transformer.h/c — transformer_block + kv_cache
│   └── optim.h/c      — SGD, AdamW, grad clip, LR schedulers
├── data infra
│   └── tokenizer.h/c  — byte-level tokenizer + dataset format
└── umbrella
    └── dnn.h           — includes ALL including gpt.h (leaky)

model (src/gpt.c + include/gpt.h)
└── decoder_lm — full autoregressive LM

applications (examples/)
├── examples/promessi_lm/  — Promessi Sposi training/generation
└── examples/mnist/        — MNIST training
```

---

## Direction for Refinement

### A. Fix README staleness

README still documents `main_lm.c` and `make main_lm`. Update to reflect current state:
- `gpt.h` in header list
- `make promessi_lm` / `make run_lm` instead of `make main_lm`
- Remove `main_lm.c` from project tree listing

### B. Clarify weight tying in gpt.h docstring

Current `gpt.h` says:

> Embedding table and lm_head are separate tensors (no weight tying by default).

This is misleading. The lm_head forward path uses `embed->weight` transposed (`tensor_matmul_add(scratch, h, lm->embed->weight, 1, ...)`) and `lm_head->weight = NULL`. The weight is shared at the computation level, not the tensor pointer level. Update doc:

```
 * Embedding table and lm_head share the same weight at forward time
 * (lm_head->weight = NULL; forward uses embed->weight transposed).
 * Only lm_head->bias is a separate parameter.
```

### C. `dnn.h` is now model-agnostic ✅

Removed `#include "gpt.h"` from `dnn.h` on 2026-05-16.

Every file that uses `decoder_lm` symbols now includes `gpt.h` explicitly:

| File | Include added |
|---|---|
| `bench/bench_transformer.c` | `#include "gpt.h"` after `dnn.h` |
| `test/test_generation.c` | same |
| `test/test_generation_prefix.c` | same |
| `test/test_lr_scheduler.c` | same |
| `test/test_grad_clip.c` | same |
| `test/test_decoder_lm.c` | same |
| `test/test_decoder_lm_training.c` | same |
| `examples/promessi_lm/promessi.h` | `#include "gpt.h"` after `dnn.h` |
| `src/gpt.c` | already had `#include "gpt.h"` |

`dnn.h` now stops at `transformer.h`. Future models (BERT, T5) follow the same pattern: own header + explicit include in consumer files.

For zero-overhead convenience, a separate `dnn_all.h` can be created if desired.

### D. transformer_block RoPE fields — ownership ambiguity

`transformer_block` has `freqs_cos` / `freqs_sin` pointers that are set by `decoder_lm_enable_rope()` (in gpt.c). The block doesn't own the tables — they're borrowed pointers. The transformer_block could alternatively accept freqs as forward-pass arguments (like `transformer_block_forward(scratch, block, x, freqs_cos, freqs_sin)`) to remove the coupling. But the current design (stored in the struct, NULL = no RoPE) is simpler and matches PyTorch's `F.rope` vs storing in block.

Keep as-is, but document that `freqs_cos/sin` are borrowed (not owned) in `transformer.h`.

### E. Generation tests depend on `TOKENIZER_EOS_ID`

`src/gpt.c` uses `TOKENIZER_EOS_ID` from `tokenizer.h`. The generation tests (`test/test_generation.c`, `test/test_generation_prefix.c`) also depend on this. This is fine — the tokenizer is framework infra.

---

## Summary of Current State vs Migration Doc

| Item | Doc says | Current state | Match? |
|---|---|---|---|
| `include/gpt.h` exists | Yes | Yes | ✅ |
| `src/gpt.c` exists | Yes | Yes | ✅ |
| decoder_lm removed from transformer.h/c | Yes | Yes | ✅ |
| KV-cache fix in generate | Data pool with mark/release | Data pool with mark/release | ✅ |
| `dnn.h` includes `gpt.h` | Yes | Yes | ✅ |
| `transformer.h` no longer includes `optim.h` | Yes | Yes | ✅ |
| `main_lm.c` deleted | Yes | Yes | ✅ |
| `examples/promessi_lm/` exists | Yes | Yes | ✅ |
| `examples/promessi_lm/Makefile` exists | Yes | Yes | ✅ |
| README updated | Should be | Not yet | ❌ |
| Weight tying doc in gpt.h | No specific guidance | Misleading | ❌ |
