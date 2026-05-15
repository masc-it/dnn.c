# Decoder LM → GPT + Promessi Sposi Example Migration

## Goal

1. Extract `decoder_lm` (full autoregressive LM) from `src/transformer.c` into a dedicated `src/gpt.c` + `include/gpt.h` as framework code.
2. Move promessi-sposi-specific code from `main_lm.c` → `examples/promessi_lm/` following the MNIST example pattern.

## Motivation

- `transformer.c` should own only the transformer block and KV cache infrastructure. The full LM model (`decoder_lm`) is a higher-level architecture that belongs in its own translation unit.
- `main_lm.c` mixes framework model code with example-specific data loading, config, and training loop. Promessi Sposi should be a standalone example.
- New examples (Shakespeare, fine-tune, etc.) can reuse `gpt.h` framework API.

## Phase 1 — Create `include/gpt.h` + `src/gpt.c`

### What moves from `transformer.c` / `transformer.h`

| Symbol | Moves to |
|---|---|
| `decoder_lm` struct (all fields) | `include/gpt.h` |
| `decoder_lm_create` | `src/gpt.c` |
| `decoder_lm_forward` | `src/gpt.c` |
| `decoder_lm_train_step` | `src/gpt.c` |
| `decoder_lm_generate` | `src/gpt.c` |
| `decoder_lm_num_parameters` | `src/gpt.c` |
| `decoder_lm_init_weights` | `src/gpt.c` |
| `decoder_lm_enable_rope` | `src/gpt.c` |
| static helpers (`_argmax`, `_sample_with_temp`, `_randn`, `_init_linear`, `_init_module_weights`) | `src/gpt.c` (file-static) |

### What stays in `transformer.h` / `transformer.c`

| Symbol | Reason |
|---|---|
| `kv_cache` struct + create/append/get_K/get_V | KV cache is a transformer-block-level primitive |
| `transformer_block` struct + create/forward | Core building block, generic |
| `transformer_block_forward_cached` | Operates on a single block using KV cache |
| `transformer_block_num_parameters` | Block-level parameter query |

### Type dependency chain

```
gpt.h
 ├── tensor.h      (tensor, mem_pool)
 ├── module.h      (module, module_item, MODULE_ITEM_CHILD)
 ├── nn.h          (embedding, linear, layer_norm, swiglu_ffn)
 ├── transformer.h (transformer_block, kv_cache)
 └── optim.h       (adamw_opt, clip_grad_norm)
```

Note: current `transformer.h` includes `tensor.h`, `module.h`, `nn.h`, and `optim.h`, but after this split `transformer.h` should drop `optim.h`. `gpt.h` includes all direct dependencies explicitly to avoid fragile transitive dependencies.

### New file: `include/gpt.h`

```c
#ifndef DNN_GPT_H
#define DNN_GPT_H

#include "tensor.h"
#include "module.h"
#include "nn.h"
#include "transformer.h"   /* for transformer_block, kv_cache */
#include "optim.h"         /* for adamw_opt, clip_grad_norm */

/* ── Decoder-only Language Model (GPT) ──
 *
 * Full autoregressive LM: embed → N×transformer_block → norm → lm_head
 *
 *   input_ids [B, N] (int tokens) → logits [B, N, vocab_size]
 *
 * All parameters in params pool.  Autograd wired through entire graph.
 * No KV-cache during training (teacher forcing computes full seq in one shot).
 */

typedef struct {
    module               base;
    embedding           *embed;
    transformer_block  **blocks;
    int                  n_layers;
    layer_norm          *norm;
    linear              *lm_head;
    int                  d_model;
    int                  vocab_size;
} decoder_lm;

decoder_lm *decoder_lm_create(struct mem_pool *params_pool,
                               int vocab_size, int d_model,
                               int n_layers, int n_heads, int d_k,
                               int intermediate_size);

tensor *decoder_lm_forward(struct mem_pool *scratch,
                            decoder_lm *lm, const tensor *input_ids);

tensor *decoder_lm_train_step(struct mem_pool *scratch_pool,
                               struct mem_pool *data_pool,
                               decoder_lm *lm, const tensor *input_ids,
                               adamw_opt *opt, float grad_clip,
                               float *grad_norm_out);

int *decoder_lm_generate(struct mem_pool *scratch_pool,
                          struct mem_pool *data_pool,
                          decoder_lm *lm, const tensor *prompt_ids,
                          int max_new_tokens, float temperature,
                          int use_cache, int *n_out);

void decoder_lm_enable_rope(struct mem_pool *params_pool,
                             decoder_lm *lm, int max_seq_len, float base);

void decoder_lm_init_weights(decoder_lm *lm);

long long decoder_lm_num_parameters(decoder_lm *lm);

#endif /* DNN_GPT_H */
```

### New file: `src/gpt.c`

Copy-paste from `src/transformer.c`:

| Lines in transformer.c | Destination in gpt.c |
|---|---|
| 206–262 (`decoder_lm_create`) | gpt.c |
| 264–296 (`decoder_lm_forward`) | gpt.c |
| 298–355 (`decoder_lm_train_step`) | gpt.c |
| 560–605 (`_argmax`, `_sample_with_temp`) | gpt.c (file-static, before `decoder_lm_generate`) |
| 607–766 (`decoder_lm_generate`) | gpt.c |
| 773–775 (`decoder_lm_num_parameters`) | gpt.c |
| 778–836 (`_randn`, `_init_linear`, `_init_module_weights`, `decoder_lm_init_weights`) | gpt.c |
| 839–850 (`decoder_lm_enable_rope`) | gpt.c |

Also copy the `/* ── Decoder-only Language Model ── */` section comment (line 205)
and `/* ── Sampling helpers ── */` section comment (line 559)
and `/* ── Autoregressive generation ── */` section comment (line 607)
and `/* ── Parameter count ── */` section comment around line 773
and `/* ── Weight initialization (GPT-2 style) ── */` section comment around line 777
and `/* ── RoPE position encoding ── */` section comment around line 838.

Includes required in `gpt.c`:

```c
#include "gpt.h"
#include "rope.h"      /* tensor_rope_freqs_init */
#include "ops.h"       /* tensor_cross_entropy */
#include "pool.h"      /* mem_pool_reset / mark / release */
#include "autograd.h"  /* dnn_backward, dnn_no_grad_enter/exit */
#include "tokenizer.h" /* TOKENIZER_EOS_ID */
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
```

NOT needed:
- `norm.h` — `layer_norm_forward` is declared in `nn.h`
- `attention.h` / `multihead.h` — attention ops called only by `transformer_block_forward` in `transformer.c`
- `simd.h` / `Accelerate.h` — no SIMD attention or BLAS calls in gpt.c (all cached attention is in `transformer_block_forward_cached` in `transformer.c`)
- `nn.h` — declared types already available via `gpt.h` → `transformer.h` → `nn.h`. But including it explicitly in gpt.c is harmless.

### Deletions from `transformer.h`

Remove from `include/transformer.h`:

- Entire `decoder_lm` struct (from `/* ── Decoder-only Language Model ── */` comment to `} decoder_lm;` inclusive)
- `decoder_lm_create` declaration
- `decoder_lm_forward` declaration
- `decoder_lm_train_step` declaration
- `decoder_lm_generate` declaration
- `decoder_lm_enable_rope` declaration
- `decoder_lm_init_weights` declaration
- `decoder_lm_num_parameters` declaration

After removal, `#include "optim.h"` is no longer used in `transformer.h`.
Remove it to keep the header minimal:

```diff
 #include "tensor.h"
 #include "module.h"
 #include "nn.h"
-#include "optim.h"
```

### Deletions from `transformer.c`

Removed code blocks (exact line ranges in 850-line file):

| Lines | What | Notes |
|---|---|---|
| 205 | `/* ── Decoder-only Language Model ── */` comment | section header, remove |
| 206–262 | `decoder_lm_create` fn | entire function block |
| 263 | blank line | remove |
| 264–296 | `decoder_lm_forward` fn | entire function block |
| 297 | blank line | remove |
| 298–355 | `decoder_lm_train_step` fn | entire function block |
| 356 | blank line | remove |
| 559 | `/* ── Sampling helpers ── */` comment | section header, remove |
| 560–570 | `_argmax` fn | static, remove |
| 571 | blank line | remove |
| 572–605 | `_sample_with_temp` fn | static, remove |
| 606 | blank line | remove |
| 607 | `/* ── Autoregressive generation ── */` comment | section header, remove |
| 608–766 | `decoder_lm_generate` fn | entire function block |
| 767–768 | blank lines | remove |
| 772 | blank line | remove |
| 773–775 | `decoder_lm_num_parameters` fn | remove |
| 776 | blank line | remove |
| 777 | `/* ── Weight initialization (GPT-2 style) ── */` comment | section header, remove |
| 778–780 | blank + `/* Box-Muller */` comment | remove |
| 781–786 | `_randn` fn | static, remove |
| 787–788 | blank lines | remove |
| 789–798 | `_init_linear` fn | static, remove |
| 799–801 | blank + comment | remove |
| 802–815 | `_init_module_weights` fn | static, remove |
| 816–817 | blank lines | remove |
| 818–836 | `decoder_lm_init_weights` fn | remove |
| 837–838 | blank lines | remove |
| 839–850 | `decoder_lm_enable_rope` fn | remove (line 850 is final `}`) |

After deletions, `transformer.c` should be ~555 lines with only:
- Includes + BLAS conditional (16 lines)
- kv_cache functions (~90 lines) 
- transformer_block functions (~110 lines)
- transformer_block_forward_cached (~210 lines)
- transformer_block_num_parameters (3 lines)

### Update `include/dnn.h`

Add `#include "gpt.h"` after transformer.h:

```c
#include "transformer.h"
#include "gpt.h"        /* <-- add here */
#endif /* DNN_H */
```

### Update `Makefile` (top-level)

No change needed — the wildcard `$(wildcard $(SRCDIR)/*.c)` picks up `src/gpt.c` automatically.

### Test/bench file compatibility

All test files that include `transformer.h` also include `dnn.h` first:

```
test/test_decoder_lm.c:       #include "dnn.h" ... #include "transformer.h"
test/test_generation.c:       #include "dnn.h" ... #include "transformer.h"
test/test_generation_prefix.c:#include "dnn.h" ... #include "transformer.h"
test/test_decoder_lm_training.c: #include "dnn.h" ... #include "transformer.h"
test/test_grad_clip.c:        #include "dnn.h" ... #include "transformer.h"
test/test_transformer.c:      #include "dnn.h" ... #include "transformer.h"
bench/bench_transformer.c:    #include "dnn.h" ... #include "transformer.h"
```

Since `dnn.h` → `gpt.h` provides the `decoder_lm` struct and declarations,
and is included before `transformer.h`, all files compile without changes.

If any file only includes `transformer.h` without `dnn.h`, it needs to add `#include "gpt.h"`.
Currently none do. Safe.

---

## Phase 2 — Move `main_lm.c` → `examples/promessi_lm/`

### File map

| Current path | Destination |
|---|---|
| `main_lm.c` (config, data, train loop, generation) | `examples/promessi_lm/promessi_lm.c` (runner with `main()`) |
| (extracted from main_lm.c) | `examples/promessi_lm/promessi_data.c` (`load_dataset`, `shuffle_int`) |
| (extracted from main_lm.c) | `examples/promessi_lm/promessi.h` (constants, dataset struct, forward decls) |
| (new) | `examples/promessi_lm/Makefile` |

### `examples/promessi_lm/promessi.h` — Internal example header

```c
#ifndef PROMESSI_EXAMPLE_H
#define PROMESSI_EXAMPLE_H

#include "dnn.h"    /* pulls in gpt.h, transformer.h, tokenizer.h, etc. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* ── Config ── */
#define D_MODEL        256
#define N_HEADS         4
#define D_K            64   /* D_MODEL / N_HEADS */
#define INTERMEDIATE   512  /* 2x D_MODEL */
#define N_LAYERS         2
#define VOCAB_SIZE     261
#define BATCH_SIZE      16
#define MAX_EPOCHS      10
#define LR            8e-4f
#define MIN_LR            6e-5f
#define OVERFIT         1    /* 1 = train on 10 batches only for overfit test */
#if OVERFIT
#  undef  MAX_EPOCHS
#  define MAX_EPOCHS 1000
#endif
#define LOG_EVERY       10
#define GEN_EVERY       30
#define GEN_NEW_TOKENS  64

/* ── Binary dataset ── */
typedef struct {
    int   num_sequences;
    int   seq_len;
    int   vocab_size;
    int  *data;          /* [num_sequences, seq_len] flat int32 IDs */
    long  data_n;        /* num_sequences * seq_len */
} lm_dataset;

lm_dataset load_dataset(const char *path);
void       shuffle_int(int *arr, int n);

#endif /* PROMESSI_EXAMPLE_H */
```

### `examples/promessi_lm/promessi_data.c` — Data loading

Copy from `main_lm.c`, but remove `static` from the function definitions so they match `promessi.h` external declarations:

- `load_dataset()` — binary dataset reader (`lm_dataset` struct, fopen/fread, magic check, header parse, malloc for data)
- `shuffle_int()` — Fisher-Yates shuffle

Include:

```c
#include "promessi.h"
```

Definitions must be:

```c
lm_dataset load_dataset(const char *path) { ... }
void shuffle_int(int *arr, int n) { ... }
```

These are pure data functions. No model or training code.

Note: `load_dataset` uses `TOKENIZER_DATA_MAGIC` from `tokenizer.h`, which is available
via `promessi.h` → `dnn.h` → `tokenizer.h`.

### `examples/promessi_lm/promessi_lm.c` — Runner

Same as current `main_lm.c` except:

- `#include "promessi.h"` replaces all bare includes
- `load_dataset` and `shuffle_int` function definitions are removed (they live in `promessi_data.c`)
- Data path stays `"data/promessi.bin"` (relative to repo root)
- `decoder_lm_*` calls resolve through `gpt.h` (indirectly via `dnn.h` umbrella)

Diff:

```diff
- #include "dnn.h"
- #include "transformer.h"
- #include "optim.h"
- #include "tokenizer.h"
- #include <stdio.h>
- #include <stdlib.h>
- #include <string.h>
- #include <math.h>
- #include <assert.h>
- #include <time.h>
+ #include "promessi.h"
```

Then remove the entire `/* ── Binary dataset loader ── */` block (`load_dataset` fn + its struct + comment)
and the `/* Fisher-Yates shuffle */` block (`shuffle_int` fn).

Everything else stays identical.

### `examples/promessi_lm/Makefile`

Following `examples/mnist/Makefile` pattern — builds runner + data file against `libdnn.a`:

```makefile
CC ?= gcc
CFLAGS := -Wall -Wextra -pedantic -std=c11 -O3 -g
EXTRA_CFLAGS ?=
OMP_PREFIX ?= /opt/homebrew/opt/libomp
INCDIRS := -I../../include -I$(OMP_PREFIX)/include
CPPFLAGS := -DACCELERATE_NEW_LAPACK $(INCDIRS)
OMPFLAGS := -Xpreprocessor -fopenmp
LDFLAGS := -L$(OMP_PREFIX)/lib -L../../build
LDLIBS := -lomp -lz -ldnn -framework Accelerate

BUILDDIR := ../../build
LIB := $(BUILDDIR)/libdnn.a

.PHONY: all promessi_lm

all: $(BUILDDIR)/promessi_lm

promessi_lm: $(BUILDDIR)/promessi_lm

$(BUILDDIR):
	mkdir -p $@

$(LIB):
	$(MAKE) -C ../..

$(BUILDDIR)/promessi_lm: promessi_lm.c promessi_data.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    promessi_lm.c promessi_data.c $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(BUILDDIR)/promessi_lm
```

Note: `$(BUILDDIR)` is `../../build`, same as `examples/mnist/Makefile`. Both examples share the same build directory and lib.

### Top-level Makefile changes

Replace the old `main_lm` target/rule with delegation to example sub-make:

```diff
-# OLD — remove these two rules:
-main_lm: $(BUILDDIR)/main_lm
-	$(BUILDDIR)/main_lm
-
-$(BUILDDIR)/main_lm: main_lm.c $(LIB) | $(BUILDDIR)
-	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
-
-# NEW — add after MNIST section, before bench rules:
+# LM examples live in examples/promessi_lm/.
+PROMESSI_EXAMPLES := examples/promessi_lm
+
+promessi_lm: $(LIB)
+	$(MAKE) -C $(PROMESSI_EXAMPLES) $@
+
+run_lm: promessi_lm
+	$(BUILDDIR)/promessi_lm
```

Update `.PHONY`:

```diff
-.PHONY: all clean test bench main_lm main_prep_data run_lm
+.PHONY: all clean test bench main_prep_data promessi_lm run_lm
.PHONY: mnist_mlp mnist_cnn mnist_cnn_pool
```

(Keep `main_prep_data` as-is — it's a standalone data preparation tool, not part of this migration.)

### Delete `main_lm.c`

After migration, `rm main_lm.c`. Promessi training/generation code lives in `examples/promessi_lm/`.

`main_prep_data.c` stays top-level. It is the generic tokenizer dataset-prep tool that produces `data/promessi.bin`; it is not part of this migration.

---

## Known bug to fix while moving `decoder_lm_generate`

Current cached generation path allocates KV caches from `scratch_pool`, then calls `mem_pool_reset(scratch_pool)` inside the generation loop. That invalidates the caches.

Fix during move to `gpt.c`. Allocate caches from `data_pool` with mark/release, or keep a scratch mark after cache allocation and release back to that mark instead of full reset.

Preferred v1 fix: allocate caches from `data_pool` because generation output already lives there and caches must survive per-token scratch resets.

Sketch:

```c
size_t cache_mark = mem_pool_mark(data_pool);
kv_cache **caches = _mem_pool_alloc(data_pool,
                                    (size_t)lm->n_layers * sizeof(kv_cache*), NULL);
for (int i = 0; i < lm->n_layers; i++)
    caches[i] = kv_cache_create(data_pool, B, H, max_seq, d_k);

/* per token: mem_pool_reset(scratch_pool) is now safe */

mem_pool_release(data_pool, cache_mark);  /* before return, after done_generate */
```

Do not release past generated output. Since output is allocated before caches in current code, `cache_mark` must be taken **after** output allocation and before cache allocation.

## README updates

Update README:

- add `gpt.h` to public header list
- remove `main_lm.c` from top-level file tree
- replace `make main_lm` with `make promessi_lm` or `make run_lm`
- clarify `decoder_lm` / GPT API lives in `gpt.h`, not `transformer.h`
- mention Promessi Sposi LM is now `examples/promessi_lm/`

## Migration steps (ordered)

1. Create `include/gpt.h` with `decoder_lm` struct + all function declarations.
2. Gut `include/transformer.h`: remove decoder_lm struct, all `decoder_lm_*` declarations.
   Also remove `#include "optim.h"` (no longer needed in transformer.h).
3. Create `src/gpt.c` with all decoder_lm functions from `src/transformer.c`.
4. Delete the same code from `src/transformer.c`.
5. Add `#include "gpt.h"` to `include/dnn.h`.
6. Build library and run tests to verify no breakage:
   ```sh
   make clean && make && make test
   ```
7. Create `examples/promessi_lm/promessi.h`.
8. Create `examples/promessi_lm/promessi_data.c` (load_dataset + shuffle_int).
9. Create `examples/promessi_lm/promessi_lm.c` (runner, former main_lm.c stripped of data functions).
10. Create `examples/promessi_lm/Makefile`.
11. Update top-level `Makefile`: replace main_lm targets with promessi_lm delegation.
12. Update README as described above.
13. Delete `main_lm.c`.
14. Build everything and run tests:
    ```sh
    make clean && make && make test
    make -C examples/promessi_lm promessi_lm
    ```
15. Smoke test (if `data/promessi.bin` exists):
    ```sh
    make run_lm
    ```
