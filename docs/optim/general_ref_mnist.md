# MNIST Examples Restructuring

## Goal

Move MNIST-specific code out of library into `examples/mnist/`. Library (`libdnn.a`) stays framework-only: tensor ops, autograd, modules, optimizers, pools, generic layers.

MNIST models, data loading, and train/eval loops become example code.

## Motivation

- Library should not depend on MNIST.
- New examples (CIFAR-10, GPT fine-tune, etc.) should not touch library sources.
- Examples should be copy-paste reference code for users building models.

## File map

### Removed from library / top-level

| Current path | Destination |
|---|---|
| `include/mnist.h` | `examples/mnist/mnist.h` |
| `src/mnist.c` | split across files below |
| `main.c` | `examples/mnist/mnist_cnn.c` (runner only) |
| `test/test_cnn_profile.c` | dropped |
| `test/test_cnn_stress.c` | dropped |

### New structure

```
examples/mnist/
  mnist.h                — internal example header
  mnist_data.c           — download, parse IDX, load images/labels
  mnist_train.c          — generic training impl + eval helper, no model wrappers
  mnist_mlp_model.c      — MLP struct create + forward + train wrapper
  mnist_cnn_model.c      — stride-2 CNN struct create + forward + train wrapper
  mnist_cnn_pool_model.c — avg-pool CNN struct create + forward + train wrapper
  mnist_mlp.c            — MLP runner executable
  mnist_cnn.c            — CNN runner executable (former top-level main.c)
  mnist_cnn_pool.c       — pool-CNN runner executable
  Makefile               — builds examples against ../../build/libdnn.a
```

Naming rule:

- `*_model.c` files define model structs, create/forward, and model-specific train wrapper.
- runner files (`mnist_mlp.c`, `mnist_cnn.c`, `mnist_cnn_pool.c`) contain `main()` only.
- `mnist_train.c` contains shared generic backend only; no wrapper refs to models it does not link.

This avoids unresolved symbols when building a single runner.

### What stays in library

All framework headers and sources remain:

```
include/
  tensor.h, ops.h, autograd.h, module.h, nn.h, norm.h, conv.h,
  pool.h, optim.h, attention.h, multihead.h, rope.h, tokenizer.h,
  transformer.h, dnn.h
src/
  tensor.c, ops_*.c, autograd.c, module.c, nn.c, norm.c, conv.c,
  pool.c, optim.c, attention.c, multihead.c, rope.c, tokenizer.c,
  transformer.c
```

`main_lm.c` stays top-level as primary library showcase.

### Tests dropped

- `test/test_cnn_profile.c`
- `test/test_cnn_stress.c`

Accepted. No replacement required in this migration.

## Deleted library API surface

### `include/mnist.h`

Entire file removed from public include dir. No library header should include it.

### `src/mnist.c`

Functions moved:

| Function | Destination |
|---|---|
| `mnist_download` | `examples/mnist/mnist_data.c` |
| `mnist_load_images` | `examples/mnist/mnist_data.c` |
| `mnist_load_labels` | `examples/mnist/mnist_data.c` |
| `mnist_model_create` | `examples/mnist/mnist_mlp_model.c` |
| `mnist_model_forward` | `examples/mnist/mnist_mlp_model.c` |
| `mnist_model_create_cnn` | `examples/mnist/mnist_cnn_model.c` |
| `mnist_model_forward_cnn` | `examples/mnist/mnist_cnn_model.c` |
| `mnist_model_create_cnn_pool` | `examples/mnist/mnist_cnn_pool_model.c` |
| `mnist_model_forward_cnn_pool` | `examples/mnist/mnist_cnn_pool_model.c` |
| `mnist_train_impl` | `examples/mnist/mnist_train.c` |
| `mnist_train` | `examples/mnist/mnist_mlp_model.c` |
| `mnist_train_cnn` | `examples/mnist/mnist_cnn_model.c` |
| `mnist_train_cnn_pool` | `examples/mnist/mnist_cnn_pool_model.c` |
| `mnist_eval` | `examples/mnist/mnist_train.c` |

### `main.c`

Removed from top-level. Replaced by `examples/mnist/mnist_cnn.c` runner.

## Example Makefile

Examples compile model + shared data/train files with one runner each.

```makefile
# examples/mnist/Makefile
CC ?= gcc
CFLAGS := -Wall -Wextra -pedantic -std=c11 -O3 -g
EXTRA_CFLAGS ?=
OMP_PREFIX ?= /opt/homebrew/opt/libomp
INCDIRS := -I../../include -I$(OMP_PREFIX)/include
CPPFLAGS := -DACCELERATE_NEW_LAPACK $(INCDIRS)
OMPFLAGS := -Xpreprocessor -fopenmp
LDFLAGS := -L$(OMP_PREFIX)/lib -L../../build
LDLIBS := -lomp -lz -ldnn -framework Accelerate

LIB := ../../build/libdnn.a
COMMON := mnist_data.c mnist_train.c

all: mnist_mlp mnist_cnn mnist_cnn_pool

$(LIB):
	$(MAKE) -C ../..

mnist_mlp: mnist_mlp.c mnist_mlp_model.c $(COMMON) $(LIB)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_mlp.c mnist_mlp_model.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@

mnist_cnn: mnist_cnn.c mnist_cnn_model.c $(COMMON) $(LIB)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_cnn.c mnist_cnn_model.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@

mnist_cnn_pool: mnist_cnn_pool.c mnist_cnn_pool_model.c $(COMMON) $(LIB)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_cnn_pool.c mnist_cnn_pool_model.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f mnist_mlp mnist_cnn mnist_cnn_pool
```

## Shared internal header: `mnist.h`

```c
#ifndef MNIST_EXAMPLE_H
#define MNIST_EXAMPLE_H

#include "dnn.h"

/* ── Constants ──
 * Paths are relative to repo root when running examples from repo root.
 * If running from examples/mnist/, pass explicit paths or adjust cwd.
 */
#define MNIST_DATA_DIR "data/mnist"
#define MNIST_TRAIN_N  60000
#define MNIST_TEST_N   10000
#define MNIST_ROWS     28
#define MNIST_COLS     28
#define MNIST_PIXELS   784
#define MNIST_CLASSES  10

/* ── Models ── */

typedef struct {
    module  base;
    linear *fc1;    /* 784 → 256 */
    linear *fc2;    /* 256 → 10  */
} mnist_model;

typedef struct {
    module   base;
    conv2d  *conv1;  /*  1→32, 3×3, pad=1, s1 */
    conv2d  *conv2;  /* 32→64, 3×3, pad=1, s2 */
    conv2d  *conv3;  /* 64→64, 3×3, pad=1, s2 */
    linear  *fc1;    /* 3136 → 128 */
    linear  *fc2;    /* 128  → 10  */
} mnist_model_cnn;

typedef struct {
    module   base;
    conv2d  *conv1;  /*  1→32, 3×3, pad=1, s1 */
    conv2d  *conv2;  /* 32→64, 3×3, pad=1, s1 */
    conv2d  *conv3;  /* 64→64, 3×3, pad=1, s1 */
    linear  *fc1;    /* 3136 → 128 */
    linear  *fc2;    /* 128  → 10  */
} mnist_model_cnn_pool;

/* ── Data loading ── */
int      mnist_download(void);
tensor  *mnist_load_images(struct mem_pool *data, const char *path);
tensor  *mnist_load_labels(struct mem_pool *data, const char *path);

/* ── Model lifecycle ── */
mnist_model          *mnist_model_create(struct mem_pool *params);
tensor               *mnist_model_forward(struct mem_pool *scratch, struct module *base, const tensor *x);

mnist_model_cnn      *mnist_model_create_cnn(struct mem_pool *params);
tensor               *mnist_model_forward_cnn(struct mem_pool *scratch, struct module *base, const tensor *x);

mnist_model_cnn_pool *mnist_model_create_cnn_pool(struct mem_pool *params);
tensor               *mnist_model_forward_cnn_pool(struct mem_pool *scratch, struct module *base, const tensor *x);

/* ── Generic train/eval backend ── */
typedef tensor *(*mnist_forward_fn)(struct mem_pool *, struct module *, const tensor *);

void mnist_train_impl(struct dnn_ctx *ctx,
                      tensor *train_images, tensor *train_labels,
                      int epochs, int batch_size, float lr,
                      int val_n, int patience,
                      struct module *model, mnist_forward_fn forward_fn);

float mnist_eval(struct mem_pool *scratch, struct module *model,
                 mnist_forward_fn forward_fn,
                 tensor *images, tensor *labels);

/* ── Model-specific train wrappers (defined beside each model) ── */
void mnist_train(struct dnn_ctx *ctx, mnist_model *m,
                 tensor *train_images, tensor *train_labels,
                 int epochs, int batch_size, float lr,
                 int val_n, int patience);

void mnist_train_cnn(struct dnn_ctx *ctx, mnist_model_cnn *m,
                     tensor *train_images, tensor *train_labels,
                     int epochs, int batch_size, float lr,
                     int val_n, int patience);

void mnist_train_cnn_pool(struct dnn_ctx *ctx, mnist_model_cnn_pool *m,
                          tensor *train_images, tensor *train_labels,
                          int epochs, int batch_size, float lr,
                          int val_n, int patience);

#endif /* MNIST_EXAMPLE_H */
```

## Generic train impl shape

`mnist_train.c` must not reference concrete model functions. It derives params from module tree:

```c
void mnist_train_impl(struct dnn_ctx *ctx,
                      tensor *train_images, tensor *train_labels,
                      int epochs, int batch_size, float lr,
                      int val_n, int patience,
                      struct module *model, mnist_forward_fn forward_fn) {
    int n_params;
    tensor **params = module_parameters(model, &n_params);
    adamw_opt *opt = adamw_create(ctx->params, params, n_params, lr,
                                  0.9f, 0.999f, 1e-8f, 0.01f);
    /* shared loop */
}
```

Model-specific wrappers live beside models:

```c
/* mnist_cnn_model.c */
void mnist_train_cnn(struct dnn_ctx *ctx, mnist_model_cnn *m,
                     tensor *train_images, tensor *train_labels,
                     int epochs, int batch_size, float lr,
                     int val_n, int patience) {
    mnist_train_impl(ctx, train_images, train_labels,
                     epochs, batch_size, lr, val_n, patience,
                     &m->base, mnist_model_forward_cnn);
}
```

## Runner files

`examples/mnist/mnist_cnn.c` is former top-level `main.c`, adjusted to include local `mnist.h` and link to `mnist_cnn_model.c`.

Runner should be minimal:

```c
#include "mnist.h"

int main(void) {
    dnn_ctx ctx;
    dnn_ctx_init(&ctx, /* params */ 64UL << 20,
                       /* scratch */ 192UL << 20,
                       /* data */ 256UL << 20);
    mnist_download();
    /* load data, create model, train, eval */
    dnn_ctx_destroy(&ctx);
    return 0;
}
```

## Cleanup of top-level Makefile

Delete MNIST top-level targets/rules:

```makefile
run: $(BUILDDIR)/main $(LIB)
main: $(BUILDDIR)/main
$(BUILDDIR)/main: main.c $(LIB)
```

Also remove `main` and `run` from `.PHONY` if they only referred to MNIST.
Keep `main_lm`, `main_prep_data`, `run_lm`.

Add comment:

```makefile
# MNIST examples live in examples/mnist/.
```

## README/docs updates

Update README:

- remove `include/mnist.h` from public header list
- remove `main.c` from top-level file tree
- replace `make run` / `make main` MNIST commands with:

```sh
make
make -C examples/mnist mnist_cnn
./examples/mnist/mnist_cnn
```

- mention MNIST is an example, not library API

## Migration steps

1. Create `examples/mnist/`.
2. Copy `include/mnist.h` → `examples/mnist/mnist.h`; remove from `include/`.
3. Split `src/mnist.c`:
   - data functions → `mnist_data.c`
   - generic train/eval → `mnist_train.c`
   - MLP model + wrapper → `mnist_mlp_model.c`
   - CNN model + wrapper → `mnist_cnn_model.c`
   - pool CNN model + wrapper → `mnist_cnn_pool_model.c`
4. Copy top-level `main.c` → `examples/mnist/mnist_cnn.c` and adjust as runner.
5. Add `mnist_mlp.c` and `mnist_cnn_pool.c` runners.
6. Add examples Makefile.
7. Remove `include/mnist.h`, `src/mnist.c`, `main.c`, `test/test_cnn_profile.c`, `test/test_cnn_stress.c`.
8. Remove top-level `run`/`main` MNIST targets.
9. Update README.
10. Build:
    ```sh
    make clean && make
    make test
    make -C examples/mnist
    ```
11. Smoke run one example if data/network available:
    ```sh
    ./examples/mnist/mnist_cnn
    ```
