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
