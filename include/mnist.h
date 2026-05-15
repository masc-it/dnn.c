#ifndef MNIST_H
#define MNIST_H

#include "module.h"
#include "dnn.h"

/* ── Constants ── */

#define MNIST_DATA_DIR "data/mnist"

#define MNIST_TRAIN_N  60000
#define MNIST_TEST_N   10000
#define MNIST_ROWS     28
#define MNIST_COLS     28
#define MNIST_PIXELS   784
#define MNIST_CLASSES  10

/* ── Models ── */

/* MLP: 784 → 256 → 10 */
typedef struct {
    module  base;   /* first field */
    linear *fc1;    /* 784 → 256 */
    linear *fc2;    /* 256 → 10  */
} mnist_model;

/* CNN: convs 1→32→64→64 + FC 3136→128→10 (stride-2 for downsampling) */
typedef struct {
    module   base;                 /* first field */
    conv2d  *conv1;               /*  1→32, 3×3, pad=1, s1 → 28×28 */
    conv2d  *conv2;               /* 32→64, 3×3, pad=1, s2 → 14×14 */
    conv2d  *conv3;               /* 64→64, 3×3, pad=1, s2 →  7×7  */
    linear  *fc1;                 /* 3136 → 128 */
    linear  *fc2;                 /* 128  → 10  */
} mnist_model_cnn;

/* CNN (pool variant): stride-1 convs + avg_pool2d for downsampling.
 * All 3×3 convs use Winograd (stride=1).
 *   28 → conv1 → 28 → conv2 → 28 → pool → 14 → conv3 → 14 → pool → 7
 */
typedef struct {
    module   base;                 /* first field */
    conv2d  *conv1;               /*  1→32, 3×3, pad=1, s1 → 28×28 */
    conv2d  *conv2;               /* 32→64, 3×3, pad=1, s1 → 28×28 */
    conv2d  *conv3;               /* 64→64, 3×3, pad=1, s1 → 14×14 */
    linear  *fc1;                 /* 3136 → 128 */
    linear  *fc2;                 /* 128  → 10  */
} mnist_model_cnn_pool;

/* ── Data loading ── */

/* Download & decompress all 4 MNIST files via curl+gunzip.
   Skips files that already exist. Returns 0 on success. */
int mnist_download(void);

/* Load images from decompressed IDX file into float tensor [N, 784].
   Pixels normalized to [0, 1]. */
tensor *mnist_load_images(struct mem_pool *data, const char *path);

/* Load labels from decompressed IDX file into int tensor [N]. */
tensor *mnist_load_labels(struct mem_pool *data, const char *path);

/* ── Model lifecycle ── */

mnist_model    *mnist_model_create(struct mem_pool *params_pool);
tensor         *mnist_model_forward(struct mem_pool *scratch, struct module *base, const tensor *x);

mnist_model_cnn *mnist_model_create_cnn(struct mem_pool *params_pool);
tensor          *mnist_model_forward_cnn(struct mem_pool *scratch, struct module *base, const tensor *x);

mnist_model_cnn_pool *mnist_model_create_cnn_pool(struct mem_pool *params_pool);
tensor               *mnist_model_forward_cnn_pool(struct mem_pool *scratch, struct module *base, const tensor *x);

/* ── Training / eval ── */

/* Train model with AdamW. Prints per-epoch loss.
 *
 *   The last val_n samples of train_images/labels are held out as a
 *   validation set for early stopping.  No separate tensors needed.
 *   Set val_n = 0 to skip early stopping.
 *   patience — epochs without val improvement before early stop.
 */
void   mnist_train(struct dnn_ctx *ctx, mnist_model *m,
                   tensor *train_images, tensor *train_labels,
                   int epochs, int batch_size, float lr,
                   int val_n, int patience);

void   mnist_train_cnn(struct dnn_ctx *ctx, mnist_model_cnn *m,
                       tensor *train_images, tensor *train_labels,
                       int epochs, int batch_size, float lr,
                       int val_n, int patience);

void   mnist_train_cnn_pool(struct dnn_ctx *ctx, mnist_model_cnn_pool *m,
                            tensor *train_images, tensor *train_labels,
                            int epochs, int batch_size, float lr,
                            int val_n, int patience);

/* Compute accuracy (0.0 – 1.0) on given dataset. Runs in no-grad mode.
 *
 *   forward_fn — model forward function, receives module* and input, returns logits [N, 10].
 *   model must be castable to module* (module base is first field).
 */
float  mnist_eval(struct mem_pool *scratch, struct module *model,
                  tensor *(*forward_fn)(struct mem_pool *, struct module *, const tensor *),
                  tensor *images, tensor *labels);

#endif /* MNIST_H */
