#include "mnist.h"
#include "tensor_int.h"
#include "pool_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <zlib.h>

/* ================================================================
 *  Internal helpers
 * ================================================================ */

/* read big-endian 32-bit integer */
static int read_be32(const unsigned char *p) {
    return ((int)p[0] << 24) | ((int)p[1] << 16) | ((int)p[2] << 8) | p[3];
}

/* parse IDX header, fill dims[] with ndim values.  returns 0 on success. */
static int read_idx_header(gzFile f, int *magic, int *dims) {
    unsigned char hdr[16];
    if (gzread(f, hdr, 4) < 4) return -1;
    *magic = read_be32(hdr);
    int ndim = *magic & 0xFF;
    if (gzread(f, hdr, ndim * 4) < ndim * 4) return -1;
    for (int i = 0; i < ndim; i++) dims[i] = read_be32(hdr + i * 4);
    return 0;
}

/* Fisher-Yates shuffle in-place */
static void shuffle_int(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t  = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

/* ================================================================
 *  Data loading
 * ================================================================ */

int mnist_download(void) {
    static const char *urls[] = {
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "train-images-idx3-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "train-labels-idx1-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "t10k-images-idx3-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "t10k-labels-idx1-ubyte.gz",
    };
    static const char *outfiles[] = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    };
    int n = 4;

    for (int i = 0; i < n; i++) {
        /* skip if already downloaded */
        FILE *f = fopen(outfiles[i], "rb");
        if (f) { fclose(f); continue; }

        char cmd[1024];
        int r = snprintf(cmd, sizeof(cmd),
                         "curl -fSL '%s' 2>/dev/null | gunzip > '%s'",
                         urls[i], outfiles[i]);
        (void)r;  /* suppress unused-variable warnings */
        assert(r > 0 && r < (int)sizeof(cmd));

        int ret = system(cmd);
        if (ret != 0) {
            fprintf(stderr, "mnist_download: failed (%d) for %s\n",
                    ret, urls[i]);
            return -1;
        }
    }
    return 0;
}

tensor *mnist_load_images(const char *path) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "mnist_load_images: can't open %s\n", path); return NULL; }

    int magic, dims[4];
    if (read_idx_header(f, &magic, dims) < 0) { gzclose(f); return NULL; }

    int N = dims[0], rows = dims[1], cols = dims[2];
    int n_pixels = N * rows * cols;
    int img_size = rows * cols;  /* 784 */

    /* read raw unsigned-byte pixels */
    unsigned char *raw = malloc(n_pixels);
    if (gzread(f, raw, n_pixels) < n_pixels) {
        free(raw); gzclose(f); return NULL;
    }
    gzclose(f);

    /* create float tensor [N, 784] in data pool, values in [0, 1] */
    tensor *t = tensor_zeros_data(2, (int[]){N, img_size});
    float  *dp = (float *)t->data;
    for (int i = 0; i < n_pixels; i++)
        dp[i] = raw[i] / 255.0f;

    free(raw);
    return t;
}

tensor *mnist_load_labels(const char *path) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "mnist_load_labels: can't open %s\n", path); return NULL; }

    int magic, dims[4];
    if (read_idx_header(f, &magic, dims) < 0) { gzclose(f); return NULL; }

    int N = dims[0];

    unsigned char *raw = malloc(N);
    if (gzread(f, raw, N) < N) { free(raw); gzclose(f); return NULL; }
    gzclose(f);

    /* create int tensor [N] in data pool – store ints in the float data region */
    tensor *t = tensor_zeros_data(1, (int[]){N});
    int    *lp = (int *)t->data;
    for (int i = 0; i < N; i++) lp[i] = raw[i];

    free(raw);
    return t;
}

/* ================================================================
 *  Model
 * ================================================================ */

mnist_model *mnist_model_create(void) {
    mnist_model *m = mem_params_alloc(sizeof(mnist_model), NULL);
    m->fc1 = linear_create(MNIST_PIXELS, 256);
    m->fc2 = linear_create(256, MNIST_CLASSES);
    return m;
}

tensor *mnist_model_forward(mnist_model *m, const tensor *x) {
    tensor *h = linear_forward(m->fc1, x);
    h = tensor_relu(h);
    h = tensor_dropout(h, 0.2f);
    h = linear_forward(m->fc2, h);
    return h;
}

/* ── CNN model ── */

static float kaiming_bound(int in_c, int k_h, int k_w) {
    return sqrtf(6.0f / (float)(in_c * k_h * k_w));
}

mnist_model_cnn *mnist_model_create_cnn(void) {
    mnist_model_cnn *m = mem_params_alloc(sizeof(mnist_model_cnn), NULL);

    /* conv1: 1→32, 3×3 */
    float b1 = kaiming_bound(1, 3, 3);
    m->conv1_w = tensor_uniform(4, (int[]){32, 1, 3, 3}, 1, b1);
    m->conv1_b = tensor_zeros(1, (int[]){32}, 1);

    /* conv2: 32→64, 3×3 */
    float b2 = kaiming_bound(32, 3, 3);
    m->conv2_w = tensor_uniform(4, (int[]){64, 32, 3, 3}, 1, b2);
    m->conv2_b = tensor_zeros(1, (int[]){64}, 1);

    /* conv3: 64→64, 3×3 */
    float b3 = kaiming_bound(64, 3, 3);
    m->conv3_w = tensor_uniform(4, (int[]){64, 64, 3, 3}, 1, b3);
    m->conv3_b = tensor_zeros(1, (int[]){64}, 1);

    /* FC: 3136→128→10 */
    m->fc1 = linear_create(3136, 128);
    m->fc2 = linear_create(128, 10);

    return m;
}

tensor *mnist_model_forward_cnn(mnist_model_cnn *m, const tensor *x) {
    /* x: (N, 784) → reshape to (N, 1, 28, 28) — contiguous view */
    int N = tensor_shape(x, 0);
    tensor *h = tensor_reshape((tensor*)x, 4, (int[]){N, 1, 28, 28});

    h = tensor_conv2d(h, m->conv1_w, m->conv1_b, 1, 1);
    h = tensor_relu(h);

    h = tensor_conv2d(h, m->conv2_w, m->conv2_b, 2, 1);
    h = tensor_relu(h);

    h = tensor_conv2d(h, m->conv3_w, m->conv3_b, 2, 1);
    h = tensor_relu(h);

    h = tensor_reshape(h, 2, (int[]){N, -1});  /* (N, 3136) */
    h = linear_forward(m->fc1, h);
    h = tensor_relu(h);
    h = tensor_dropout(h, 0.5f);
    h = linear_forward(m->fc2, h);
    return h;
}

/* ================================================================
 *  Training (generic backend, shared by MLP and CNN wrappers)
 * ================================================================ */

typedef tensor *(*forward_fn_t)(void *, const tensor *);

static void mnist_train_impl(
    tensor *train_images, tensor *train_labels,
    int epochs, int batch_size, float lr,
    int val_n, int patience,
    tensor **params, int n_params,
    void *model, forward_fn_t forward_fn) {
    int N          = tensor_shape(train_images, 0);
    int tr_n       = N - val_n;          /* training sample count */
    int n_batches  = (tr_n + batch_size - 1) / batch_size;
    int use_val    = val_n > 0;

    adamw_opt *opt = adamw_create(params, n_params, lr,
                                  0.9f, 0.999f, 1e-8f, 0.01f);

    /* early-stopping state */
    float  best_val_acc = -1.0f;
    int    no_improve   = 0;
    float **best_bufs  = NULL;
    int    *best_sizes = NULL;
    if (use_val) {
        best_bufs  = malloc((size_t)n_params * sizeof(float*));
        best_sizes = malloc((size_t)n_params * sizeof(int));
        for (int i = 0; i < n_params; i++) {
            best_sizes[i] = tensor_numel(params[i]);
            best_bufs[i]  = malloc((size_t)best_sizes[i] * sizeof(float));
        }
    }

    /* shuffle index buffer (only over training samples) */
    int *indices = malloc((size_t)tr_n * sizeof(int));

    printf("  N=%d  train=%d  val=%d  batches=%d  batch=%d  epochs=%d  lr=%.4f",
           N, tr_n, val_n, n_batches, batch_size, epochs, lr);
    if (use_val) printf("  patience=%d", patience);
    printf("\n");

    float *img_data = (float *)train_images->data;
    int   *lbl_data = (int   *)train_labels->data;

    int stopped_early = 0;
    for (int epoch = 0; epoch < epochs && !stopped_early; epoch++) {
        /* shuffle training indices */
        for (int i = 0; i < tr_n; i++) indices[i] = i;
        shuffle_int(indices, tr_n);

        double epoch_loss = 0.0;

        for (int b = 0; b < n_batches; b++) {
            int start = b * batch_size;
            int bs    = (b == n_batches - 1) ? tr_n - start : batch_size;

            /* ── build batch tensors in scratch pool ── */
            tensor *bx = _tensor_scratch_create(2, (int[]){bs, MNIST_PIXELS}, 0);
            tensor *by = _tensor_scratch_create(1, (int[]){bs}, 0);

            float *xd = (float *)bx->data;
            int   *yd = (int   *)by->data;

            for (int i = 0; i < bs; i++) {
                int idx = indices[start + i];
                memcpy(xd + i * MNIST_PIXELS,
                       img_data + idx * MNIST_PIXELS,
                       MNIST_PIXELS * sizeof(float));
                yd[i] = lbl_data[idx];
            }

            /* ── forward ── */
            tensor *logits = forward_fn(model, bx);
            tensor *loss   = tensor_cross_entropy(logits, by, 1);

            /* ── backward ── */
            dnn_backward(loss);

            float loss_val = ((float *)loss->data)[0];
            epoch_loss    += loss_val;

            /* ── update ── */
            adamw_step(opt);
            adamw_zero_grad(opt);

            /* ── free scratch ── */
            mem_pool_reset(_mem_pool_scratch());

            if (b > 0 && b % 100 == 0)
                printf("    batch %4d/%d  loss %.6f\r", b, n_batches, loss_val);
        }

        printf("  epoch %3d/%d  loss %.6f",
               epoch + 1, epochs, epoch_loss / n_batches);

        /* ── early-stopping check ── */
        if (use_val) {
            /* manual validation eval — iterate last val_n samples via direct
               offset into the original data buffers (no tensor_slice views) */
            dnn_grad_ctx ctx = dnn_no_grad_enter();
            int vbatch = batch_size < 256 ? batch_size : 256;
            int correct = 0;
            for (int s = tr_n; s < N; s += vbatch) {
                int bs = (s + vbatch > N) ? N - s : vbatch;

                tensor *bx = _tensor_scratch_create(2, (int[]){bs, MNIST_PIXELS}, 0);
                tensor *by = _tensor_scratch_create(1, (int[]){bs}, 0);

                memcpy(bx->data, img_data + (size_t)s * MNIST_PIXELS,
                       (size_t)bs * MNIST_PIXELS * sizeof(float));
                memcpy(by->data, lbl_data + s,
                       (size_t)bs * sizeof(int));

                tensor *logits = forward_fn(model, bx);
                float *ld = (float *)logits->data;
                int   *yd = (int   *)by->data;

                for (int i = 0; i < bs; i++) {
                    int pred = 0;
                    float best = ld[i * MNIST_CLASSES];
                    for (int j = 1; j < MNIST_CLASSES; j++) {
                        if (ld[i * MNIST_CLASSES + j] > best) {
                            best = ld[i * MNIST_CLASSES + j];
                            pred = j;
                        }
                    }
                    if (pred == yd[i]) correct++;
                }

                mem_pool_reset(_mem_pool_scratch());
            }
            dnn_no_grad_exit(ctx);

            float val_acc = (float)correct / (float)val_n;
            printf("  val_acc %.4f", val_acc);

            if (val_acc > best_val_acc) {
                best_val_acc = val_acc;
                no_improve   = 0;
                for (int i = 0; i < n_params; i++)
                    memcpy(best_bufs[i], tensor_data_ptr(params[i]),
                           (size_t)best_sizes[i] * sizeof(float));
            } else {
                no_improve++;
                if (no_improve >= patience) {
                    printf("  early stop (best %.4f)", best_val_acc);
                    stopped_early = 1;
                    for (int i = 0; i < n_params; i++)
                        memcpy(tensor_data_ptr(params[i]), best_bufs[i],
                               (size_t)best_sizes[i] * sizeof(float));
                }
            }
        }

        printf("\n");
    }

    free(indices);
    if (use_val) {
        for (int i = 0; i < n_params; i++) free(best_bufs[i]);
        free(best_bufs);
        free(best_sizes);
    }
    adamw_free(opt);
}

/* ── MLP training wrapper ── */

void mnist_train(mnist_model *m,
                 tensor *train_images, tensor *train_labels,
                 int epochs, int batch_size, float lr,
                 int val_n, int patience) {
    tensor *params[] = {
        m->fc1->weight, m->fc1->bias,
        m->fc2->weight, m->fc2->bias,
    };
    mnist_train_impl(train_images, train_labels,
                     epochs, batch_size, lr, val_n, patience,
                     params, 4, m,
                     (forward_fn_t)mnist_model_forward);
}

/* ── CNN training wrapper ── */

void mnist_train_cnn(mnist_model_cnn *m,
                     tensor *train_images, tensor *train_labels,
                     int epochs, int batch_size, float lr,
                     int val_n, int patience) {
    tensor *params[] = {
        m->conv1_w, m->conv1_b,
        m->conv2_w, m->conv2_b,
        m->conv3_w, m->conv3_b,
        m->fc1->weight, m->fc1->bias,
        m->fc2->weight, m->fc2->bias,
    };
    mnist_train_impl(train_images, train_labels,
                     epochs, batch_size, lr, val_n, patience,
                     params, 10, m,
                     (forward_fn_t)mnist_model_forward_cnn);
}

/* ================================================================
 *  Evaluation
 * ================================================================ */

float mnist_eval_generic(void *model,
                          tensor *(*forward_fn)(void *, const tensor *),
                          tensor *images, tensor *labels) {
    int N = tensor_shape(images, 0);

    dnn_grad_ctx ctx = dnn_no_grad_enter();

    /* process in batches to avoid huge scratch allocation.
     * 1024 is fine for MLP but CNN activations at 1024 need ~330 MB.
     * 128 keeps CNN peak ~42 MB (fits in 64 MB scratch). */
    int batch_size = 128;
    int correct = 0;

    for (int start = 0; start < N; start += batch_size) {
        int bs = (start + batch_size > N) ? N - start : batch_size;

        /* slice batch */
        tensor *bx = tensor_slice(images, 0, start, bs);
        tensor *logits = forward_fn(model, bx);
        /* logits shape: [bs, 10] */

        float *ld = (float *)logits->data;
        int   *lb = (int   *)labels->data;

        for (int i = 0; i < bs; i++) {
            int pred   = 0;
            float best = ld[i * MNIST_CLASSES];
            for (int j = 1; j < MNIST_CLASSES; j++) {
                if (ld[i * MNIST_CLASSES + j] > best) {
                    best = ld[i * MNIST_CLASSES + j];
                    pred = j;
                }
            }
            /* labels tensor has offset 0 b/c it's not sliced in eval */
            if (pred == lb[start + i]) correct++;
        }

        /* reset scratch for next batch (invalidates bx, logits) */
        mem_pool_reset(_mem_pool_scratch());
    }

    dnn_no_grad_exit(ctx);
    return (float)correct / (float)N;
}
