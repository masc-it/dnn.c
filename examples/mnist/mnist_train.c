#include "mnist.h"
#include "context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ================================================================
 *  Internal helpers
 * ================================================================ */

/* Fisher-Yates shuffle in-place */
static void shuffle_int(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t  = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

/* ================================================================
 *  Training (generic backend, shared by MLP and CNN wrappers)
 * ================================================================ */

void mnist_train_impl(struct dnn_ctx *ctx,
                      tensor *train_images, tensor *train_labels,
                      int epochs, int batch_size, float lr,
                      int val_n, int patience,
                      struct module *model, mnist_forward_fn forward_fn) {
    struct mem_pool *params_pool = ctx->params;
    struct mem_pool *scratch     = ctx->scratch;

    int n_params;
    tensor **params = module_parameters(model, &n_params);

    int N          = tensor_shape(train_images, 0);
    int tr_n       = N - val_n;          /* training sample count */
    int n_batches  = (tr_n + batch_size - 1) / batch_size;
    int use_val    = val_n > 0;

    adamw_opt *opt = adamw_create(params_pool, params, n_params, lr,
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

        struct timespec epoch_t0;
        clock_gettime(CLOCK_MONOTONIC, &epoch_t0);

        for (int b = 0; b < n_batches; b++) {
            int start = b * batch_size;
            int bs    = (b == n_batches - 1) ? tr_n - start : batch_size;

            /* ── build batch tensors in scratch pool ── */
            tensor *bx = tensor_scratch(scratch, 2, (int[]){bs, MNIST_PIXELS}, 0);
            tensor *by = tensor_scratch(scratch, 1, (int[]){bs}, 0);

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
            tensor *logits = forward_fn(scratch, model, bx);
            tensor *loss   = tensor_cross_entropy(scratch, logits, by, 1);

            /* ── backward ── */
            dnn_backward(scratch, loss);

            float loss_val = ((float *)loss->data)[0];
            epoch_loss    += loss_val;

            /* ── update ── */
            adamw_step(opt);
            adamw_zero_grad(opt);

            /* ── free scratch ── */
            mem_pool_reset(scratch);

            if (b > 0 && b % 100 == 0)
                printf("    batch %4d/%d  loss %.6f\r", b, n_batches, loss_val);
        }

        struct timespec epoch_t1;
        clock_gettime(CLOCK_MONOTONIC, &epoch_t1);
        double epoch_sec = (double)(epoch_t1.tv_sec - epoch_t0.tv_sec)
                         + (double)(epoch_t1.tv_nsec - epoch_t0.tv_nsec) / 1e9;
        double batches_per_sec = (double)n_batches / epoch_sec;

        printf("  epoch %3d/%d  loss %.6f",
               epoch + 1, epochs, epoch_loss / n_batches);

        /* ── early-stopping check ── */
        if (use_val) {
            /* manual validation eval — iterate last val_n samples via direct
               offset into the original data buffers (no tensor_slice views) */
            dnn_grad_ctx ctx_grad = dnn_no_grad_enter();
            int vbatch = batch_size < 256 ? batch_size : 256;
            int correct = 0;
            for (int s = tr_n; s < N; s += vbatch) {
                int bs = (s + vbatch > N) ? N - s : vbatch;

                tensor *bx = tensor_scratch(scratch, 2, (int[]){bs, MNIST_PIXELS}, 0);
                tensor *by = tensor_scratch(scratch, 1, (int[]){bs}, 0);

                memcpy(bx->data, img_data + (size_t)s * MNIST_PIXELS,
                       (size_t)bs * MNIST_PIXELS * sizeof(float));
                memcpy(by->data, lbl_data + s,
                       (size_t)bs * sizeof(int));

                tensor *logits = forward_fn(scratch, model, bx);
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

                mem_pool_reset(scratch);
            }
            dnn_no_grad_exit(ctx_grad);

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

        printf("  %.1f batch/s\n", batches_per_sec);
    }

    free(indices);
    if (use_val) {
        for (int i = 0; i < n_params; i++) free(best_bufs[i]);
        free(best_bufs);
        free(best_sizes);
    }
    adamw_free(opt);
}

/* ================================================================
 *  Evaluation
 * ================================================================ */

float mnist_eval(struct mem_pool *scratch, struct module *model,
                  mnist_forward_fn forward_fn,
                  tensor *images, tensor *labels) {
    int N = tensor_shape(images, 0);

    dnn_grad_ctx ctx_grad = dnn_no_grad_enter();

    /* process in batches to avoid huge scratch allocation.
     * 1024 is fine for MLP but CNN activations at 1024 need ~330 MB.
     * 128 keeps CNN peak ~42 MB (fits in 64 MB scratch). */
    int batch_size = 128;
    int correct = 0;

    for (int start = 0; start < N; start += batch_size) {
        int bs = (start + batch_size > N) ? N - start : batch_size;

        /* slice batch */
        tensor *bx = tensor_slice(scratch, images, 0, start, bs);
        tensor *logits = forward_fn(scratch, model, bx);
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
        mem_pool_reset(scratch);
    }

    dnn_no_grad_exit(ctx_grad);
    return (float)correct / (float)N;
}
