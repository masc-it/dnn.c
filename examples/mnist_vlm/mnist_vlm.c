/* ══════════════════════════════════════════════════════════════════
 *  mnist_vlm — VLM-based MNIST digit classification
 *
 *  Trains a tiny Vision-Language Model to "caption" MNIST digits:
 *  given image → generate correct digit byte token.
 *
 *  Architecture (prefix-LM decoder-only VLM):
 *    image [B,1,28,28] → patch_embed (conv2d k=7,s=7) → [B,16,D]
 *    text  [B,T]       → token_embed                   → [B,T,D]
 *    cat → [B,16+T,D] → N×GPT blocks (prefix-LM attn)
 *                      → norm → tied lm_head → logits [B,16+T,V]
 *
 *  Text format: [BOS][digit_byte][EOS], T=3
 *  Input IDs:   [BOS, digit_byte]   (T-1=2)
 *  Target IDs:  [digit_byte, EOS]   (T-1=2)
 *  Digit bytes: '0'=48 … '9'=57  |  BOS=257  EOS=258  vocab=261
 *
 *  Training: full-text CE over 2 target positions, gradient clipping,
 *            cosine LR schedule with linear warmup.
 *  Eval: forward-based argmax at position I+0 (predicts digit after BOS).
 * ══════════════════════════════════════════════════════════════════ */

#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "context.h"
#include "optim.h"
#include "tokenizer.h"
#include "mnist_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>

/* ── Config ── */
#define TRAIN_N        60000
#define TEST_N         10000
#define IMG_ROWS       28
#define IMG_COLS       28
#define IMG_PIXELS     784
#define PATCH_SIZE     7              /* 28/7=4 → I=16 patches */
#define N_IMG_TOK      16
#define TEXT_LEN       3              /* BOS + digit_byte + EOS */
#define TEXT_TRAIN_LEN 2              /* BOS + digit_byte (input), digit_byte + EOS (target) */
#define VOCAB          261            /* byte-level tokenizer */
#define BOS_ID         257
#define EOS_ID         258

/* Model */
#define D_MODEL        128
#define N_LAYERS       4
#define N_HEADS        4
#define D_K            32             /* D_MODEL / N_HEADS */
#define INTERMEDIATE   256

/* Training */
#define BATCH_SIZE     128
#define MAX_EPOCHS     10
#define LR             5e-4f
#define MIN_LR         1e-5f
#define GRAD_CLIP      1.0f
#define VAL_N          5000
#define PATIENCE       2
#define LOG_EVERY      50
#define EVAL_EVERY     1

/* ══════════════════════════════════════════════════════════════════
 *  Data download
 * ══════════════════════════════════════════════════════════════════ */

static int download_mnist(void) {
    int r = system("mkdir -p " DATA_DIR);
    if (r) return -1;

    const char *urls[] = {
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "train-images-idx3-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "train-labels-idx1-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "t10k-images-idx3-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "t10k-labels-idx1-ubyte.gz",
    };
    const char *names[] = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    };

    for (int i = 0; i < 4; i++) {
        char path[256];
        snprintf(path, sizeof(path), DATA_DIR "/%s", names[i]);
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); continue; }
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "curl -fSL '%s' 2>/dev/null | gunzip > '%s'", urls[i], path);
        int ret = system(cmd);
        if (ret) return -1;
    }
    return 0;
}



/* ══════════════════════════════════════════════════════════════════
 *  mkdir -p: create path component by component
 * ══════════════════════════════════════════════════════════════════ */

static void mkdir_p(const char *path) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(buf, 0755);
            *p = '/';
        }
    }
    mkdir(buf, 0755);
}

/* ══════════════════════════════════════════════════════════════════
 *  Helpers
 * ══════════════════════════════════════════════════════════════════ */

static void shuffle_int(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = dnn_rng_uniform_int(dnn_get_rng(), i + 1);
        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

/* Fill input_ids + target_ids tensors from labels + indices.
 * input_ids  [bs, 2] = [BOS, digit_byte]
 * target_ids [bs, 2] = [digit_byte, EOS] */
static void fill_text_batch(int *input_data, int *target_data,
                            const int *labels, int batch_size,
                            const int *indices) {
    for (int i = 0; i < batch_size; i++) {
        int d = labels[indices[i]];
        input_data[i*2+0] = BOS_ID;
        input_data[i*2+1] = 48 + d;
        target_data[i*2+0] = 48 + d;
        target_data[i*2+1] = EOS_ID;
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Single-sample eval: print predicted digit vs label
 * ══════════════════════════════════════════════════════════════════ */

static void eval_print_sample(struct mem_pool *scratch, struct mem_pool *data,
                               vision_lm *vlm,
                               const float *img_flat, int label,
                               int idx) {
    tensor *img = tensor_zeros_data(data, 4,
                                    (int[]){1, 1, IMG_ROWS, IMG_COLS});
    float *id = tensor_data_ptr(img);
    for (int r = 0; r < IMG_ROWS; r++)
        memcpy(id + (size_t)r * IMG_COLS,
               img_flat + (size_t)idx * IMG_PIXELS + (size_t)r * IMG_COLS,
               IMG_COLS * sizeof(float));

    tensor *txt = tensor_zeros_data(data, 2, (int[]){1, 1});
    ((int*)tensor_data_ptr(txt))[0] = BOS_ID;

    tensor *logits = vision_lm_forward(scratch, vlm, img, txt);
    float *row = tensor_data_ptr(logits) + N_IMG_TOK * VOCAB;

    int pred = 0;
    float best = row[0];
    for (int v = 1; v < VOCAB; v++)
        if (row[v] > best) { best = row[v]; pred = v; }

    int pd = (pred >= 48 && pred <= 57) ? pred - 48 : -1;
    printf("    [%d] true=%d  pred=%d  %s\n",
           idx, label, pd, pd == label ? "OK" : "FAIL");

    mem_pool_reset(scratch);
    mem_pool_reset(data);
}

/* ══════════════════════════════════════════════════════════════════
 *  Forward-based evaluation
 *
 *  Runs BOS → logits at position I+0 → argmax → check digit byte.
 *  Much faster and more reliable than autoregressive generation.
 * ══════════════════════════════════════════════════════════════════ */

static float eval_forward(struct mem_pool *scratch, struct mem_pool *data,
                           vision_lm *vlm,
                           const float *img_flat, const int *labels,
                           int n_total, int start, int count) {
    dnn_grad_ctx no_grad = dnn_no_grad_enter();
    int correct = 0, processed = 0;

    for (int s = start; s < start + count && s < n_total; s += BATCH_SIZE) {
        int bs = (s + BATCH_SIZE > start + count) ? (start + count - s) : BATCH_SIZE;
        if (bs > n_total - s) bs = n_total - s;

        tensor *img = tensor_scratch(scratch, 4, (int[]){bs, 1, IMG_ROWS, IMG_COLS}, 0);
        float *id = tensor_data_ptr(img);
        for (int i = 0; i < bs; i++) {
            int idx = s + i;
            for (int r = 0; r < IMG_ROWS; r++)
                memcpy(id + (size_t)i * IMG_PIXELS + (size_t)r * IMG_COLS,
                       img_flat + (size_t)idx * IMG_PIXELS + (size_t)r * IMG_COLS,
                       IMG_COLS * sizeof(float));
        }

        tensor *txt = tensor_scratch(scratch, 2, (int[]){bs, 1}, 0);
        int *tid = (int*)tensor_data_ptr(txt);
        for (int i = 0; i < bs; i++) tid[i] = BOS_ID;

        tensor *logits = vision_lm_forward(scratch, vlm, img, txt);
        float *ld = tensor_data_ptr(logits);

        for (int i = 0; i < bs; i++) {
            float *row = ld + (size_t)(N_IMG_TOK + 0) * VOCAB + (size_t)i * logits->strides[0];
            /* Find max */
            int pred_id = 0;
            float best = row[0];
            for (int v = 1; v < VOCAB; v++) {
                if (row[v] > best) { best = row[v]; pred_id = v; }
            }
            if (pred_id >= 48 && pred_id <= 57) {
                if (pred_id - 48 == labels[s + i]) correct++;
            }
            processed++;
        }
        mem_pool_reset(scratch);
    }

    dnn_no_grad_exit(no_grad);
    return (float)correct / (float)processed;
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== MNIST VLM ===\n\n");

    /* ── Download data ── */
    printf("Downloading MNIST...\n");
    if (download_mnist() < 0) { fprintf(stderr, "download failed\n"); return 1; }

    /* ── Load data ── */
    int n_train, n_test;
    float *train_img = mnist_load_images(DATA_DIR "/train-images-idx3-ubyte", &n_train);
    int   *train_lbl = mnist_load_labels(DATA_DIR "/train-labels-idx1-ubyte", &n_train);
    float *test_img  = mnist_load_images(DATA_DIR "/t10k-images-idx3-ubyte", &n_test);
    int   *test_lbl  = mnist_load_labels(DATA_DIR "/t10k-labels-idx1-ubyte", &n_test);
    if (!train_img || !train_lbl || !test_img || !test_lbl) {
        fprintf(stderr, "data load failed\n"); return 1;
    }

    printf("Loaded %d train, %d test samples\n", n_train, n_test);
    printf("Image: %dx%d  Patch: %d  Patches: %d\n",
           IMG_ROWS, IMG_COLS, PATCH_SIZE, N_IMG_TOK);

    int tr_n = n_train - VAL_N;  /* training set size (rest is validation) */

    /* ── Pools ── */
    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 16*1024*1024, 256*1024*1024, 32*1024*1024);

    /* ── Create VLM ── */
    printf("Creating VLM (D=%d, L=%d, H=%d, d_k=%d, FF=%d)...\n",
           D_MODEL, N_LAYERS, N_HEADS, D_K, INTERMEDIATE);
    dnn_seed(42);

    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB, D_MODEL,
                                       N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
                                       1, IMG_ROWS, IMG_COLS, PATCH_SIZE, 1);
    vision_lm_init_weights(vlm);

    {
        long long n = vision_lm_num_parameters(vlm);
        printf("  Parameters: %lld (%.2fM)\n", n, n / 1e6);
        module_summary(&vlm->base, 0, 0);
    }

    /* ── Optimizer ── */
    int n_params;
    tensor **all_params = module_parameters(&vlm->base, &n_params);
    printf("  %d param groups.\n", n_params);

    adamw_opt *opt = adamw_create(ctx.params, all_params, n_params, LR,
                                   0.9f, 0.999f, 1e-8f, 1e-4f);

    /* ── LR scheduler (warmup + cosine) ── */
    int n_batches = (tr_n + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_steps = n_batches * MAX_EPOCHS;
    int warmup_steps = n_batches;  /* 1 epoch linear warmup */

    lr_scheduler *sched = lr_scheduler_create(ctx.params, opt,
                                               LR_SCHEDULE_LINEAR_WARMUP_COSINE,
                                               LR, warmup_steps, total_steps,
                                               MIN_LR, 0, 0);
    printf("  LR scheduler: warmup=%d steps, cosine decay over %d steps, base_lr=%.2e, min_lr=%.2e\n",
           warmup_steps, total_steps, LR, MIN_LR);

    /* ── Training ── */
    printf("\nTraining (AdamW, lr=%.2e, batch=%d, max_epochs=%d):\n",
           LR, BATCH_SIZE, MAX_EPOCHS);
    printf("  train=%d  val=%d  test=%d  batches/epoch=%d\n",
           tr_n, VAL_N, n_test, n_batches);

    int *indices = malloc((size_t)tr_n * sizeof(int));
    int step = 0;

    float best_val_acc = 0.0f;
    int no_improve = 0;

    /* Early-stopping weight backup */
    float **best_w = malloc((size_t)n_params * sizeof(float*));
    int *best_sz = malloc((size_t)n_params * sizeof(int));
    for (int i = 0; i < n_params; i++) {
        best_sz[i] = tensor_numel(all_params[i]);
        best_w[i] = malloc((size_t)best_sz[i] * sizeof(float));
    }

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        /* Shuffle training indices */
        for (int i = 0; i < tr_n; i++) indices[i] = i;
        shuffle_int(indices, tr_n);

        double epoch_loss = 0.0;
        int epoch_batches = 0;
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int b = 0; b < n_batches; b++) {
            int start = b * BATCH_SIZE;
            int bs = (b == n_batches - 1) ? tr_n - start : BATCH_SIZE;

            /* Image batch [bs, 1, 28, 28] */
            tensor *img = tensor_scratch(ctx.scratch, 4,
                                         (int[]){bs, 1, IMG_ROWS, IMG_COLS}, 0);
            float *id = tensor_data_ptr(img);
            for (int i = 0; i < bs; i++) {
                int idx = indices[start + i];
                memcpy(id + (size_t)i * IMG_PIXELS,
                       train_img + (size_t)idx * IMG_PIXELS,
                       IMG_PIXELS * sizeof(float));
            }

            /* Text tensors [bs, 2] */
            tensor *input_ids = tensor_scratch(ctx.scratch, 2,
                                               (int[]){bs, TEXT_TRAIN_LEN}, 0);
            tensor *target_ids = tensor_scratch(ctx.scratch, 2,
                                                (int[]){bs, TEXT_TRAIN_LEN}, 0);
            fill_text_batch((int*)tensor_data_ptr(input_ids),
                            (int*)tensor_data_ptr(target_ids),
                            train_lbl, bs, indices + start);

            /* Train step: vision_lm_train_step uses global clip_grad_norm */
            float grad_norm;
            tensor *loss = vision_lm_train_step(ctx.scratch, vlm, img, input_ids, target_ids, opt, GRAD_CLIP, &grad_norm);

            float lv = tensor_data_ptr(loss)[0];
            lr_scheduler_step(sched);
            epoch_loss += lv;
            epoch_batches++;
            step++;

            /* Log every LOG_EVERY batches */
            if ((b + 1) % LOG_EVERY == 0 || b == 0 || b == n_batches - 1) {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (double)(now.tv_sec - t0.tv_sec)
                               + (double)(now.tv_nsec - t0.tv_nsec) / 1e9;
                double batch_s = (double)(b + 1) / elapsed;
                float cur_lr = lr_scheduler_get_lr(sched);

                printf("  epoch %2d/%d  batch %4d/%d  loss %.6f  lr %.2e  gn %.4e  %.1f batch/s\n",
                       epoch + 1, MAX_EPOCHS, b + 1, n_batches,
                       epoch_loss / epoch_batches, cur_lr, grad_norm, batch_s);
            }

            mem_pool_reset(ctx.scratch);
            mem_pool_reset(ctx.data);
        }

        /* ── Validation ── */
        float val_acc = eval_forward(ctx.scratch, ctx.data, vlm,
                                      train_img, train_lbl, n_train, tr_n, VAL_N);

        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        float cur_lr = lr_scheduler_get_lr(sched);

        printf("  ── epoch %2d/%d  avg loss %.6f  val_acc %.4f  lr %.2e  %.2fs  %.1f batch/s\n",
               epoch + 1, MAX_EPOCHS, epoch_loss / epoch_batches,
               val_acc, cur_lr, secs, n_batches / secs);

        /* 2 random validation samples */
        {
            int s1 = tr_n + dnn_rng_uniform_int(dnn_get_rng(), VAL_N);
            int s2 = tr_n + dnn_rng_uniform_int(dnn_get_rng(), VAL_N);
            dnn_grad_ctx ng = dnn_no_grad_enter();
            eval_print_sample(ctx.scratch, ctx.data, vlm,
                              train_img, train_lbl[s1], s1);
            eval_print_sample(ctx.scratch, ctx.data, vlm,
                              train_img, train_lbl[s2], s2);
            dnn_no_grad_exit(ng);
        }

        /* ── Save checkpoint ── */
        {
            time_t now = time(NULL);
            struct tm *tm = localtime(&now);
            char ts[32];
            strftime(ts, sizeof(ts), "%Y%m%dT%H%M%S", tm);
            char path[128];
            snprintf(path, sizeof(path),
                     "examples/mnist_vlm/ckpt/%s.bin", ts);
            mkdir_p("examples/mnist_vlm/ckpt");
            module_save(&vlm->base, path);
            printf("  ── saved %s\n", path);
        }

        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            no_improve = 0;
            for (int i = 0; i < n_params; i++)
                memcpy(best_w[i], tensor_data_ptr(all_params[i]),
                       (size_t)best_sz[i] * sizeof(float));
        } else {
            no_improve++;
            if (no_improve >= PATIENCE) {
                printf("  ── early stop at epoch %d (best val_acc=%.4f)\n",
                       epoch + 1, best_val_acc);
                for (int i = 0; i < n_params; i++)
                    memcpy(tensor_data_ptr(all_params[i]), best_w[i],
                           (size_t)best_sz[i] * sizeof(float));
                break;
            }
        }
    }

    /* ── Restore best weights ── */
    for (int i = 0; i < n_params; i++)
        memcpy(tensor_data_ptr(all_params[i]), best_w[i],
               (size_t)best_sz[i] * sizeof(float));

    /* ── Test evaluation ── */
    printf("\n=== Test ===\n");
    float test_acc = eval_forward(ctx.scratch, ctx.data, vlm,
                                   test_img, test_lbl, n_test, 0, n_test);
    printf("Test accuracy: %.4f  (%d/%d)\n", test_acc,
           (int)(test_acc * n_test), n_test);

    /* ── Sample predictions ── */
    printf("\n=== Samples ===\n");
    {
        dnn_grad_ctx ng = dnn_no_grad_enter();
        for (int s = 0; s < 10; s++)
            eval_print_sample(ctx.scratch, ctx.data, vlm,
                              test_img, test_lbl[s], s);
        dnn_no_grad_exit(ng);
    }

    /* ── Cleanup ── */
    for (int i = 0; i < n_params; i++) free(best_w[i]);
    free(best_w); free(best_sz);
    free(indices);
    free(train_img); free(train_lbl);
    free(test_img);  free(test_lbl);
    adamw_free(opt);
    dnn_ctx_destroy(&ctx);

    printf("\nDone.\n");
    return 0;
}
