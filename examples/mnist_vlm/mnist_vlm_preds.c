/* ══════════════════════════════════════════════════════════════════
 *  mnist_vlm_preds — interactive VLM prediction on test set
 *
 *  Loads latest checkpoint from examples/mnist_vlm/ckpt/,
 *  then on each Enter press picks 10 random test images
 *  and runs VLM forward to predict the digit.
 * ══════════════════════════════════════════════════════════════════ */

#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "context.h"
#include "tokenizer.h"
#include "mnist_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <dirent.h>
#define TEST_N         10000
#define IMG_ROWS       28
#define IMG_COLS       28
#define IMG_PIXELS     784
#define PATCH_SIZE     7
#define N_IMG_TOK      16
#define VOCAB          261
#define BOS_ID         257

#define D_MODEL        128
#define N_LAYERS       4
#define N_HEADS        4
#define D_K            32
#define INTERMEDIATE   256



/* ══════════════════════════════════════════════════════════════════
 *  Find latest checkpoint
 * ══════════════════════════════════════════════════════════════════ */

static int find_latest_ckpt(char *buf, size_t bufsz) {
    DIR *d = opendir("examples/mnist_vlm/ckpt");
    if (!d) return -1;
    struct dirent *e;
    char best[256] = {0};
    while ((e = readdir(d))) {
        size_t len = strlen(e->d_name);
        if (len > 4 && strcmp(e->d_name + len - 4, ".bin") == 0) {
            if (best[0] == 0 || strcmp(e->d_name, best) > 0)
                snprintf(best, sizeof(best), "%s", e->d_name);
        }
    }
    closedir(d);
    if (best[0] == 0) return -1;
    snprintf(buf, bufsz, "examples/mnist_vlm/ckpt/%s", best);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== MNIST VLM Interactive Predictions ===\n\n");

    /* ── Load test data ── */
    int n_test;
    float *test_img = mnist_load_images(DATA_DIR "/t10k-images-idx3-ubyte", &n_test);
    int   *test_lbl = mnist_load_labels(DATA_DIR "/t10k-labels-idx1-ubyte", &n_test);
    if (!test_img || !test_lbl) {
        fprintf(stderr, "data load failed — run mnist_vlm first to download\n");
        return 1;
    }
    printf("Loaded %d test samples\n", n_test);

    /* ── Find latest checkpoint ── */
    char ckpt_path[256];
    if (find_latest_ckpt(ckpt_path, sizeof(ckpt_path)) < 0) {
        fprintf(stderr, "no checkpoint found in examples/mnist_vlm/ckpt/\n");
        return 1;
    }
    printf("Loading %s\n", ckpt_path);

    /* ── Pools ── */
    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 16*1024*1024, 64*1024*1024, 4*1024*1024);

    /* ── Create VLM with same config ── */
    dnn_seed((uint64_t)time(NULL));
    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB, D_MODEL,
                                       N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
                                       1, IMG_ROWS, IMG_COLS, PATCH_SIZE, 1);
    module_load(&vlm->base, ckpt_path, 1);
    {
        long long n = vision_lm_num_parameters(vlm);
        printf("Loaded %lld params\n", n);
    }

    /* ── Interactive loop ── */
    printf("\nPress Enter to run 10 random predictions (Ctrl+C to quit)\n");
    for (;;) {
        int ch = getchar();
        if (ch == EOF) break;

        int indices[10];
        for (int i = 0; i < 10; i++) indices[i] = dnn_rng_uniform_int(dnn_get_rng(), n_test);

        dnn_grad_ctx ng = dnn_no_grad_enter();
        printf("\n");

        /* Batched forward: [10, 1, 28, 28] images + [10, 1] BOS prompts */
        tensor *imgs = tensor_zeros_data(ctx.data, 4,
                                          (int[]){10, 1, IMG_ROWS, IMG_COLS});
        float *id = tensor_data_ptr(imgs);
        for (int i = 0; i < 10; i++) {
            int idx = indices[i];
            for (int r = 0; r < IMG_ROWS; r++)
                memcpy(id + (size_t)i * IMG_PIXELS + (size_t)r * IMG_COLS,
                       test_img + (size_t)idx * IMG_PIXELS + (size_t)r * IMG_COLS,
                       IMG_COLS * sizeof(float));
        }

        tensor *txt = tensor_zeros_data(ctx.data, 2, (int[]){10, 1});
        int *tid = (int*)tensor_data_ptr(txt);
        for (int i = 0; i < 10; i++) tid[i] = BOS_ID;

        tensor *logits = vision_lm_forward(ctx.scratch, vlm, imgs, txt);
        float *ld = tensor_data_ptr(logits);

        for (int i = 0; i < 10; i++) {
            float *row = ld + (size_t)i * logits->strides[0] + N_IMG_TOK * VOCAB;
            int pred = 0;
            float best = row[0];
            for (int v = 1; v < VOCAB; v++)
                if (row[v] > best) { best = row[v]; pred = v; }
            int pd = (pred >= 48 && pred <= 57) ? pred - 48 : -1;
            printf("  [%4d] true=%d  pred=%d  %s\n",
                   indices[i], test_lbl[indices[i]], pd,
                   pd == test_lbl[indices[i]] ? "OK" : "FAIL");
        }

        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        dnn_no_grad_exit(ng);

        printf("\nPress Enter for next batch (Ctrl+C to quit)\n");
    }

    /* ── Cleanup ── */
    free(test_img);
    free(test_lbl);
    dnn_ctx_destroy(&ctx);

    printf("\nDone.\n");
    return 0;
}
