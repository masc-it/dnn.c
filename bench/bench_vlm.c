#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "context.h"
#include "optim.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static dnn_ctx ctx;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench_one(const char *label, int I, int T,
                      int d_model, int n_layers, int n_heads, int d_k,
                      int intermediate, int n_iters) {
    printf("\n=== %s: I=%d T=%d D=%d layers=%d ===\n", label, I, T, d_model, n_layers);

    int vocab = 261, C = 3, patch = 16;
    int pp = (int)sqrtf((float)I);
    if (pp < 1) pp = 1;
    while (pp * pp < I) pp++;
    int H_ = pp * patch, W_ = ((I + pp - 1) / pp) * patch;

    dnn_ctx_init(&ctx, 128*1024*1024, 512*1024*1024, 64*1024*1024);
    dnn_seed(0);

    vision_lm *vlm = vision_lm_create(ctx.params, vocab, d_model, n_layers,
                                       n_heads, d_k, intermediate,
                                       C, H_, W_, patch, 0);
    vision_lm_init_weights(vlm);

    /* Create image + text in data pool so they survive scratch resets */
    tensor *images = tensor_zeros_data(ctx.data, 4, (int[]){1, C, H_, W_});
    {
        tensor *tmp = tensor_randn(ctx.scratch, 4, (int[]){1, C, H_, W_}, 0);
        memcpy(tensor_data_ptr(images), tensor_data_ptr(tmp),
               (size_t)(C * H_ * W_) * sizeof(float));
    }

    int txt_ids[256];
    for (int i = 0; i < T; i++) txt_ids[i] = dnn_rng_uniform_int(dnn_get_rng(), vocab);
    tensor *text_ids = tensor_zeros_data(ctx.data, 2, (int[]){1, T});
    memcpy(tensor_data_ptr(text_ids), txt_ids, T * sizeof(int));

    double t0, total;

    /* Warm-up */
    vision_lm_image_embeds(ctx.scratch, vlm, images);
    mem_pool_reset(ctx.scratch);

    /* ── Patch embedding ── */
    t0 = now_sec();
    for (int it = 0; it < n_iters; it++) {
        vision_lm_image_embeds(ctx.scratch, vlm, images);
        mem_pool_reset(ctx.scratch);
    }
    total = now_sec() - t0;
    printf("  patch_embed:    %.3f ms\n", total / n_iters * 1000.0);

    /* Warm-up forward */
    vision_lm_forward_text_logits(ctx.scratch, vlm, images, text_ids);
    mem_pool_reset(ctx.scratch);

    /* ── Forward text logits ── */
    t0 = now_sec();
    for (int it = 0; it < n_iters; it++) {
        vision_lm_forward_text_logits(ctx.scratch, vlm, images, text_ids);
        mem_pool_reset(ctx.scratch);
    }
    total = now_sec() - t0;
    printf("  forward_text:   %.3f ms\n", total / n_iters * 1000.0);

    dnn_ctx_destroy(&ctx);
}

int main(void) {
    printf("=== VLM Benchmarks ===\n");

    bench_one("smoke",  16,  32,  128, 2, 4,  32, 256, 20);
    bench_one("small", 196,  64,  256, 4, 8,  32, 512, 10);
    bench_one("medium",196, 128,  512, 8, 8,  64,1024, 5);

    printf("\nDone.\n");
    return 0;
}
