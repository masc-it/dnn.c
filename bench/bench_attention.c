#include "dnn.h"
#include "attention.h"
#include "context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static dnn_ctx ctx;

/* ── Attention benchmark ──
 *
 * Benchmarks tensor_attention() fwd+bwd for various N and d
 * with the triangular tile implementation.
 *
 * Measures wall-clock time for attention only (no LM head).
 */

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Scratch size needed for training: P [B,H,N,N] + tile bufs */
static size_t scratch_needed(int B, int H, int N) {
    size_t P_bytes  = (size_t)B * H * N * N * sizeof(float);
    size_t tile_bytes = (size_t)omp_get_max_threads() * DNN_ATTENTION_TILE_ROWS * N * sizeof(float) * 2;
    size_t overhead = 64 * 1024 * 1024;
    return P_bytes + tile_bytes + overhead;
}

/* Benchmark one config: forward only, then forward+backward. */
static void bench_config(int B, int H, int N, int d, int warmup, int iters) {
    /* 3 tensors × (data + grad) × 2 safety */
    size_t param_sz  = (size_t)B * H * N * d * 3 * sizeof(float) * 6 + 4 * 1024 * 1024;
    size_t scratch_sz = scratch_needed(B, H, N);
    size_t data_sz    = 4 * 1024 * 1024;

    dnn_ctx_init(&ctx, param_sz, scratch_sz, data_sz);

    /* Create random Q, K, V */
    srand(123);
    tensor *q = tensor_randn(ctx.params, 4, (int[]){B, H, N, d}, 1);
    tensor *k = tensor_randn(ctx.params, 4, (int[]){B, H, N, d}, 1);
    tensor *v = tensor_randn(ctx.params, 4, (int[]){B, H, N, d}, 1);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        mem_pool_reset(ctx.scratch);
        tensor *out = tensor_attention(ctx.scratch, q, k, v, NULL);
        (void)out;
    }

    /* Measure forward (no-grad) */
    dnn_grad_ctx no_grad = dnn_no_grad_enter();
    double fwd_min = 1e9;
    for (int i = 0; i < iters; i++) {
        mem_pool_reset(ctx.scratch);
        double t0 = wall_time();
        tensor *out = tensor_attention(ctx.scratch, q, k, v, NULL);
        double elapsed = wall_time() - t0;
        if (elapsed < fwd_min) fwd_min = elapsed;
        (void)out;
    }
    dnn_no_grad_exit(no_grad);

    /* Measure forward+backward (training step) */
    double step_min = 1e9;
    for (int i = 0; i < iters; i++) {
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        /* Zero grads before each step */
        tensor_zero_grad(q);
        tensor_zero_grad(k);
        tensor_zero_grad(v);

        double t0 = wall_time();
        tensor *out = tensor_attention(ctx.scratch, q, k, v, NULL);
        dnn_backward(ctx.scratch, out);
        double elapsed = wall_time() - t0;
        if (elapsed < step_min) step_min = elapsed;
    }

    printf("  B=%d H=%d N=%d d=%d  "
           "fwd: %7.3f ms  step: %7.3f ms\n",
           B, H, N, d,
           fwd_min * 1000, step_min * 1000);

    dnn_ctx_destroy(&ctx);
}

int main(void) {
    printf("═══ Attention benchmark (triangular tile, TB=%d) ═══\n\n", DNN_ATTENTION_TILE_ROWS);
    printf("%-30s  %12s  %12s\n", "Config", "forward", "fwd+bwd");
    printf("%-30s  %12s  %12s\n", "------", "-------", "-------");

    int warmup = 3, iters = 10;

    /* Small GPT debug config */
    bench_config(1, 4, 128,  64, warmup, iters);

    /* Normal training config */
    bench_config(2, 8,  512,  64, warmup, iters);

    /* Long context */
    bench_config(1, 12, 1024, 64, warmup, iters);
    bench_config(1, 12, 2048, 64, warmup, iters);

    /* Edge cases */
    bench_config(1, 1, 1,   4,  warmup, iters);    /* N=1 */
    bench_config(1, 1, 7,   16, warmup, iters);     /* N < TB */
    bench_config(1, 1, 64,  16, warmup, iters);     /* N == TB */
    bench_config(1, 1, 65,  16, warmup, iters);     /* N == TB+1 */
    bench_config(1, 1, 128, 16, warmup, iters);     /* N == 2*TB */

    /* Stress: smaller N, no 4096 to keep scratch reasonable */
    bench_config(1, 8,  2048, 128, warmup, iters);

    return 0;
}
