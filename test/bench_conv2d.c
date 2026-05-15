/* Conv2D benchmark — direct convolution baseline.
 *
 * Measures forward + backward for several kernel sizes.
 * Stores baseline timings for comparison against im2col version.
 */
#include "dnn.h"
#include "context.h"
#include "conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static dnn_ctx ctx;

/* high-res wall clock in seconds */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* run a benchmark for one config */
static double bench(int N, int C, int H, int W, int out_C, int kH, int kW,
                    int stride, int pad, int warmup, int iters) {

    dnn_ctx_init(&ctx, 32 * 1024 * 1024, 32 * 1024 * 1024, 8*1024*1024);

    tensor *x = tensor_randn(ctx.params, 4, (int[]){N, C, H, W}, 1);
    tensor *w = tensor_randn(ctx.params, 4, (int[]){out_C, C, kH, kW}, 1);
    tensor *b = tensor_randn(ctx.params, 1, (int[]){out_C}, 1);

    /* warmup */
    for (int i = 0; i < warmup; i++) {
        tensor *out = tensor_conv2d(ctx.scratch, x, w, b, stride, pad);
        tensor *loss = tensor_sum(ctx.scratch, out, 0);
        dnn_backward(ctx.scratch, loss);
        mem_pool_reset(ctx.params);
        mem_pool_reset(ctx.scratch);
        /* re-create input tensors after reset */
        x = tensor_randn(ctx.params, 4, (int[]){N, C, H, W}, 1);
        w = tensor_randn(ctx.params, 4, (int[]){out_C, C, kH, kW}, 1);
        b = tensor_randn(ctx.params, 1, (int[]){out_C}, 1);
    }

    /* timed runs */
    double total = 0.0;
    for (int i = 0; i < iters; i++) {
        mem_pool_reset(ctx.params);
        mem_pool_reset(ctx.scratch);
        x = tensor_randn(ctx.params, 4, (int[]){N, C, H, W}, 1);
        w = tensor_randn(ctx.params, 4, (int[]){out_C, C, kH, kW}, 1);
        b = tensor_randn(ctx.params, 1, (int[]){out_C}, 1);

        double t0 = now_sec();
        tensor *out = tensor_conv2d(ctx.scratch, x, w, b, stride, pad);
        tensor *loss = tensor_sum(ctx.scratch, out, 0);
        dnn_backward(ctx.scratch, loss);
        total += now_sec() - t0;
    }

    return total / iters;
}

int main(void) {
    printf("── Conv2D Baseline (direct convolution) ──\n");
    printf("%-40s %10s\n", "config", "ms/fwd+bwd");
    printf("%s\n", "──────────────────────────────────────────────────────────");

    /* tiny: 1x1 kernel (just a matmul per pixel) */
    /* config                     N   C  H  W  oC kH kW st pd */
    double t;
    t = bench(1,   3, 32, 32,  16, 1, 1, 1, 0, 3, 10);
    printf("%-40s %10.3f\n", "1, 3, 32, 32 -> 16 (1x1 k, s1)", t * 1000);

    t = bench(1,   3, 32, 32,  16, 3, 3, 1, 1, 3, 10);
    printf("%-40s %10.3f\n", "1, 3, 32, 32 -> 16 (3x3 k, s1, p1)", t * 1000);

    t = bench(8,   3, 32, 32,  16, 3, 3, 1, 1, 3, 10);
    printf("%-40s %10.3f\n", "8, 3, 32, 32 -> 16 (3x3 k, s1, p1)", t * 1000);

    t = bench(1,  16, 32, 32,  32, 3, 3, 1, 1, 3, 10);
    printf("%-40s %10.3f\n", "1, 16, 32, 32 -> 32 (3x3 k, s1, p1)", t * 1000);

    t = bench(1,   3, 64, 64,  16, 3, 3, 1, 1, 2,  5);
    printf("%-40s %10.3f\n", "1, 3, 64, 64 -> 16 (3x3 k, s1, p1)", t * 1000);

    t = bench(1,   3, 32, 32,  16, 5, 5, 2, 2, 3, 10);
    printf("%-40s %10.3f\n", "1, 3, 32, 32 -> 16 (5x5 k, s2, p2)", t * 1000);

    t = bench(4,  16, 16, 16,  32, 3, 3, 1, 1, 2,  5);
    printf("%-40s %10.3f\n", "4, 16, 16, 16 -> 32 (3x3 k, s1, p1)", t * 1000);

    t = bench(1, 128, 16, 16, 256, 3, 3, 1, 1, 2,  5);
    printf("%-40s %10.3f\n", "1, 128, 16, 16 -> 256 (3x3 k, s1, p1)", t * 1000);

    printf("\n");

    return 0;
}
