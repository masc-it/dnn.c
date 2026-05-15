#include "dnn.h"
#include "context.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

static dnn_ctx ctx;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* ── Batched matmul benchmark: time fwd + bwd ──
 *
 * Shapes tested:
 *   [B, M, K] @ [B, K, N]
 */
static double time_batched(int B, int M, int K, int N, int warmup, int trials) {
    double *t = malloc(trials * sizeof(double));
    for (int tr = -warmup; tr < trials; tr++) {

    dnn_ctx_init(&ctx, 512 * 1024 * 1024, 512 * 1024 * 1024, 512 * 1024 * 1024);

        tensor *a = tensor_randn(ctx.params, 3, (int[]){B, M, K}, 1);
        tensor *b = tensor_randn(ctx.params, 3, (int[]){B, K, N}, 1);

        double t0 = now_us();
        tensor *c = tensor_matmul(ctx.scratch, a, b);
        tensor *l = tensor_sum(ctx.scratch, c, 0);
        dnn_backward(ctx.scratch, l);
        double dt = now_us() - t0;

        if (tr >= 0) t[tr] = dt;
    }
    /* median */
    for (int i = 0; i < trials; i++)
        for (int j = i + 1; j < trials; j++)
            if (t[i] > t[j]) { double tmp = t[i]; t[i] = t[j]; t[j] = tmp; }
    double med = trials % 2 ? t[trials / 2] : (t[trials / 2 - 1] + t[trials / 2]) / 2.0;
    free(t);
    return med;
}

/* ── Broadcast a: [M, K] @ [B, K, N] ── */
static double time_broadcast_a(int B, int M, int K, int N, int warmup, int trials) {
    double *t = malloc(trials * sizeof(double));
    for (int tr = -warmup; tr < trials; tr++) {

    dnn_ctx_init(&ctx, 512 * 1024 * 1024, 512 * 1024 * 1024, 512 * 1024 * 1024);

        tensor *a = tensor_randn(ctx.params, 2, (int[]){M, K}, 1);
        tensor *b = tensor_randn(ctx.params, 3, (int[]){B, K, N}, 1);

        double t0 = now_us();
        tensor *c = tensor_matmul(ctx.scratch, a, b);
        tensor *l = tensor_sum(ctx.scratch, c, 0);
        dnn_backward(ctx.scratch, l);
        double dt = now_us() - t0;

        if (tr >= 0) t[tr] = dt;
    }
    for (int i = 0; i < trials; i++)
        for (int j = i + 1; j < trials; j++)
            if (t[i] > t[j]) { double tmp = t[i]; t[i] = t[j]; t[j] = tmp; }
    double med = trials % 2 ? t[trials / 2] : (t[trials / 2 - 1] + t[trials / 2]) / 2.0;
    free(t);
    return med;
}

/* ── 4D batched: [B1,B2,M,K] @ [B1,B2,K,N] ── */
static double time_4d(int B1, int B2, int M, int K, int N, int warmup, int trials) {
    double *t = malloc(trials * sizeof(double));
    for (int tr = -warmup; tr < trials; tr++) {

    dnn_ctx_init(&ctx, 512 * 1024 * 1024, 512 * 1024 * 1024, 512 * 1024 * 1024);

        tensor *a = tensor_randn(ctx.params, 4, (int[]){B1, B2, M, K}, 1);
        tensor *b = tensor_randn(ctx.params, 4, (int[]){B1, B2, K, N}, 1);

        double t0 = now_us();
        tensor *c = tensor_matmul(ctx.scratch, a, b);
        tensor *l = tensor_sum(ctx.scratch, c, 0);
        dnn_backward(ctx.scratch, l);
        double dt = now_us() - t0;

        if (tr >= 0) t[tr] = dt;
    }
    for (int i = 0; i < trials; i++)
        for (int j = i + 1; j < trials; j++)
            if (t[i] > t[j]) { double tmp = t[i]; t[i] = t[j]; t[j] = tmp; }
    double med = trials % 2 ? t[trials / 2] : (t[trials / 2 - 1] + t[trials / 2]) / 2.0;
    free(t);
    return med;
}

/* ── 2D reference: [M, K] @ [K, N] (existing benchmark) ── */
static double time_2d(int M, int K, int N, int warmup, int trials) {
    double *t = malloc(trials * sizeof(double));
    for (int tr = -warmup; tr < trials; tr++) {

    dnn_ctx_init(&ctx, 512 * 1024 * 1024, 512 * 1024 * 1024, 512 * 1024 * 1024);

        tensor *a = tensor_randn(ctx.params, 2, (int[]){M, K}, 1);
        tensor *b = tensor_randn(ctx.params, 2, (int[]){K, N}, 1);

        double t0 = now_us();
        tensor *c = tensor_matmul(ctx.scratch, a, b);
        tensor *l = tensor_sum(ctx.scratch, c, 0);
        dnn_backward(ctx.scratch, l);
        double dt = now_us() - t0;

        if (tr >= 0) t[tr] = dt;
    }
    for (int i = 0; i < trials; i++)
        for (int j = i + 1; j < trials; j++)
            if (t[i] > t[j]) { double tmp = t[i]; t[i] = t[j]; t[j] = tmp; }
    double med = trials % 2 ? t[trials / 2] : (t[trials / 2 - 1] + t[trials / 2]) / 2.0;
    free(t);
    return med;
}

int main(void) {
    /* ── Configurations representing realistic transformer sizes ── */
    struct { int B, M, K, N; const char *label; } cfgs[] = {
        /* small: B=1 seq=64  d_model=256  intermediate=1024 */
        {1,   64,   256,  256,   "B1_M64_K256_N256"},
        /* single token, medium width */
        {1,   1,    512,  512,   "B1_M1_K512_N512"},
        /* typical transformer: B=8 seq=512 d=512 */
        {8,   512,  512,  512,   "B8_M512_K512_N512"},
        /* FFN intermediate: B=8 seq=512 d=512→2048 */
        {8,   512,  512,  2048,  "B8_M512_K512_N2048"},
        {8,   512,  2048, 512,   "B8_M512_K2048_N512"},
        /* larger batch */
        {32,  128,  256,  256,   "B32_M128_K256_N256"},
        /* attention QK^T: B=8 H=8 N=64 d=64 */
        {64,  64,   64,   64,    "B64_M64_K64_N64"},
        /* wide projection */
        {2,   1,    4096, 4096,  "B2_M1_K4096_N4096"},
        /* 4D: B1=2 B2=4 H=8 N=64 d=64 */
        {0,   0,    0,    0,     NULL},  /* sentinel */
    };

    printf("=== dnn.c batched matmul benchmark ===\n");
    printf("%-28s  %10s  %10s  %8s\n", "config", "c_fwd+bwd_us", "gflop/s", "batch_flops");
    printf("-------------------------------------------------------------------\n");

    /* 2D reference */
    int two_d_cfgs[][3] = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };
    for (int i = 0; i < 3; i++) {
        int M = two_d_cfgs[i][0], K = two_d_cfgs[i][1], N = two_d_cfgs[i][2];
        double us = time_2d(M, K, N, 3, 10);
        double flops = 2.0 * M * K * N;  /* fwd + bwd: each matmul counted once */
        printf("%-28s  %10.0f  %10.2f  %8s\n", "2D ref", us, flops / (us * 1e3), "");
    }

    printf("-------------------------------------------------------------------\n");

    for (int i = 0; cfgs[i].label != NULL; i++) {
        int B = cfgs[i].B, M = cfgs[i].M, K = cfgs[i].K, N = cfgs[i].N;
        double us = time_batched(B, M, K, N, 3, 10);
        double flops = 2.0 * B * M * K * N;  /* fwd + bwd */
        printf("%-28s  %10.0f  %10.2f  %7.1fG\n",
               cfgs[i].label, us, flops / (us * 1e3), flops / 1e9);
    }

    printf("-------------------------------------------------------------------\n");

    /* Broadcast-a benchmark */
    printf("--- Broadcast a: [M,K] @ [B,K,N] ---\n");
    int bcast_cfgs[][4] = {
        {8, 512, 512, 512},
        {8, 512, 512, 2048},
        {32, 128, 256, 256},
    };
    for (int i = 0; i < 3; i++) {
        int B = bcast_cfgs[i][0], M = bcast_cfgs[i][1], K = bcast_cfgs[i][2], N = bcast_cfgs[i][3];
        double us = time_broadcast_a(B, M, K, N, 3, 10);
        double flops = 2.0 * B * M * K * N;
        printf("bcast_a B%-2d M%d K%d N%d  %10.0f  %10.2f  %7.1fG\n",
               B, M, K, N, us, flops / (us * 1e3), flops / 1e9);
    }

    printf("--- 4D: [B1,B2,M,K] @ [B1,B2,K,N] ---\n");
    int cfg4d[][5] = {
        {2, 4, 64, 64, 64},
        {2, 8, 128, 64, 64},
    };
    for (int i = 0; i < 2; i++) {
        int B1 = cfg4d[i][0], B2 = cfg4d[i][1], M = cfg4d[i][2], K = cfg4d[i][3], N = cfg4d[i][4];
        double us = time_4d(B1, B2, M, K, N, 3, 10);
        double flops = 2.0 * B1 * B2 * M * K * N;
        printf("4d B1=%d B2=%d M%d K%d N%d  %10.0f  %10.2f  %7.1fG\n",
               B1, B2, M, K, N, us, flops / (us * 1e3), flops / 1e9);
    }

    dnn_ctx_destroy(&ctx);


    return 0;
}
