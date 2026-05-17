/* bench_conv2d_vlm — conv2d benchmarks targeting VLM patch-embed.
 *
 * Focuses on the actual conv2d shapes used in imagenet_vlm:
 *   - patch_embed: (B, 3, 224, 224) → (B, 128, 14, 14), kernel=16 stride=16 pad=0
 *   - Also sweeps batch sizes and smaller configs.
 *
 * Measures: forward only, forward+backward, memory bandwidth.
 */

#include "dnn.h"
#include "context.h"
#include "conv.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* ── Timing helper (fwd+bwd) — creates context once, reuses tensors ── */
static double time_fwd_bwd(int N, int C, int H, int W, int out_C,
                            int kH, int kW, int stride, int pad,
                            int warmup, int trials) {
    double *times = malloc(trials * sizeof(double));
    if (!times) return -1;

    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 512*1024*1024, 1024*1024*1024, 512*1024*1024);

    tensor *x = tensor_randn(ctx.params, 4, (int[]){N, C, H, W}, 1);
    tensor *w = tensor_randn(ctx.params, 4, (int[]){out_C, C, kH, kW}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){out_C}, 1);

    for (int t = -warmup; t < trials; t++) {
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        double t0 = now_us();
        tensor *out = tensor_conv2d(ctx.scratch, x, w, b, stride, pad);
        tensor *loss = tensor_sum(ctx.scratch, out, 0);
        dnn_backward(ctx.scratch, loss);
        double dt = now_us() - t0;

        if (t >= 0) times[t] = dt;
    }

    dnn_ctx_destroy(&ctx);

    /* median */
    for (int i = 0; i < trials; i++)
        for (int j = i+1; j < trials; j++)
            if (times[i] > times[j]) {
                double tmp = times[i]; times[i] = times[j]; times[j] = tmp;
            }
    double median = (trials % 2)
        ? times[trials/2]
        : (times[trials/2-1] + times[trials/2]) / 2.0;
    free(times);
    return median;
}

/* Forward-only timing (no-grad) */
static double time_fwd_only(int N, int C, int H, int W, int out_C,
                             int kH, int kW, int stride, int pad,
                             int warmup, int trials) {
    double *times = malloc(trials * sizeof(double));
    if (!times) return -1;

    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 512*1024*1024, 1024*1024*1024, 512*1024*1024);

    tensor *x = tensor_randn(ctx.params, 4, (int[]){N, C, H, W}, 0);
    tensor *w = tensor_randn(ctx.params, 4, (int[]){out_C, C, kH, kW}, 0);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){out_C}, 0);

    dnn_grad_ctx ng = dnn_no_grad_enter();

    for (int t = -warmup; t < trials; t++) {
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        double t0 = now_us();
        tensor *out = tensor_conv2d(ctx.scratch, x, w, b, stride, pad);
        volatile float sink = tensor_data_ptr(out)[0];
        (void)sink;
        double dt = now_us() - t0;

        if (t >= 0) times[t] = dt;
    }

    dnn_no_grad_exit(ng);
    dnn_ctx_destroy(&ctx);

    for (int i = 0; i < trials; i++)
        for (int j = i+1; j < trials; j++)
            if (times[i] > times[j]) {
                double tmp = times[i]; times[i] = times[j]; times[j] = tmp;
            }
    double median = (trials % 2)
        ? times[trials/2]
        : (times[trials/2-1] + times[trials/2]) / 2.0;
    free(times);
    return median;
}

/* Calculate FLOPs for conv2d */
static long long conv_flops(int N, int C, int H, int W, int out_C,
                             int kH, int kW, int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    return 2LL * N * out_C * H_out * W_out * C * kH * kW;
}

/* Memory bandwidth estimate */
static long long conv_memory_bytes(int N, int C, int H, int W, int out_C,
                                    int kH, int kW, int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    long long M = (long long)N * H_out * W_out;
    long long K = (long long)C * kH * kW;
    long long col_bytes = K * M * 4;
    long long input_bytes = (long long)N * C * H * W * 4;
    long long weight_bytes = (long long)out_C * K * 4;
    long long output_bytes = (long long)N * out_C * H_out * W_out * 4;
    return input_bytes + weight_bytes + output_bytes + 2 * col_bytes;
}

int main(void) {
    printf("=== Conv2D VLM-focused Benchmarks ===\n");
    printf("Machine: Apple Silicon (Accelerate BLAS)\n");
    printf("Config: VLM patch_embed = (B,3,224,224)->(B,128,14,14), k=16 s=16 p=0\n\n");

    /* Phase 1: VLM patch-embed config, sweep batch sizes */
    printf("── Phase 1: VLM patch_embed (C=3, H=224, W=224, out_C=128, k=16, s=16, p=0) ──\n");
    printf("%5s  %8s  %8s  %8s  %10s  %10s  %10s\n",
           "B", "fwd_us", "bwd_us", "fwd+bwd", "fwd_GFLOPS", "bwd_GFLOPS", "BW_GB/s");

    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64};
    int C=3, H=224, W=224, out_C=128, k=16, s=16, p=0;
    long long flops = conv_flops(1, C, H, W, out_C, k, k, s, p);
    long long mem = conv_memory_bytes(1, C, H, W, out_C, k, k, s, p);

    for (int bi = 0; bi < 7; bi++) {
        int N = batch_sizes[bi];
        double fwd = time_fwd_only(N, C, H, W, out_C, k, k, s, p, 2, 7);
        double fwd_bwd = time_fwd_bwd(N, C, H, W, out_C, k, k, s, p, 2, 7);
        if (fwd < 0 || fwd_bwd < 0) { fprintf(stderr, "alloc error\n"); return 1; }

        double bwd = fwd_bwd - fwd;
        double fwd_gflops = flops * N / fwd / 1000.0;
        double bwd_gflops = flops * N / bwd / 1000.0;
        double bw = (mem * N / fwd) / 1000.0;

        printf("%5d  %8.0f  %8.0f  %8.0f  %10.1f  %10.1f  %10.1f\n",
               N, fwd, bwd, fwd_bwd, fwd_gflops, bwd_gflops, bw);
    }

    /* Phase 2: kernel size sweep at BS=64 */
    printf("\n── Phase 2: Kernel size sweep (BS=64, C=3, H=224, W=224, out_C=128, s=16, p=auto) ──\n");
    printf("%8s  %8s  %8s  %10s  %10s\n",
           "kernel", "fwd_us", "bwd_us", "fwd_GFLOPS", "BW_GB/s");

    int kernels[] = {3, 5, 7, 11, 16};
    int pads[] = {1, 2, 3, 5, 0};
    for (int ki = 0; ki < 5; ki++) {
        int kk = kernels[ki], pp = pads[ki];
        flops = conv_flops(64, C, H, W, out_C, kk, kk, s, pp);
        mem = conv_memory_bytes(64, C, H, W, out_C, kk, kk, s, pp);
        double fwd = time_fwd_only(64, C, H, W, out_C, kk, kk, s, pp, 2, 5);
        double fwd_bwd = time_fwd_bwd(64, C, H, W, out_C, kk, kk, s, pp, 2, 5);
        if (fwd < 0 || fwd_bwd < 0) { fprintf(stderr, "alloc error\n"); return 1; }
        double bwd = fwd_bwd - fwd;
        double fwd_gflops = flops / fwd / 1000.0;
        double bw = (mem / fwd) / 1000.0;
        printf("kx%d  %8.0f  %8.0f  %10.1f  %10.1f\n",
               kk, fwd, bwd, fwd_gflops, bw);
    }

    /* Phase 3: 3x3 stride=1 vs 3x3 stride=2 (common conv shapes that use Winograd vs im2col) */
    printf("\n── Phase 3: 3x3 conv shapes (BS=64) ──\n");
    printf("%14s  %8s  %8s  %10s  %10s\n",
           "cfg", "fwd_us", "bwd_us", "fwd_GFLOPS", "BW_GB/s");

    int cfgs3[][5] = {
        {64, 3, 56, 56, 32},     /* early conv, small spatial */
        {64, 32, 56, 56, 64},    /* mid conv */
        {64, 64, 28, 28, 128},   /* deeper conv */
        {32, 128, 28, 28, 256},  /* deep conv, smaller batch */
    };
    for (int ci = 0; ci < 3; ci++) {
        int *cfg = cfgs3[ci];
        int N=cfg[0], cc=cfg[1], hh=cfg[2], ww=cfg[3], oc=cfg[4];
        flops = conv_flops(N, cc, hh, ww, oc, 3, 3, 1, 1);
        mem = conv_memory_bytes(N, cc, hh, ww, oc, 3, 3, 1, 1);
        double fwd = time_fwd_only(N, cc, hh, ww, oc, 3, 3, 1, 1, 2, 5);
        double fwd_bwd = time_fwd_bwd(N, cc, hh, ww, oc, 3, 3, 1, 1, 2, 5);
        if (fwd < 0 || fwd_bwd < 0) { fprintf(stderr, "alloc error\n"); return 1; }
        double bwd = fwd_bwd - fwd;
        double fwd_gflops = flops / fwd / 1000.0;
        double bw = (mem / fwd) / 1000.0;
        printf("(%d,%d,%d,%d)->%d  %8.0f  %8.0f  %10.1f  %10.1f\n",
               N, cc, hh, ww, oc, fwd, bwd, fwd_gflops, bw);
    }

    printf("\nDone.\n");
    return 0;
}
