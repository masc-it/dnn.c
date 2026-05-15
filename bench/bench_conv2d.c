/* Conv2D benchmark — baseline before im2col optimization.
 * Output: tab-separated rows for easy parsing.
 *
 * Compile: make bench
 * Usage:   ./bench_conv2d
 */
#include "dnn.h"
#include "conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* time a single conv2d forward+backward, return microseconds */
static double time_conv(int N, int C, int H, int W, int out_C, int kH, int kW,
                         int stride, int pad, int warmup, int trials) {
    double *times = malloc(trials * sizeof(double));
    if (!times) return -1;

    for (int t = -warmup; t < trials; t++) {
        mem_pool params  = mem_pool_create(4 * 1024 * 1024);
        mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
        mem_pool_set_defaults(&params, &scratch, NULL);

        tensor *x = tensor_randn(4, (int[]){N, C, H, W}, 1);
        tensor *w = tensor_randn(4, (int[]){out_C, C, kH, kW}, 1);
        tensor *b = tensor_zeros(1, (int[]){out_C}, 1);

        double t0 = now_us();
        tensor *out = tensor_conv2d(x, w, b, stride, pad);
        tensor *loss = tensor_sum(out, 0);
        dnn_backward(loss);
        double dt = now_us() - t0;

        mem_pool_destroy(&params);
        mem_pool_destroy(&scratch);

        if (t >= 0) times[t] = dt;
    }

    /* median (robust against outliers) */
    for (int i = 0; i < trials; i++)
        for (int j = i + 1; j < trials; j++)
            if (times[i] > times[j]) {
                double tmp = times[i]; times[i] = times[j]; times[j] = tmp;
            }

    double median = (trials % 2)
                      ? times[trials / 2]
                      : (times[trials / 2 - 1] + times[trials / 2]) / 2.0;
    free(times);
    return median;
}

int main(void) {
    printf("# Conv2D benchmark (direct convolution)\n");
    printf("# Column: N C H W out_C kH kW stride pad   median_us  flops_M\n");
    printf("# warmup=2 trials=7\n\n");

    /* Pre-defined configurations */
    int configs[][9] = {
        /* tiny: 1x1 kernel, small feature map */
        {1, 3, 32, 32, 16,  1, 1, 1, 0},
        /* small: 3x3, stride=1, pad=1 (same-spatial) */
        {1, 3, 32, 32, 16,  3, 3, 1, 1},
        /* medium */
        {1, 16, 64, 64, 32, 3, 3, 1, 1},
        /* large-ish but still small for direct conv */
        {1, 3, 128, 128, 8,  3, 3, 1, 1},
        /* stride 2 */
        {1, 3, 64, 64, 16,  3, 3, 2, 0},
        /* bigger kernels */
        {1, 3, 32, 32, 16,  5, 5, 1, 2},
        {1, 3, 32, 32, 16,  7, 7, 1, 3},
        /* batch 4 */
        {4, 3, 32, 32, 16,  3, 3, 1, 1},
        /* batch 8 */
        {8, 3, 32, 32, 8,  3, 3, 1, 1},
    };
    int ncfg = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < ncfg; i++) {
        int *c = configs[i];
        int N=c[0], C=c[1], H=c[2], W=c[3], out_C=c[4], kH=c[5], kW=c[6];
        int stride=c[7], pad=c[8];

        double us = time_conv(N, C, H, W, out_C, kH, kW, stride, pad, 2, 7);
        if (us < 0) { fprintf(stderr, "alloc error\n"); return 1; }

        /* rough FLOP estimate: 2 * N * out_C * H_out * W_out * C * kH * kW */
        int H_out = (H + 2*pad - kH) / stride + 1;
        int W_out = (W + 2*pad - kW) / stride + 1;
        int flops = 2 * N * out_C * H_out * W_out * C * kH * kW;
        double mflops = (double)flops / us;

        printf("%d %d %d %d %d %d %d %d %d   %7.0f  %7.0f\n",
               N, C, H, W, out_C, kH, kW, stride, pad, us, mflops);
    }

    printf("\n# Baseline captured. Date: " __DATE__ " " __TIME__ "\n");
    return 0;
}
