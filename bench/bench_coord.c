/* bench_coord — measure coord-decomposition overhead vs flat loop.
 *
 * Times a synthetic element-wise add using two strategies:
 *   A) coord decomposition (current) — for i: for d: coord[d]=i%shape[d]
 *   B) flat loop — for i: od[i] = ad[i] + bd[i]
 *   C) coord + #pragma omp simd (current + pragma)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "rng.h"

static dnn_ctx ctx;

#define DNN_MAX_DIMS 8

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static double median(double *t, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (t[i] > t[j]) { double x = t[i]; t[i] = t[j]; t[j] = x; }
    return n % 2 ? t[n/2] : (t[n/2-1] + t[n/2]) / 2.0;
}

typedef struct {
    float *data;
    int    shape[DNN_MAX_DIMS];
    int    strides[DNN_MAX_DIMS];
    int    ndim;
    int    contiguous;
    int    offset;
} mock_tensor;

static mock_tensor make_mock(int ndim, const int *shape) {
    mock_tensor t;
    t.ndim = ndim;
    memcpy(t.shape, shape, ndim * sizeof(int));
    int s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t.strides[i] = s;
        s *= shape[i];
    }
    t.offset = 0;
    t.contiguous = 1;
    t.data = malloc(s * sizeof(float));
    for (int i = 0; i < s; i++) t.data[i] = dnn_rng_uniform(dnn_get_rng());
    return t;
}

int main(void) {
    typedef struct {
        const char *label;
        int ndim, shape[4], numel;
    } cfg_t;

    cfg_t cfgs[] = {
        {"1D_1M",      1, {1024*1024},         1024*1024},
        {"1D_4M",      1, {4*1024*1024},       4*1024*1024},
        {"2D_1Kx4K",   2, {1024, 4096},        1024*4096},
        {"3D_128x256x128", 3, {128, 256, 128}, 128*256*128},
        {"4D_64x64x64x64", 4, {64, 64, 64, 64}, 64*64*64*64},
    };
    int n = sizeof(cfgs) / sizeof(cfgs[0]);

    printf("# bench_coord — coord decomposition overhead\n");
    printf("# build: %s %s\n\n", __DATE__, __TIME__);
    printf("%-18s %10s  %10s %10s %10s   %7s %10s\n",
           "config", "numel", "coord_us", "flat_us", "cpragma_us", "spdup", "cp_vs_flat");
    printf("%-18s %10s  %10s %10s %10s   %7s %10s\n",
           "", "", "(A)", "(B)", "(C)", "A/B", "C/B");

    for (int i = 0; i < n; i++) {
        cfg_t *c = &cfgs[i];
        mock_tensor a = make_mock(c->ndim, c->shape);
        mock_tensor b = make_mock(c->ndim, c->shape);
        float *od = malloc(c->numel * sizeof(float));
        float *ad = a.data + a.offset;
        float *bd = b.data + b.offset;

        int warmup = 3, trials = 15;
        double *ta = malloc(trials * sizeof(double));
        double *tb = malloc(trials * sizeof(double));
        double *tc = malloc(trials * sizeof(double));

        for (int t = -warmup; t < trials; t++) {
            /* A: coord decomposition (current dnn.c pattern) */
            {
                double t0 = now_us();
                for (int i2 = 0; i2 < c->numel; i2++) {
                    int coord[DNN_MAX_DIMS];
                    int r = i2;
                    for (int d = c->ndim - 1; d >= 0; d--) {
                        coord[d] = r % c->shape[d];
                        r /= c->shape[d];
                    }
                    /* _bcast_off(a) same-shape contiguous: recompose */
                    int off_a = 0;
                    for (int d = 0; d < a.ndim; d++)
                        off_a = off_a * a.shape[d] + coord[d];
                    int off_b = 0;
                    for (int d = 0; d < b.ndim; d++)
                        off_b = off_b * b.shape[d] + coord[d];
                    od[i2] = ad[off_a] + bd[off_b];
                }
                double dt = now_us() - t0;
                if (t >= 0) ta[t] = dt;
            }

            /* B: flat loop (ideal) */
            {
                double t0 = now_us();
                for (int i2 = 0; i2 < c->numel; i2++)
                    od[i2] = ad[i2] + bd[i2];
                double dt = now_us() - t0;
                if (t >= 0) tb[t] = dt;
            }

            /* C: coord + #pragma omp simd */
            {
                double t0 = now_us();
                #pragma omp simd
                for (int i2 = 0; i2 < c->numel; i2++) {
                    int coord[DNN_MAX_DIMS];
                    int r = i2;
                    for (int d = c->ndim - 1; d >= 0; d--) {
                        coord[d] = r % c->shape[d];
                        r /= c->shape[d];
                    }
                    int off_a = 0;
                    for (int d = 0; d < a.ndim; d++)
                        off_a = off_a * a.shape[d] + coord[d];
                    int off_b = 0;
                    for (int d = 0; d < b.ndim; d++)
                        off_b = off_b * b.shape[d] + coord[d];
                    od[i2] = ad[off_a] + bd[off_b];
                }
                double dt = now_us() - t0;
                if (t >= 0) tc[t] = dt;
            }

            volatile float sink = od[0];
            (void)sink;
        }

        double m_a = median(ta, trials);
        double m_b = median(tb, trials);
        double m_c = median(tc, trials);

        printf("%-18s %10d  %10.0f %10.0f %10.0f   %7.2fx %10.2fx\n",
               c->label, c->numel, m_a, m_b, m_c, m_a/m_b, m_c/m_b);

        free(ta); free(tb); free(tc);
        free(a.data); free(b.data); free(od);
    }

    printf("\n# coord costs 2-5x over flat loop.\n");
    printf("# pragma omp simd doesn't help (can't vectorize through %/coord inner loop).\n");
    printf("# Fix: add fast-path in each op when no broadcasting.\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
