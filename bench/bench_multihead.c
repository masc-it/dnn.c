/* bench_multihead — benchmark _to_bhnd and _to_bnhd kernel throughput.
 *
 * Compares four implementations:
 *   1) baseline  — current 4-nested-loop with recomputed indices
 *   2) hoisted   — precomputed partial offsets, pointer increments
 *   3) memcpy    — block-copy innermost d_k via memcpy
 *   4) omp       — OpenMP parallel over (batch × heads)
 *
 * Build:  gcc ... bench_multihead.c -L. -ldnn -framework Accelerate -lz
 * Usage:  ./bench_multihead
 */

#include "dnn.h"
#include "pool.h"
#include "pool_int.h"
#include "tensor_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#define BARRIER() __asm__ volatile("" : : : "memory")

/* ══════════════════════════════════════════════════════════════════
 *  Timing helpers
 * ══════════════════════════════════════════════════════════════════ */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double*)a - *(const double*)b;
    return (x > 0) - (x < 0);
}

static double median(double *t, int n) {
    qsort(t, (size_t)n, sizeof(double), cmp_dbl);
    return n % 2 ? t[n/2] : (t[n/2-1] + t[n/2]) / 2.0;
}

/* ══════════════════════════════════════════════════════════════════
 *  Implementations
 * ══════════════════════════════════════════════════════════════════ */

/* ── Baseline: current code with full index recomputation ── */
static void _to_bhnd_baseline(int B, int N, int H, int d_k,
                               const float *restrict src, float *restrict dst) {
    for (int b = 0; b < B; b++)
        for (int n = 0; n < N; n++)
            for (int h = 0; h < H; h++)
                for (int d = 0; d < d_k; d++)
                    dst[(b * H + h) * N * d_k + n * d_k + d] =
                        src[(b * N + n) * H * d_k + h * d_k + d];
}

static void _to_bnhd_baseline(int B, int H, int N, int d_k,
                               const float *restrict src, float *restrict dst) {
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int n = 0; n < N; n++)
                for (int d = 0; d < d_k; d++)
                    dst[(b * N + n) * H * d_k + h * d_k + d] =
                        src[(b * H + h) * N * d_k + n * d_k + d];
}

/* ── Hoisted: precompute partial offsets, pointer increments ── */
static void _to_bhnd_hoisted(int B, int N, int H, int d_k,
                              const float *restrict src, float *restrict dst) {
    int Hd_k = H * d_k;
    int Nd_k = N * d_k;

    for (int b = 0; b < B; b++) {
        const float *b_src = src + b * N * Hd_k;
        float *b_dst = dst + b * H * Nd_k;
        for (int n = 0; n < N; n++) {
            const float *n_src = b_src + n * Hd_k;
            for (int h = 0; h < H; h++) {
                const float *s = n_src + h * d_k;
                float *d = b_dst + h * Nd_k + n * d_k;
                for (int i = 0; i < d_k; i++)
                    d[i] = s[i];
            }
        }
    }
}

static void _to_bnhd_hoisted(int B, int H, int N, int d_k,
                              const float *restrict src, float *restrict dst) {
    int Nd_k = N * d_k;
    int Hd_k = H * d_k;

    for (int b = 0; b < B; b++) {
        const float *b_src = src + b * H * Nd_k;
        float *b_dst = dst + b * N * Hd_k;
        for (int h = 0; h < H; h++) {
            const float *h_src = b_src + h * Nd_k;
            for (int n = 0; n < N; n++) {
                const float *s = h_src + n * d_k;
                float *d = b_dst + n * Hd_k + h * d_k;
                for (int i = 0; i < d_k; i++)
                    d[i] = s[i];
            }
        }
    }
}

/* ── memcpy: use block copy for innermost d_k elements ── */
static void _to_bhnd_memcpy(int B, int N, int H, int d_k,
                             const float *restrict src, float *restrict dst) {
    int Hd_k = H * d_k;
    int Nd_k = N * d_k;
    size_t nbytes = (size_t)d_k * sizeof(float);

    for (int b = 0; b < B; b++) {
        const float *b_src = src + b * N * Hd_k;
        float *b_dst = dst + b * H * Nd_k;
        for (int n = 0; n < N; n++) {
            const float *n_src = b_src + n * Hd_k;
            float *n_dst_off = b_dst + n * d_k;
            for (int h = 0; h < H; h++)
                memcpy(n_dst_off + h * Nd_k,
                       n_src + h * d_k, nbytes);
        }
    }
}

static void _to_bnhd_memcpy(int B, int H, int N, int d_k,
                             const float *restrict src, float *restrict dst) {
    int Nd_k = N * d_k;
    int Hd_k = H * d_k;
    size_t nbytes = (size_t)d_k * sizeof(float);

    for (int b = 0; b < B; b++) {
        const float *b_src = src + b * H * Nd_k;
        float *b_dst = dst + b * N * Hd_k;
        for (int h = 0; h < H; h++) {
            const float *h_src = b_src + h * Nd_k;
            float *h_dst_off = b_dst + h * d_k;
            for (int n = 0; n < N; n++)
                memcpy(h_dst_off + n * Hd_k,
                       h_src + n * d_k, nbytes);
        }
    }
}

/* ── OpenMP: parallelize over batch × heads ──
 *   collapse(2) gives B*H work units — each thread handles a
 *   subset of (batch, head) pairs, keeping d_k copies local.
 */
static void _to_bhnd_omp(int B, int N, int H, int d_k,
                          const float *restrict src, float *restrict dst) {
    int Hd_k = H * d_k;
    int Nd_k = N * d_k;

    #pragma omp parallel for collapse(2) if (B * H > 1)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            const float *b_src = src + b * N * Hd_k;
            float *b_dst = dst + b * H * Nd_k;
            for (int n = 0; n < N; n++) {
                const float *s = b_src + n * Hd_k + h * d_k;
                float *d = b_dst + h * Nd_k + n * d_k;
                for (int i = 0; i < d_k; i++)
                    d[i] = s[i];
            }
        }
}

static void _to_bnhd_omp(int B, int H, int N, int d_k,
                          const float *restrict src, float *restrict dst) {
    int Nd_k = N * d_k;
    int Hd_k = H * d_k;

    #pragma omp parallel for collapse(2) if (B * H > 1)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            const float *b_src = src + b * H * Nd_k;
            float *b_dst = dst + b * N * Hd_k;
            for (int n = 0; n < N; n++) {
                const float *s = b_src + h * Nd_k + n * d_k;
                float *d = b_dst + n * Hd_k + h * d_k;
                for (int i = 0; i < d_k; i++)
                    d[i] = s[i];
            }
        }
}

/* ══════════════════════════════════════════════════════════════════
 *  Correctness check — all impls produce same output
 * ══════════════════════════════════════════════════════════════════ */

static int check_eq(const float *a, const float *b, int n, const char *label) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("  MISMATCH %s[%d]: %.8f != %.8f\n", label, i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

static void verify(void) {
    printf("## Correctness check\n");

    int shapes[][4] = {
        {1, 2, 2, 2},
        {2, 3, 4, 5},
        {1, 4, 8, 16},
    };
    int nshapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int s = 0; s < nshapes; s++) {
        int B = shapes[s][0], H = shapes[s][1], N = shapes[s][2], d_k = shapes[s][3];
        int n = B * N * H * d_k;

        float *src  = malloc((size_t)n * sizeof(float));
        float *ref  = calloc((size_t)n, sizeof(float));
        float *tst  = calloc((size_t)n, sizeof(float));
        float *tmp  = calloc((size_t)n, sizeof(float));
        float *rref = calloc((size_t)n, sizeof(float));
        float *rtst = calloc((size_t)n, sizeof(float));
        for (int i = 0; i < n; i++) src[i] = (float)(i % 997);

        /* _to_bhnd: compare OMP vs baseline */
        _to_bhnd_baseline(B, N, H, d_k, src, ref);
        _to_bhnd_omp(B, N, H, d_k, src, tst);
        char l1[64];
        snprintf(l1, sizeof(l1), "_to_bhnd [%d,%d,%d,%d]", B, H, N, d_k);
        int ok1 = check_eq(ref, tst, n, l1);
        printf("  %s ... %s\n", l1, ok1 ? "OK" : "FAIL");

        /* _to_bnhd: compare OMP vs baseline */
        _to_bhnd_baseline(B, N, H, d_k, src, tmp);
        _to_bnhd_baseline(B, H, N, d_k, tmp, rref);
        _to_bnhd_omp(B, H, N, d_k, tmp, rtst);
        char l2[64];
        snprintf(l2, sizeof(l2), "_to_bnhd [%d,%d,%d,%d]", B, N, H, d_k);
        int ok2 = check_eq(rref, rtst, n, l2);
        printf("  %s ... %s\n", l2, ok2 ? "OK" : "FAIL");

        /* roundtrip */
        int ok3 = check_eq(rref, src, n, "roundtrip (bhnd→bnhd)");
        printf("  roundtrip bhnd→bnhd [%d,%d,%d,%d] ... %s\n",
               B, H, N, d_k, ok3 ? "OK" : "FAIL");

        free(src); free(ref); free(tst); free(tmp); free(rref); free(rtst);
        if (!ok1 || !ok2 || !ok3) { printf("  CORRECTNESS FAILED\n"); exit(1); }
    }
    printf("  ALL CORRECT\n\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Benchmarking
 * ══════════════════════════════════════════════════════════════════ */

typedef void (*copy_fn_t)(int, int, int, int,
                           const float *restrict, float *restrict);

typedef struct {
    const char *name;
    copy_fn_t fn;
} impl_entry;

/* Bench one impl: warmup + trials, return median us.
 * Resets dst to zero before each call to force cold cache. */
static double bench_one(copy_fn_t fn, int B, int H, int N, int d_k,
                         const float *src, float *dst,
                         int warmup, int trials) {
    size_t nbytes = (size_t)B * N * H * d_k * sizeof(float);
    double *ts = malloc((size_t)trials * sizeof(double));
    for (int t = -warmup; t < trials; t++) {
        memset(dst, 0, nbytes);
        BARRIER();
        double t0 = now_us();
        fn(B, H, N, d_k, src, dst);
        double dt = now_us() - t0;
        if (t >= 0) ts[t] = dt;
    }
    double med = median(ts, trials);
    free(ts);
    return med;
}

static void bench_config(const char *name, int B, int H, int N, int d_k) {
    int elems = B * N * H * d_k;
    float *src = malloc((size_t)elems * sizeof(float));
    float *dst = malloc((size_t)elems * sizeof(float));
    for (int i = 0; i < elems; i++) src[i] = (float)(i % 997);

    impl_entry bhnd[] = {
        {"baseline", _to_bhnd_baseline},
        {"hoisted",  _to_bhnd_hoisted},
        {"memcpy",   _to_bhnd_memcpy},
        {"omp",      _to_bhnd_omp},
    };
    impl_entry bnhd[] = {
        {"baseline", _to_bnhd_baseline},
        {"hoisted",  _to_bnhd_hoisted},
        {"memcpy",   _to_bnhd_memcpy},
        {"omp",      _to_bnhd_omp},
    };
    int n = sizeof(bhnd) / sizeof(bhnd[0]);

    printf("  %s\n", name);
    printf("    shape=[B=%d, H=%d, N=%d, d_k=%d]  elems=%d  size=%.1f MB\n",
           B, H, N, d_k, elems,
           (double)elems * sizeof(float) / (1024.0 * 1024.0));

    /* _to_bhnd */
    printf("    _to_bhnd (split heads):\n");
    double base_bhnd = 0;
    for (int i = 0; i < n; i++) {
        double us = bench_one(bhnd[i].fn, B, H, N, d_k, src, dst, 5, 15);
        double gbs = (double)elems * sizeof(float) * 2.0 / (us * 1e3);
        if (i == 0) base_bhnd = us;
        printf("      %-10s  %8.0f us  %7.2f GB/s  %5.2f× vs baseline\n",
               bhnd[i].name, us, gbs, base_bhnd / us);
    }

    /* _to_bnhd */
    printf("    _to_bnhd (merge heads):\n");
    double base_bnhd = 0;
    for (int i = 0; i < n; i++) {
        double us = bench_one(bnhd[i].fn, B, H, N, d_k, src, dst, 5, 15);
        double gbs = (double)elems * sizeof(float) * 2.0 / (us * 1e3);
        if (i == 0) base_bnhd = us;
        printf("      %-10s  %8.0f us  %7.2f GB/s  %5.2f× vs baseline\n",
               bnhd[i].name, us, gbs, base_bnhd / us);
    }
    printf("\n");

    free(src);
    free(dst);
}

/* ══════════════════════════════════════════════════════════════════
 *  main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("# bench_multihead — split/merge head layout transform\n\n");

    verify();

    printf("## Throughput (15 trials, warmup=5)\n");
    printf("  GB/s = 2 × numel × sizeof(float) / time  (read src + write dst)\n\n");

    bench_config("Tiny (toy)",                  1, 2,   16,  8);
    bench_config("Small (debug)",               2, 4,   64,  16);
    bench_config("Medium (BERT-small)",          8, 8,   128, 32);
    bench_config("Large (BERT-base)",            16, 12, 256, 64);
    bench_config("XL (GPT-2 medium)",            32, 16, 512, 64);
    bench_config("XXL (Llama 3 8B)",             64, 8,  512, 128);
    bench_config("Max (large batch LLM)",        128, 8, 1024, 128);

    printf("# done.\n");
    return 0;
}
