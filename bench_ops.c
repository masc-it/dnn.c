/* Bench ops — measure vectorizable hot-loop throughput.
 *
 * Two sections:
 *   1) Raw kernel benchmarks — compares SIMD vs scalar inline implementations
 *   2) Library-level benchmarks — calls dnn functions (uses whatever SIMD is in lib)
 *
 * Build:  gcc ... bench_ops.c -L. -ldnn -framework Accelerate -lz
 * Usage:  ./bench_ops
 */

#include "dnn.h"
#include "nn.h"
#include "norm.h"
#include "pool.h"
#include "pool_int.h"
#include "tensor_int.h"
#include "simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Compiler barrier — prevents dead-code elimination of benchmark loops.
   Has zero runtime cost; only constrains the optimizer. */
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
 *  Kernel benchmarking — inline SIMD vs scalar implementations
 * ══════════════════════════════════════════════════════════════════ */

/* ── 1. ReLU forward ── */

typedef struct {
    float *out, *in;
    int n;
    int iterations;
} relu_fwd_arg;

static void relu_fwd_run_simd(void *vp) {
    relu_fwd_arg *a = (relu_fwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++)
        simd_relu_fwd(a->out, a->in, a->n);
    BARRIER();
}

static void relu_fwd_run_scalar(void *vp) {
    relu_fwd_arg *a = (relu_fwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++)
        for (int i = 0; i < a->n; i++)
            a->out[i] = a->in[i] > 0.0f ? a->in[i] : 0.0f;
    BARRIER();
}

/* ── 2. ReLU backward ── */

typedef struct {
    float *grad_acc, *in, *grad_out;
    int n;
    int iterations;
} relu_bwd_arg;

static void relu_bwd_run_simd(void *vp) {
    relu_bwd_arg *a = (relu_bwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++)
        simd_relu_bwd(a->grad_acc, a->in, a->grad_out, a->n);
    BARRIER();
}

static void relu_bwd_run_scalar(void *vp) {
    relu_bwd_arg *a = (relu_bwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++)
        for (int i = 0; i < a->n; i++)
            if (a->in[i] > 0.0f)
                a->grad_acc[i] += a->grad_out[i];
    BARRIER();
}

/* ── 3. Cross-entropy forward (single row) ── */

typedef struct {
    float *row;
    int C, target;
    float *result_loss;
    int iterations;
} ce_fwd_arg;

static void ce_fwd_row_simd(void *vp) {
    ce_fwd_arg *a = (ce_fwd_arg*)vp;
    float loss = 0.0f;
    for (int it = 0; it < a->iterations; it++) {
        float mx = simd_reduce_max_f32(a->row, a->C);
        float se = simd_exp_sum_shifted_f32(a->row, a->C, mx);
        loss += logf(se) + mx - a->row[a->target];
    }
    *a->result_loss = loss;
    BARRIER();
}

static void ce_fwd_row_scalar(void *vp) {
    ce_fwd_arg *a = (ce_fwd_arg*)vp;
    float loss = 0.0f;
    for (int it = 0; it < a->iterations; it++) {
        float mx = a->row[0];
        for (int c = 1; c < a->C; c++)
            if (a->row[c] > mx) mx = a->row[c];
        float se = 0.0f;
        for (int c = 0; c < a->C; c++)
            se += expf(a->row[c] - mx);
        loss += logf(se) + mx - a->row[a->target];
    }
    *a->result_loss = loss;
    BARRIER();
}

/* ── 4. Cross-entropy backward (single row) ── */

typedef struct {
    float *grad_row, *logits_row;
    float max_val, sum_exp;
    int target, C;
    float scale;
    int iterations;
} ce_bwd_arg;

#if DNN_HAVE_NEON
static void ce_bwd_row_simd(void *vp) {
    ce_bwd_arg *a = (ce_bwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++)
        simd_ce_bwd_row_kernel(a->grad_row, a->logits_row,
                               a->max_val, a->sum_exp,
                               a->target, a->scale, a->C);
    BARRIER();
}
#endif

static void ce_bwd_row_scalar(void *vp) {
    ce_bwd_arg *a = (ce_bwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++) {
        float mx = a->max_val, se = a->sum_exp;
        int tgt = a->target;
        for (int c = 0; c < a->C; c++) {
            float sm = expf(a->logits_row[c] - mx) / se;
            a->grad_row[c] += (sm - (c == tgt ? 1.0f : 0.0f)) * a->scale;
        }
    }
    BARRIER();
}

/* ── 5. Softmax forward (single row) ── */

typedef struct {
    float *out_row, *in_row;
    int C;
    int iterations;
} softmax_fwd_arg;

#if DNN_HAVE_NEON
static void softmax_fwd_row_simd(void *vp) {
    softmax_fwd_arg *a = (softmax_fwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++) {
        float mx = simd_reduce_max_f32(a->in_row, a->C);
        float se = simd_exp_sum_shifted_f32(a->in_row, a->C, mx);
        float32x4_t vmax = vdupq_n_f32(mx);
        float32x4_t vse  = vdupq_n_f32(se);
        int c = 0;
        for (; c + 4 <= a->C; c += 4) {
            float32x4_t shifted = vsubq_f32(vld1q_f32(a->in_row + c), vmax);
            vst1q_f32(a->out_row + c, vdivq_f32(simd_expf_f32(shifted), vse));
        }
        for (; c < a->C; c++)
            a->out_row[c] = expf(a->in_row[c] - mx) / se;
    }
    BARRIER();
}
#endif

static void softmax_fwd_row_scalar(void *vp) {
    softmax_fwd_arg *a = (softmax_fwd_arg*)vp;
    for (int it = 0; it < a->iterations; it++) {
        float mx = a->in_row[0];
        for (int c = 1; c < a->C; c++)
            if (a->in_row[c] > mx) mx = a->in_row[c];
        float se = 0.0f;
        for (int c = 0; c < a->C; c++)
            se += expf(a->in_row[c] - mx);
        for (int c = 0; c < a->C; c++)
            a->out_row[c] = expf(a->in_row[c] - mx) / se;
    }
    BARRIER();
}

/* ══════════════════════════════════════════════════════════════════
 *  Benchmark runner for inline kernels
 * ══════════════════════════════════════════════════════════════════ */

static double bench_kernel(void (*fn)(void*), void *arg,
                            int warmup, int trials, int iterations) {
    for (int t = 0; t < warmup; t++) fn(arg);
    double *ts = malloc((size_t)trials * sizeof(double));
    for (int t = 0; t < trials; t++) {
        double t0 = now_us();
        fn(arg);
        double dt = now_us() - t0;
        ts[t] = dt;
    }
    double med = median(ts, trials);
    free(ts);
    return med / (double)iterations;
}

/* ══════════════════════════════════════════════════════════════════
 *  Accuracy verification
 * ══════════════════════════════════════════════════════════════════ */

static void verify_accuracy(void) {
    int n_err = 0;
    printf("\n## Accuracy verification (simd_expf vs libm expf)\n");

#if DNN_HAVE_NEON
    srand(42);
    double max_rel_err = 0.0;
    for (int i = 0; i < 10000; i++) {
        float x = (float)rand() / (float)RAND_MAX * 174.0f - 87.0f;
        float32x4_t vx = vdupq_n_f32(x);
        float simd_val = vgetq_lane_f32(simd_expf_f32(vx), 0);
        float ref_val  = expf(x);
        float rel_err  = fabsf(simd_val - ref_val) / (fabsf(ref_val) + 1e-30f);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 1e-4f && ref_val > 1e-10f) {
            if (n_err < 5)
                printf("  x=%8.4f  simd=%12.6e  ref=%12.6e  rel_err=%8.2e\n",
                       x, simd_val, ref_val, rel_err);
            n_err++;
        }
    }
    printf("  max rel err: %.2e  (%d/10000 > 1e-4 threshold)\n",
           max_rel_err, n_err);
#else
    printf("  NEON not available.\n");
#endif
    printf("\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Raw kernel benchmarks
 * ══════════════════════════════════════════════════════════════════ */

static void run_kernel_benchmarks(void) {
    int const N          = 1024 * 1024;    /* 1M elements for relu */
    int const ROW_C      = 10000;           /* row size for CE/softmax */
    int const RELU_ITERS = 100;
    int const ROW_ITERS  = 1000;
    int const WARMUP     = 3;
    int const TRIALS     = 10;

    printf("## Raw kernel benchmarks (SIMD vs scalar)\n");
    printf("  relu_N=%d  ce_C=%d  relu_iters=%d  row_iters=%d  warmup=%d  trials=%d\n\n",
           N, ROW_C, RELU_ITERS, ROW_ITERS, WARMUP, TRIALS);

    float *in       = malloc((size_t)N * sizeof(float));
    float *out      = malloc((size_t)N * sizeof(float));
    float *grad_acc = calloc((size_t)N, sizeof(float));
    float *grad_out = malloc((size_t)N * sizeof(float));
    float *row      = malloc((size_t)ROW_C * sizeof(float));
    float *out_row  = malloc((size_t)ROW_C * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++)          in[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
    for (int i = 0; i < N; i++)          grad_out[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < ROW_C; i++)      row[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;

    typedef struct {
        const char *name;
        void (*simd_fn)(void*);
        void (*scalar_fn)(void*);
        void *arg_simd, *arg_scalar;
        int iterations;
    } kb_t;

    relu_fwd_arg      rf  = {out, in, N, RELU_ITERS};
    relu_bwd_arg      rb  = {grad_acc, in, grad_out, N, RELU_ITERS};
    float ce_loss;
    ce_fwd_arg        cf  = {row, ROW_C, 3, &ce_loss, ROW_ITERS};
    float *grad_rowX  = calloc((size_t)ROW_C, sizeof(float));
    float mx_row = simd_reduce_max_f32(row, ROW_C);
    float se_row = simd_exp_sum_shifted_f32(row, ROW_C, mx_row);
    ce_bwd_arg        cb  = {grad_rowX, row, mx_row, se_row, 3,
                             1.0f / 64.0f, ROW_ITERS};
    softmax_fwd_arg   sf  = {out_row, row, ROW_C, ROW_ITERS};

    kb_t benches[] = {
        {"relu_fwd_1M",      relu_fwd_run_simd,   relu_fwd_run_scalar,   &rf, &rf, RELU_ITERS},
        {"relu_bwd_1M",      relu_bwd_run_simd,   relu_bwd_run_scalar,   &rb, &rb, RELU_ITERS},
        {"ce_fwd_C10K",      ce_fwd_row_simd,     ce_fwd_row_scalar,     &cf, &cf, ROW_ITERS},
#if DNN_HAVE_NEON
        {"ce_bwd_C10K",      ce_bwd_row_simd,     ce_bwd_row_scalar,     &cb, &cb, ROW_ITERS},
        {"softmax_fwd_C10K", softmax_fwd_row_simd, softmax_fwd_row_scalar, &sf, &sf, ROW_ITERS},
#else
        {"ce_bwd_C10K",      ce_bwd_row_scalar,   ce_bwd_row_scalar,     &cb, &cb, ROW_ITERS},
        {"softmax_fwd_C10K", softmax_fwd_row_scalar, softmax_fwd_row_scalar, &sf, &sf, ROW_ITERS},
#endif
    };
    int nbenches = sizeof(benches) / sizeof(benches[0]);

    printf("%-18s  %10s  %10s  %7s  %s\n",
           "kernel", "simd_us", "scalar_us", "speedup", "impl");
    printf("%-18s  %10s  %10s  %7s  %s\n",
           "------", "-------", "--------", "-------", "----");

    for (int i = 0; i < nbenches; i++) {
        kb_t *b = &benches[i];
        memset(grad_acc, 0, (size_t)N * sizeof(float));
        memset(grad_rowX, 0, (size_t)ROW_C * sizeof(float));

        double t_simd   = bench_kernel(b->simd_fn,   b->arg_simd,
                                       WARMUP, TRIALS, b->iterations);
        double t_scalar = bench_kernel(b->scalar_fn, b->arg_scalar,
                                       WARMUP, TRIALS, b->iterations);
        double speedup = t_scalar / t_simd;

        const char *impl = "scalar";
#if DNN_HAVE_NEON
        impl = "NEON";
#else
        impl = "same";
#endif
        if (b->simd_fn == b->scalar_fn) impl = "same";

        printf("%-18s  %10.1f  %10.1f  %7.2fx  %s\n",
               b->name, t_simd, t_scalar, speedup, impl);
    }

    printf("\n  Platform: ");
#if DNN_HAVE_NEON
    printf("ARM64 NEON active\n");
#else
    printf("scalar (%s)\n", "DNN_NO_SIMD or non-ARM");
#endif

    free(in); free(out); free(grad_acc); free(grad_out);
    free(row); free(out_row); free(grad_rowX);
}

/* ══════════════════════════════════════════════════════════════════
 *  Library-level benchmarks (via dnn functions)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct bench_ctx {
    tensor *target;
    tensor *ln_w;
    tensor *ln_b;
    tensor *extra;
} bench_ctx;

static bench_ctx *make_ctx(void) {
    bench_ctx *c = malloc(sizeof(bench_ctx));
    memset(c, 0, sizeof(*c));
    return c;
}

static void free_ctx(bench_ctx *c) { free(c); }

static tensor *do_relu(tensor *x, bench_ctx *c) {
    (void)c;
    return tensor_relu(x);
}

static tensor *do_xent(tensor *x, bench_ctx *c) {
    if (!c->target) {
        int N = tensor_shape(x, 0);
        int *td = malloc((size_t)N * sizeof(int));
        for (int i = 0; i < N; i++) td[i] = rand() % tensor_shape(x, 1);
        c->target = malloc(sizeof(tensor));
        memset(c->target, 0, sizeof(tensor));
        c->target->ndim = 1;
        c->target->shape[0] = N;
        c->target->strides[0] = 1;
        c->target->contiguous = 1;
        c->target->data = td;
    }
    return tensor_cross_entropy(x, c->target, 1);
}

static tensor *do_ln(tensor *x, bench_ctx *c) {
    if (!c->ln_w) {
        int d = tensor_shape(x, tensor_ndim(x) - 1);
        float *wd = malloc((size_t)d * sizeof(float));
        float *bd = malloc((size_t)d * sizeof(float));
        c->ln_w = malloc(sizeof(tensor));
        c->ln_b = malloc(sizeof(tensor));
        memset(c->ln_w, 0, sizeof(tensor));
        memset(c->ln_b, 0, sizeof(tensor));
        c->ln_w->ndim = 1; c->ln_w->shape[0] = d; c->ln_w->data = wd;
        c->ln_b->ndim = 1; c->ln_b->shape[0] = d; c->ln_b->data = bd;
        c->ln_w->strides[0] = 1; c->ln_b->strides[0] = 1;
        c->ln_w->contiguous = 1; c->ln_b->contiguous = 1;
        for (int j = 0; j < d; j++) { wd[j] = 1.0f; bd[j] = 0.0f; }
    }
    return tensor_layer_norm(x, c->ln_w, c->ln_b, 1e-5f);
}

static tensor *do_add(tensor *x, bench_ctx *c) {
    int n = tensor_numel(x);
    if (!c->extra) {
        float *d = malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) d[i] = (float)rand() / RAND_MAX;
        c->extra = malloc(sizeof(tensor));
        memset(c->extra, 0, sizeof(tensor));
        c->extra->ndim = tensor_ndim(x);
        memcpy(c->extra->shape, x->shape, (size_t)tensor_ndim(x) * sizeof(int));
        c->extra->data = d;
        c->extra->contiguous = 1;
        int s = 1;
        for (int i = tensor_ndim(x) - 1; i >= 0; i--) {
            c->extra->strides[i] = s;
            s *= c->extra->shape[i];
        }
    }
    return tensor_add(x, c->extra);
}

static tensor *do_mul(tensor *x, bench_ctx *c) {
    if (!c->extra) {
        int n = tensor_numel(x);
        float *d = malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) d[i] = (float)rand() / RAND_MAX;
        c->extra = malloc(sizeof(tensor));
        memset(c->extra, 0, sizeof(tensor));
        c->extra->ndim = tensor_ndim(x);
        memcpy(c->extra->shape, x->shape, (size_t)tensor_ndim(x) * sizeof(int));
        c->extra->data = d;
        c->extra->contiguous = 1;
        int s = 1;
        for (int i = tensor_ndim(x) - 1; i >= 0; i--) {
            c->extra->strides[i] = s;
            s *= c->extra->shape[i];
        }
    }
    return tensor_mul(x, c->extra);
}

typedef tensor *(*fwd_fn_t)(tensor*, bench_ctx*);

static double bench_fwd(fwd_fn_t fwd, tensor *x, bench_ctx *ctx,
                         int warmup, int trials) {
    double *ts = malloc((size_t)trials * sizeof(double));
    for (int t = -warmup; t < trials; t++) {
        mem_pool_reset(_mem_pool_scratch());
        double t0 = now_us();
        tensor *y = fwd(x, ctx);
        /* sink via volatile read to prevent DCE */
        if (y) { volatile float sink = ((float*)y->data)[0]; (void)sink; }
        double dt = now_us() - t0;
        if (t >= 0) ts[t] = dt;
    }
    double med = median(ts, trials);
    free(ts);
    return med;
}

static double bench_step(fwd_fn_t fwd, tensor *x, bench_ctx *ctx,
                          int warmup, int trials) {
    double *ts = malloc((size_t)trials * sizeof(double));
    for (int t = -warmup; t < trials; t++) {
        mem_pool_reset(_mem_pool_scratch());
        double t0 = now_us();
        tensor *y  = fwd(x, ctx);
        if (y) { dnn_backward(y); volatile float sink = ((float*)y->data)[0]; (void)sink; }
        double dt = now_us() - t0;
        if (t >= 0) ts[t] = dt;
    }
    double med = median(ts, trials);
    free(ts);
    return med;
}

static void run_library_benchmarks(void) {
    typedef struct {
        const char *name;
        int ndim, shape[4];
        fwd_fn_t fwd;
        int wants_grad;
    } bench_cfg;

    bench_cfg cfgs[] = {
        {"relu_1M",  1, {1024*1024},       do_relu, 1},
        {"relu_4M",  2, {1024, 4096},      do_relu, 1},
        {"xent_64x10",   2, {64, 10},      do_xent, 1},
        {"xent_64x1000", 2, {64, 1000},    do_xent, 1},
        {"xent_256x1000",2, {256, 1000},   do_xent, 1},
        {"xent_64_10000",2, {64, 10000},   do_xent, 1},
        {"ln_64x1024",    2, {64, 1024},   do_ln,   1},
        {"ln_128x4096",   2, {128, 4096},  do_ln,   1},
        {"ln_16x4096x64", 3, {16, 4096, 64}, do_ln, 1},
        {"add_1M",  1, {1024*1024},        do_add,  1},
        {"add_4M",  1, {4*1024*1024},      do_add,  1},
        {"mul_1M",  1, {1024*1024},        do_mul,  1},
        {"mul_4M",  1, {4*1024*1024},      do_mul,  1},
    };
    int ncfgs = sizeof(cfgs) / sizeof(cfgs[0]);

    printf("\n## Library-level benchmarks (dnn functions, SIMD via library)\n");
    printf("  warmup=3  trials=10\n\n");
    printf("%-18s  %10s  %8s  %8s  %8s  %7s  %7s  %7s\n",
           "op", "shape", "fwd_us", "bwd_us", "step_us",
           "fwd_GBs", "bwd_GBs", "step_GBs");
    printf("%-18s  %10s  %8s  %8s  %8s  %7s  %7s  %7s\n",
           "--", "-----", "------", "------", "------",
           "-------", "-------", "-------");

    mem_pool params  = mem_pool_create(256 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(256 * 1024 * 1024);
    mem_pool data    = mem_pool_create(64 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    for (int i = 0; i < ncfgs; i++) {
        bench_cfg *c = &cfgs[i];
        mem_pool_reset(&params);
        mem_pool_reset(&scratch);

        tensor *x = tensor_randn(c->ndim, c->shape, c->wants_grad);
        bench_ctx *ctx = make_ctx();
        int numel = tensor_numel(x);
        double bytes_fwd  = (double)numel * sizeof(float) * 2.0;
        double bytes_bwd  = (double)numel * sizeof(float) * 4.0;
        double bytes_step = bytes_fwd + bytes_bwd;

        double fwd_us  = bench_fwd(c->fwd, x, ctx, 3, 10);
        double step_us = bench_step(c->fwd, x, ctx, 3, 10);
        double bwd_us  = step_us - fwd_us;
        if (bwd_us < 0) bwd_us = 0;
        double fwd_gbs  = bytes_fwd / (fwd_us * 1e3);
        double bwd_gbs  = bwd_us > 0 ? bytes_bwd / (bwd_us * 1e3) : 0;
        double step_gbs = bytes_step / (step_us * 1e3);

        char shape_str[32];
        int pos = 0;
        pos += snprintf(shape_str + pos, sizeof(shape_str) - (size_t)pos, "[");
        for (int d = 0; d < c->ndim; d++)
            pos += snprintf(shape_str + pos, sizeof(shape_str) - (size_t)pos,
                            "%s%d", d ? "," : "", c->shape[d]);
        snprintf(shape_str + pos, sizeof(shape_str) - (size_t)pos, "]");

        printf("%-18s %10s  %8.0f  %8.0f  %8.0f  %7.2f  %7.2f  %7.2f\n",
               c->name, shape_str, fwd_us, bwd_us, step_us,
               fwd_gbs, bwd_gbs, step_gbs);
        free_ctx(ctx);
        mem_pool_reset(&params);
    }

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ══════════════════════════════════════════════════════════════════
 *  main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("# bench_ops — hot-loop throughput\n");
    printf("# build: %s %s\n", __DATE__, __TIME__);

    verify_accuracy();
    run_kernel_benchmarks();
    run_library_benchmarks();
    printf("\n# done.\n");
    return 0;
}
