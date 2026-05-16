/* bench_cat — cat_forward / cat_backward / _cat_copy throughput
 *
 * Transformer-sized shapes: KV cache append, GLU gated-FFN concat,
 * multi-head split combos, large batch dim=0 concat.
 *
 * Build:  make bench_cat
 * Usage:  ./build/bench_cat
 */

#include "dnn.h"
#include "context.h"
#include "pool.h"
#include "ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static dnn_ctx ctx;

#define BARRIER() __asm__ volatile("" : : : "memory")

/* ── Timing ── */

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

/* ── Benchmark helpers ── */

typedef struct {
    int ndim;
    int shape_a[8];
    int shape_b[8];
    int dim;
    int wants_grad;
    const char *label;
} cat_cfg;

/* Allocate two tensors + run cat forward, measure median time */
static double bench_cat_fwd(const cat_cfg *cfg, int warmup, int trials) {
    tensor *a = tensor_randn(ctx.params, cfg->ndim, cfg->shape_a, cfg->wants_grad);
    tensor *b = tensor_randn(ctx.params, cfg->ndim, cfg->shape_b, cfg->wants_grad);

    double *ts = malloc((size_t)trials * sizeof(double));
    for (int t = -warmup; t < trials; t++) {
        mem_pool_reset(ctx.scratch);
        double t0 = now_us();
        tensor *y = tensor_cat(ctx.scratch, a, b, cfg->dim);
        BARRIER();
        volatile float sink = y ? ((float*)y->data)[0] : 0.0f; (void)sink;
        double dt = now_us() - t0;
        if (t >= 0) ts[t] = dt;
    }
    double med = median(ts, trials);
    free(ts);
    return med;
}

/* Forward + backward step */
static double bench_cat_step(const cat_cfg *cfg, int warmup, int trials) {
    double *ts = malloc((size_t)trials * sizeof(double));
    for (int t = -warmup; t < trials; t++) {
        mem_pool_reset(ctx.params);
        mem_pool_reset(ctx.scratch);
        tensor *ai = tensor_randn(ctx.params, cfg->ndim, cfg->shape_a, cfg->wants_grad);
        tensor *bi = tensor_randn(ctx.params, cfg->ndim, cfg->shape_b, cfg->wants_grad);

        double t0 = now_us();
        tensor *y = tensor_cat(ctx.scratch, ai, bi, cfg->dim);
        dnn_backward(ctx.scratch, y);
        BARRIER();
        volatile float sink = y ? ((float*)y->data)[0] : 0.0f; (void)sink;
        double dt = now_us() - t0;
        if (t >= 0) ts[t] = dt;
    }
    double med = median(ts, trials);
    free(ts);
    return med;
}

/* Non-contiguous (transposed) variant: transpose inputs to force strided path */
static double bench_cat_fwd_strided(const cat_cfg *cfg, int warmup, int trials) {
    /* Allocate contiguous base tensors from params pool (outlives scratch resets) */
    tensor *a_base = tensor_randn(ctx.params, cfg->ndim, cfg->shape_a, cfg->wants_grad);
    tensor *b_base = tensor_randn(ctx.params, cfg->ndim, cfg->shape_b, cfg->wants_grad);
    /* Cat dim after swapping last two dims */
    int cat_dim = cfg->dim;
    if (cat_dim == cfg->ndim-1) cat_dim = cfg->ndim-2;
    else if (cat_dim == cfg->ndim-2) cat_dim = cfg->ndim-1;

    double *ts = malloc((size_t)trials * sizeof(double));
    for (int t = -warmup; t < trials; t++) {
        mem_pool_reset(ctx.scratch);
        /* Create transposed views fresh each iter (tensor struct lives in scratch) */
        tensor *a = tensor_transpose(ctx.scratch, a_base, cfg->ndim-1, cfg->ndim-2);
        tensor *b = tensor_transpose(ctx.scratch, b_base, cfg->ndim-1, cfg->ndim-2);
        double t0 = now_us();
        tensor *y = tensor_cat(ctx.scratch, a, b, cat_dim);
        BARRIER();
        volatile float sink = y ? ((float*)y->data)[0] : 0.0f; (void)sink;
        double dt = now_us() - t0;
        if (t >= 0) ts[t] = dt;
    }
    double med = median(ts, trials);
    free(ts);
    return med;
}

/* ── Main ── */

int main(void) {
    printf("# bench_cat — cat forward / backward throughput\n");
    printf("# Transformer-sized shapes\n");
    printf("# build: %s %s\n\n", __DATE__, __TIME__);

    /* Pool sizes: big enough for [4096,11008] tensors + grads */
    dnn_ctx_init(&ctx, 1536 * 1024 * 1024, 1024 * 1024 * 1024, 128 * 1024 * 1024);

    int warmup = 3;
    int trials = 10;

    /* ── Benchmark configs ──
     *
     * Scenario             ndim  shape_a              shape_b              dim  grad  label
     * ───────────────────────────────────────────────────────────────────────────────────────
     * KV cache (GPT-2)       3   [1,128,768]          [1,1,768]            1    1     cat_kv_128x768
     * KV cache (7B)          3   [1,1024,4096]        [1,1,4096]           1    1     cat_kv_1Kx4K
     * KV cache (7B full)     3   [1,4096,4096]        [1,1,4096]           1    1     cat_kv_4Kx4K
     * KV cache (13B full)    3   [1,4096,5120]        [1,1,5120]           1    1     cat_kv_4Kx5K
     * KV cache 2D (GPT-2)    2   [128,768]            [1,768]              0    1     cat_kv2d_128x768
     * KV cache 2D (7B)       2   [4096,4096]          [1,4096]             0    1     cat_kv2d_4Kx4K
     * GLU FFN (GPT-2)        2   [768,3072]           [768,3072]           1    1     cat_glu_768x3K
     * GLU FFN (7B)           2   [4096,11008]         [4096,11008]         1    1     cat_glu_4Kx11K
     * Multi-head split (7B)  4   [4,128,16,128]       [4,128,16,128]       2    1     cat_heads_4Kx128
     * Large dim=0            2   [4096,4096]          [4096,4096]          0    1     cat_vstack_4Kx4K
     */
    cat_cfg cfgs[] = {
        /* 1) KV cache append along seq dim (dim=1 for 3D: [B, N, d]) */
        {3, {1,128,768},   {1,1,768},   1, 1, "cat_kv_128x768"},
        {3, {1,1024,4096}, {1,1,4096},  1, 1, "cat_kv_1Kx4K"},
        {3, {1,4096,4096}, {1,1,4096},  1, 1, "cat_kv_4Kx4K"},
        {3, {1,4096,5120}, {1,1,5120},  1, 1, "cat_kv_4Kx5K"},

        /* 2) KV cache 2D (legacy: [N, d] with dim=0) */
        {2, {128,768},     {1,768},     0, 1, "cat_kv2d_128x768"},
        {2, {4096,4096},   {1,4096},    0, 1, "cat_kv2d_4Kx4K"},

        /* 3) GLU gated FFN: concat gate & up projections along dim=1 */
        {2, {768,3072},    {768,3072},  1, 1, "cat_glu_768x3K"},
        {2, {4096,11008},  {4096,11008},1, 1, "cat_glu_4Kx11K"},
        /* 4) Multi-head split: concat along head dim */
        {4, {4,128,16,128},{4,128,16,128}, 2, 1, "cat_heads_4x128x16x128"},

        /* 5) Large dim=0 vertical stack */
        {2, {4096,4096},   {4096,4096}, 0, 1, "cat_vstack_4Kx4K"},
    };
    int ncfgs = sizeof(cfgs) / sizeof(cfgs[0]);

    /* ── Header ── */
    printf("%-22s  %-24s  %-24s  %7s  %7s  %7s  %7s  %7s  %7s\n",
           "op", "shape_a", "shape_b",
           "fwd_us", "bwd_us", "step_us",
           "fwd_GBs", "bwd_GBs", "step_GBs");
    printf("%-22s  %-24s  %-24s  %7s  %7s  %7s  %7s  %7s  %7s\n",
           "--", "------", "------",
           "------", "------", "------",
           "-------", "-------", "-------");

    for (int i = 0; i < ncfgs; i++) {
        cat_cfg *cfg = &cfgs[i];

        /* Compute total data touched */
        int numel_a = 1; for (int d = 0; d < cfg->ndim; d++) numel_a *= cfg->shape_a[d];
        int numel_b = 1; for (int d = 0; d < cfg->ndim; d++) numel_b *= cfg->shape_b[d];
        int numel_out = numel_a + numel_b;

        /* Bytes moved:
         *   fwd: read a + read b + write out = (numel_a + numel_b + numel_out) * sizeof(float)
         *   bwd: read grad_out + write grad_a + write grad_b (accumulate)
         *   step: fwd + bwd
         */
        double bytes_fwd  = (double)(numel_a + numel_b + numel_out) * sizeof(float);
        double bytes_bwd  = (double)(numel_out + numel_a + numel_b) * sizeof(float);
        double bytes_step = bytes_fwd + bytes_bwd;

        mem_pool_reset(ctx.params);
        mem_pool_reset(ctx.scratch);

        double fwd_us  = bench_cat_fwd(cfg, warmup, trials);
        double step_us = bench_cat_step(cfg, warmup, trials);
        double bwd_us  = step_us - fwd_us;
        if (bwd_us < 0) bwd_us = 0;

        double fwd_gbs  = bytes_fwd  / (fwd_us * 1e3);
        double bwd_gbs  = bwd_us > 0 ? bytes_bwd / (bwd_us * 1e3) : 0;
        double step_gbs = bytes_step / (step_us * 1e3);

        /* Format shapes */
        char sa[32], sb[32];
        int pos = 0;
        pos += snprintf(sa + pos, sizeof(sa) - (size_t)pos, "[");
        for (int d = 0; d < cfg->ndim; d++)
            pos += snprintf(sa + pos, sizeof(sa) - (size_t)pos, "%s%d", d?",":"", cfg->shape_a[d]);
        snprintf(sa + pos, sizeof(sa) - (size_t)pos, "]");

        pos = 0;
        pos += snprintf(sb + pos, sizeof(sb) - (size_t)pos, "[");
        for (int d = 0; d < cfg->ndim; d++)
            pos += snprintf(sb + pos, sizeof(sb) - (size_t)pos, "%s%d", d?",":"", cfg->shape_b[d]);
        snprintf(sb + pos, sizeof(sb) - (size_t)pos, "]");

        printf("%-22s  %-24s  %-24s  %7.0f  %7.0f  %7.0f  %7.2f  %7.2f  %7.2f\n",
               cfg->label, sa, sb, fwd_us, bwd_us, step_us,
               fwd_gbs, bwd_gbs, step_gbs);

        mem_pool_reset(ctx.params);
    }

    /* ── Non-contiguous (strided) baseline ──
     *  Transpose inputs to force the coord-decompose fallback path.
     */
    printf("\n## Strided (non-contiguous) forward — coord-decompose fallback\n\n");
    printf("%-22s  %-24s  %-24s  %7s  %7s\n",
           "op", "shape_a", "shape_b", "fwd_us", "fwd_GBs");
    printf("%-22s  %-24s  %-24s  %7s  %7s\n",
           "--", "------", "------", "------", "-------");

    cat_cfg strided_cfgs[] = {
        {3, {1,128,768},   {1,1,768},   1, 0, "strided_kv_128x768"},
        {3, {1,1024,4096}, {1,1,4096},  1, 0, "strided_kv_1Kx4K"},
        {2, {768,3072},    {768,3072},  1, 0, "strided_glu_768x3K"},
        {2, {4096,4096},   {4096,4096}, 1, 0, "strided_concat_4Kx4K"},
    };
    int ns = sizeof(strided_cfgs) / sizeof(strided_cfgs[0]);

    for (int i = 0; i < ns; i++) {
        cat_cfg *cfg = &strided_cfgs[i];
        int numel_a = 1; for (int d = 0; d < cfg->ndim; d++) numel_a *= cfg->shape_a[d];
        int numel_b = 1; for (int d = 0; d < cfg->ndim; d++) numel_b *= cfg->shape_b[d];
        int numel_out = numel_a + numel_b;
        double bytes_fwd = (double)(numel_a + numel_b + numel_out) * sizeof(float);

        mem_pool_reset(ctx.params);
        mem_pool_reset(ctx.scratch);

        double fwd_us = bench_cat_fwd_strided(cfg, warmup, trials);
        double fwd_gbs = bytes_fwd / (fwd_us * 1e3);

        char sa[32], sb[32];
        int pos = 0;
        pos += snprintf(sa + pos, sizeof(sa) - (size_t)pos, "[");
        for (int d = 0; d < cfg->ndim; d++)
            pos += snprintf(sa + pos, sizeof(sa) - (size_t)pos, "%s%d", d?",":"", cfg->shape_a[d]);
        snprintf(sa + pos, sizeof(sa) - (size_t)pos, "]");

        pos = 0;
        pos += snprintf(sb + pos, sizeof(sb) - (size_t)pos, "[");
        for (int d = 0; d < cfg->ndim; d++)
            pos += snprintf(sb + pos, sizeof(sb) - (size_t)pos, "%s%d", d?",":"", cfg->shape_b[d]);
        snprintf(sb + pos, sizeof(sb) - (size_t)pos, "]");

        printf("%-22s  %-24s  %-24s  %7.0f  %7.2f\n",
               cfg->label, sa, sb, fwd_us, fwd_gbs);
    }

    dnn_ctx_destroy(&ctx);
    return 0;
}
