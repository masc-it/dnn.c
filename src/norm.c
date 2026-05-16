#include "norm.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
#include "broadcast.h"
#include "simd.h"
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── ln_backward ── */

static void ln_backward(grad_fn *fn, tensor *grad_output) {
    tensor *x      = fn->inputs[0];
    tensor *weight = (tensor*)fn->saved_tensors[0];
    tensor *bias   = (tensor*)fn->saved_tensors[1];
    float  *mean   = (float*)fn->saved_tensors[2];
    float  *rstd   = (float*)fn->saved_tensors[3];

    int ndim = x->ndim;
    int d    = x->shape[ndim - 1];
    int n    = tensor_numel(x) / d;

    float *xd = tensor_data_ptr(x);
    float *gd = (float*)grad_output->data;  /* always contiguous */
    float *wd = weight ? tensor_data_ptr(weight) : NULL;

    float inv_d = 1.0f / (float)d;

    /* dγ = sum(d_out * y, over batch dims) — y = (x - μ) * rstd */
    if (weight && tensor_requires_grad(weight)) {
        float *wg = _grad_ensure(weight);
#pragma omp parallel for
        for (int s = 0; s < n; s++) {
            float m = mean[s], rs = rstd[s];
            float *x_slice = xd + s * d;
            float *g_slice = gd + s * d;
#if DNN_HAVE_NEON
            float32x4_t vm  = vdupq_n_f32(m);
            float32x4_t vrs = vdupq_n_f32(rs);
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t vx = vld1q_f32(x_slice + j);
                float32x4_t vg = vld1q_f32(g_slice + j);
                float32x4_t vy = vmulq_f32(vsubq_f32(vx, vm), vrs);
                float32x4_t vw = vld1q_f32(wg + j);
                vst1q_f32(wg + j, vfmaq_f32(vw, vg, vy));
            }
            for (; j < d; j++) {
                float y = (x_slice[j] - m) * rs;
                wg[j] += g_slice[j] * y;
            }
#else
            #pragma omp simd
            for (int j = 0; j < d; j++) {
                float y = (x_slice[j] - m) * rs;
                wg[j] += g_slice[j] * y;
            }
#endif
        }
    }

    /* dβ = sum(d_out, over batch dims) */
    if (bias && tensor_requires_grad(bias)) {
        float *bg = _grad_ensure(bias);
#pragma omp parallel for
        for (int s = 0; s < n; s++) {
            float *g_slice = gd + s * d;
#if DNN_HAVE_NEON
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t vg = vld1q_f32(g_slice + j);
                float32x4_t vb = vld1q_f32(bg + j);
                vst1q_f32(bg + j, vaddq_f32(vb, vg));
            }
            for (; j < d; j++) bg[j] += g_slice[j];
#else
            #pragma omp simd
            for (int j = 0; j < d; j++)
                bg[j] += g_slice[j];
#endif
        }
    }

    /* dx = rstd * (dy - mean(dy) - xmu * mean(dy * xmu) * rstd²)
     *
     * Precompute dy = gd * weight (or 1.0f if no weight) into a local VLA
     * so the second loop avoids re-reading gd + re-multiplying.  Saves ~1
     * load + 1 multiply + 1 branch per element in the second pass.
     *
     * The VLA lives on each thread's stack (OpenMP parallel region → private).
     * d is typically hidden-dim size (128-4096 in most models, ~12KB for 3136). */
    if (tensor_requires_grad(x)) {
        float *xg = _grad_ensure(x);
#pragma omp parallel for
        for (int s = 0; s < n; s++) {
            float m = mean[s], rs = rstd[s];
            float *x_slice = xd + s * d;
            float *g_slice = gd + s * d;
            float *xg_slice = xg + s * d;

            float dy_buf[d];  /* VLA, per-thread stack — stores dy = gd * weight */
            float sum_dy = 0.0f, sum_dy_xmu = 0.0f;

#if DNN_HAVE_NEON
            float32x4_t vm  = vdupq_n_f32(m);
            float32x4_t vone = vdupq_n_f32(1.0f);
            float32x4_t vsum_dy = vdupq_n_f32(0.0f);
            float32x4_t vsum_dy_xmu = vdupq_n_f32(0.0f);
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t vx  = vld1q_f32(x_slice + j);
                float32x4_t vg  = vld1q_f32(g_slice + j);
                float32x4_t vw  = wd ? vld1q_f32(wd + j) : vone;
                float32x4_t vxmu = vsubq_f32(vx, vm);
                float32x4_t vdy = vmulq_f32(vg, vw);
                vst1q_f32(dy_buf + j, vdy);
                vsum_dy     = vaddq_f32(vsum_dy, vdy);
                vsum_dy_xmu = vfmaq_f32(vsum_dy_xmu, vdy, vxmu);
            }
            sum_dy     = vaddvq_f32(vsum_dy);
            sum_dy_xmu = vaddvq_f32(vsum_dy_xmu);
            for (; j < d; j++) {
                float w = wd ? wd[j] : 1.0f;
                dy_buf[j] = g_slice[j] * w;
                float xmu = x_slice[j] - m;
                sum_dy     += dy_buf[j];
                sum_dy_xmu += dy_buf[j] * xmu;
            }
#else
            #pragma omp simd reduction(+:sum_dy,sum_dy_xmu)
            for (int j = 0; j < d; j++) {
                float w = wd ? wd[j] : 1.0f;
                dy_buf[j] = g_slice[j] * w;
                float xmu = x_slice[j] - m;
                sum_dy     += dy_buf[j];
                sum_dy_xmu += dy_buf[j] * xmu;
            }
#endif

            float mean_dy     = sum_dy * inv_d;
            float mean_dy_xmu = sum_dy_xmu * inv_d;

#if DNN_HAVE_NEON
            float32x4_t vmean_dy     = vdupq_n_f32(mean_dy);
            float32x4_t vmean_dy_xmu = vdupq_n_f32(mean_dy_xmu);
            float32x4_t vrs          = vdupq_n_f32(rs);
            float32x4_t vrs2         = vdupq_n_f32(rs * rs);
            j = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t vxmu  = vsubq_f32(vld1q_f32(x_slice + j), vm);
                float32x4_t vdy   = vld1q_f32(dy_buf + j);
                float32x4_t vx    = vld1q_f32(xg_slice + j);
                /* dx = rs * (dy - mean_dy - xmu * mean_dy_xmu * rs²) */
                float32x4_t vdx   = vsubq_f32(vsubq_f32(vdy, vmean_dy),
                                              vmulq_f32(vxmu, vmulq_f32(vmean_dy_xmu, vrs2)));
                vst1q_f32(xg_slice + j, vfmaq_f32(vx, vrs, vdx));
            }
            for (; j < d; j++) {
                float xmu = x_slice[j] - m;
                float dx  = rs * (dy_buf[j] - mean_dy - xmu * mean_dy_xmu * rs * rs);
                xg_slice[j] += dx;
            }
#else
            #pragma omp simd
            for (int j = 0; j < d; j++) {
                float xmu = x_slice[j] - m;
                float dx  = rs * (dy_buf[j] - mean_dy - xmu * mean_dy_xmu * rs * rs);
                xg_slice[j] += dx;
            }
#endif
        }
    }
}

/* ── layer_norm forward ── */

tensor *tensor_layer_norm(struct mem_pool *scratch, const tensor *x, const tensor *weight,
                          const tensor *bias, float eps) {
    assert(x);
    assert(tensor_is_contiguous(x) && "tensor_layer_norm: x must be contiguous");

    int ndim = x->ndim;
    int d    = x->shape[ndim - 1];
    int n    = tensor_numel(x) / d;  /* slices over all dims except last */

    float *xd = tensor_data_ptr((tensor*)x);

    /* mean and rstd per slice (allocated in scratch, valid until backward runs) */
    float *mean = _mem_pool_alloc(scratch, n * sizeof(float), NULL);
    float *rstd = _mem_pool_alloc(scratch, n * sizeof(float), NULL);

    /* pass 1 (fused): Welford online mean + M2 → rstd
     *
     * Single pass computes both mean and variance using Welford's
     * numerically stable online algorithm:
     *   delta = x - mean
     *   mean += delta / count
     *   delta2 = x - mean
     *   M2 += delta * delta2
     *   var = M2 / count
     *
     * Saves 1 full d-element read per slice (33% of norm fwd time).
     */
#pragma omp parallel for
    for (int s = 0; s < n; s++) {
        float m = 0.0f, M2 = 0.0f;
        for (int j = 0; j < d; j++) {
            float x = xd[s * d + j];
            float delta = x - m;
            m += delta / (float)(j + 1);
            float delta2 = x - m;
            M2 += delta * delta2;
        }
        mean[s] = m;
        rstd[s] = 1.0f / sqrtf(M2 / (float)d + eps);
    }

    /* pass 3: compute output = γ * ((x - μ) * rstd) + β */
    tensor *out = tensor_scratch(scratch, ndim, x->shape, 0);
    float *od   = (float*)out->data;
    float *wd   = weight ? tensor_data_ptr((tensor*)weight) : NULL;
    float *bd   = bias   ? tensor_data_ptr((tensor*)bias)   : NULL;

#pragma omp parallel for
    for (int s = 0; s < n; s++) {
        float m = mean[s], rs = rstd[s];
        float *x_slice = xd + s * d;
        float *o_slice = od + s * d;
#if DNN_HAVE_NEON
        float32x4_t vm  = vdupq_n_f32(m);
        float32x4_t vrs = vdupq_n_f32(rs);
        int j = 0;
        for (; j + 4 <= d; j += 4) {
            float32x4_t vx = vld1q_f32(x_slice + j);
            float32x4_t vy = vmulq_f32(vsubq_f32(vx, vm), vrs);
            float32x4_t vw = wd ? vld1q_f32(wd + j) : vdupq_n_f32(1.0f);
            float32x4_t vb = bd ? vld1q_f32(bd + j) : vdupq_n_f32(0.0f);
            vst1q_f32(o_slice + j, vfmaq_f32(vb, vy, vw));
        }
        for (; j < d; j++) {
            float y = (x_slice[j] - m) * rs;
            o_slice[j] = y * (wd ? wd[j] : 1.0f) + (bd ? bd[j] : 0.0f);
        }
#else
        #pragma omp simd
        for (int j = 0; j < d; j++) {
            float y = (x_slice[j] - m) * rs;
            o_slice[j] = y * (wd ? wd[j] : 1.0f) + (bd ? bd[j] : 0.0f);
        }
#endif
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(x) ||
         (weight && tensor_requires_grad(weight)) ||
         (bias   && tensor_requires_grad(bias)))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = ln_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)x;

        fn->n_saved = 4;
        fn->saved_tensors = _mem_pool_alloc(scratch, 4 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)weight;  /* may be NULL */
        fn->saved_tensors[1] = (tensor*)bias;    /* may be NULL */
        fn->saved_tensors[2] = (tensor*)mean;
        fn->saved_tensors[3] = (tensor*)rstd;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── rms_backward ── */

static void rms_backward(grad_fn *fn, tensor *grad_output) {
    tensor *x      = fn->inputs[0];
    tensor *weight = (tensor*)fn->saved_tensors[0];
    float  *rstd   = (float*)fn->saved_tensors[1];

    int ndim = x->ndim;
    int d    = x->shape[ndim - 1];
    int n    = tensor_numel(x) / d;

    float *xd = tensor_data_ptr(x);
    float *gd = (float*)grad_output->data;
    float *wd = weight ? tensor_data_ptr(weight) : NULL;

    float inv_d = 1.0f / (float)d;

    /* dγ = sum(dy * x * rs, over batch dims) */
    if (weight && tensor_requires_grad(weight)) {
        float *wg = _grad_ensure(weight);
#pragma omp parallel for
        for (int s = 0; s < n; s++) {
            float rs = rstd[s];
            float *x_slice = xd + s * d;
            float *g_slice = gd + s * d;
#if DNN_HAVE_NEON
            float32x4_t vrs = vdupq_n_f32(rs);
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t vx = vld1q_f32(x_slice + j);
                float32x4_t vg = vld1q_f32(g_slice + j);
                float32x4_t vw = vld1q_f32(wg + j);
                vst1q_f32(wg + j, vfmaq_f32(vw, vg, vmulq_f32(vx, vrs)));
            }
            for (; j < d; j++) {
                wg[j] += g_slice[j] * x_slice[j] * rs;
            }
#else
            #pragma omp simd
            for (int j = 0; j < d; j++) {
                wg[j] += g_slice[j] * x_slice[j] * rs;
            }
#endif
        }
    }

    /* dx = rs * (γ * dy - rs²/d * x * S)   where S = sum(γ * dy * x, over last dim)
     *
     * Two-pass per slice:
     *   pass 1: compute S = sum(γ_i * gd_i * x_i)
     *   pass 2: compute dx_i = rs * (γ_i * gd_i - rs²/d * x_i * S)
     *
     * γ = weight (or 1.0f when NULL), dy = gd * γ saved in dy_buf VLA. */
    if (tensor_requires_grad(x)) {
        float *xg = _grad_ensure(x);
#pragma omp parallel for
        for (int s = 0; s < n; s++) {
            float rs  = rstd[s];
            float rs2 = rs * rs;
            float *x_slice = xd + s * d;
            float *g_slice = gd + s * d;
            float *xg_slice = xg + s * d;

            float dy_buf[d];  /* VLA — dy = gd * γ */
            float S = 0.0f;

#if DNN_HAVE_NEON
            float32x4_t vone = vdupq_n_f32(1.0f);
            float32x4_t vsum = vdupq_n_f32(0.0f);
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t vx  = vld1q_f32(x_slice + j);
                float32x4_t vg  = vld1q_f32(g_slice + j);
                float32x4_t vw  = wd ? vld1q_f32(wd + j) : vone;
                float32x4_t vdy = vmulq_f32(vg, vw);
                vst1q_f32(dy_buf + j, vdy);
                vsum = vfmaq_f32(vsum, vdy, vx);
            }
            S = vaddvq_f32(vsum);
            for (; j < d; j++) {
                float w = wd ? wd[j] : 1.0f;
                dy_buf[j] = g_slice[j] * w;
                S += dy_buf[j] * x_slice[j];
            }
#else
            #pragma omp simd reduction(+:S)
            for (int j = 0; j < d; j++) {
                float w = wd ? wd[j] : 1.0f;
                dy_buf[j] = g_slice[j] * w;
                S += dy_buf[j] * x_slice[j];
            }
#endif

            float S_scaled = rs2 * inv_d * S;

#if DNN_HAVE_NEON
            float32x4_t vrs   = vdupq_n_f32(rs);
            float32x4_t vrs_n = vdupq_n_f32(-rs);
            float32x4_t vss   = vdupq_n_f32(S_scaled);
            j = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t vdy  = vld1q_f32(dy_buf + j);
                float32x4_t vx   = vld1q_f32(x_slice + j);
                float32x4_t vacc = vld1q_f32(xg_slice + j);
                /* dx = rs * dy - rs * S_scaled * x */
                float32x4_t vdx  = vfmaq_f32(vmulq_f32(vrs, vdy), vrs_n, vmulq_f32(vx, vss));
                vst1q_f32(xg_slice + j, vaddq_f32(vacc, vdx));
            }
            for (; j < d; j++) {
                float dx  = rs * dy_buf[j] - rs * S_scaled * x_slice[j];
                xg_slice[j] += dx;
            }
#else
            #pragma omp simd
            for (int j = 0; j < d; j++) {
                float dx = rs * dy_buf[j] - rs * S_scaled * x_slice[j];
                xg_slice[j] += dx;
            }
#endif
        }
    }
}

/* ── RMS Norm forward ── */

tensor *tensor_rms_norm(struct mem_pool *scratch, const tensor *x,
                         const tensor *weight, float eps) {
    assert(x);
    assert(tensor_is_contiguous(x) && "tensor_rms_norm: x must be contiguous");

    int ndim = x->ndim;
    int d    = x->shape[ndim - 1];
    int n    = tensor_numel(x) / d;

    float *xd = tensor_data_ptr((tensor*)x);

    /* rstd = rsqrt(mean(x²) + eps) per slice */
    float *rstd = _mem_pool_alloc(scratch, n * sizeof(float), NULL);

#pragma omp parallel for
    for (int s = 0; s < n; s++) {
        float sum_x2 = 0.0f;
        float *x_slice = xd + s * d;
#if DNN_HAVE_NEON
        float32x4_t vsum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 4 <= d; j += 4) {
            float32x4_t vx = vld1q_f32(x_slice + j);
            vsum = vfmaq_f32(vsum, vx, vx);
        }
        sum_x2 = vaddvq_f32(vsum);
        for (; j < d; j++) {
            sum_x2 += x_slice[j] * x_slice[j];
        }
#else
        #pragma omp simd reduction(+:sum_x2)
        for (int j = 0; j < d; j++) {
            sum_x2 += x_slice[j] * x_slice[j];
        }
#endif
        float mean_x2 = sum_x2 * (1.0f / (float)d);
        rstd[s] = 1.0f / sqrtf(mean_x2 + eps);
    }

    /* output = x * rs * γ */
    tensor *out = tensor_scratch(scratch, ndim, x->shape, 0);
    float *od   = (float*)out->data;
    float *wd   = weight ? tensor_data_ptr((tensor*)weight) : NULL;

#pragma omp parallel for
    for (int s = 0; s < n; s++) {
        float rs = rstd[s];
        float *x_slice = xd + s * d;
        float *o_slice = od + s * d;
#if DNN_HAVE_NEON
        float32x4_t vrs = vdupq_n_f32(rs);
        int j = 0;
        for (; j + 4 <= d; j += 4) {
            float32x4_t vx  = vld1q_f32(x_slice + j);
            float32x4_t vy  = vmulq_f32(vx, vrs);
            float32x4_t vw  = wd ? vld1q_f32(wd + j) : vdupq_n_f32(1.0f);
            vst1q_f32(o_slice + j, vmulq_f32(vy, vw));
        }
        for (; j < d; j++) {
            float y = x_slice[j] * rs;
            o_slice[j] = y * (wd ? wd[j] : 1.0f);
        }
#else
        #pragma omp simd
        for (int j = 0; j < d; j++) {
            float y = x_slice[j] * rs;
            o_slice[j] = y * (wd ? wd[j] : 1.0f);
        }
#endif
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(x) ||
         (weight && tensor_requires_grad(weight)))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = rms_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)x;

        fn->n_saved = 2;
        fn->saved_tensors = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)weight;  /* may be NULL */
        fn->saved_tensors[1] = (tensor*)rstd;    /* rs per slice */

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
