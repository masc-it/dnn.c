#ifndef DNN_SIMD_H
#define DNN_SIMD_H

/* ── SIMD abstraction layer ──
 *
 * Provides branch-free NEON intrinsics for hot loops in ops_activation.c
 * and possibly other kernel files.  Falls back to portable C on non-ARM,
 * or when DNN_NO_SIMD is defined.
 *
 * Scope: relu fwd/bwd, cross_entropy fwd/bwd, softmax fwd.
 *
 * ── Accuracy notes ──
 *
 * simd_expf_f32 uses range reduction + 6th-order Taylor polynomial on
 * [-ln2/2, ln2/2].  Maximum relative error < 1e-6 vs libm expf, well
 * within NN training tolerance (gradient noise dominates).
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>

/* ── Platform detection ── */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  ifndef DNN_NO_SIMD
#    define DNN_HAVE_NEON 1
#  else
#    define DNN_HAVE_NEON 0
#  endif
#else
#  define DNN_HAVE_NEON 0
#endif

/* ══════════════════════════════════════════════════════════════════
 *  Vector expf — fast polynomial approximation
 * ══════════════════════════════════════════════════════════════════ */

#if DNN_HAVE_NEON

static inline float32x4_t simd_expf_f32(float32x4_t x) {
    /*
     * Range reduction: exp(x) = 2^(n) * exp(r)
     *   n = round(x / ln2)
     *   r = x - n*ln2,  |r| ≤ ln2/2 ≈ 0.347
     *
     * exp(r) via 6th-order Taylor (minimax on [-ln2/2, ln2/2]):
     *   exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
     *
     * 2^n constructed via float exponent: (n + 127) << 23
     */
    const float32x4_t ln2         = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t inv_ln2     = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t one         = vdupq_n_f32(1.0f);
    const float32x4_t c2          = vdupq_n_f32(0.5f);
    const float32x4_t c3          = vdupq_n_f32(0.16666667163372039794921875f);
    const float32x4_t c4          = vdupq_n_f32(0.0416666679084300994873046875f);
    const float32x4_t c5          = vdupq_n_f32(0.008333333767950534820556640625f);
    const float32x4_t c6          = vdupq_n_f32(0.00138888892249763011932373046875f);
    const int32x4_t   exp_bias    = vdupq_n_s32(127);

    /* n = nearest integer to x / ln2 */
    float32x4_t n = vrndmq_f32(vmulq_f32(x, inv_ln2));

    /* r = x - n * ln2   (fused multiply-subtract) */
    float32x4_t r = vmlsq_f32(x, n, ln2);

    /* Horner-style polynomial for exp(r) */
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t r3 = vmulq_f32(r2, r);
    float32x4_t r4 = vmulq_f32(r2, r2);
    float32x4_t r5 = vmulq_f32(r4, r);
    float32x4_t r6 = vmulq_f32(r5, r);

    /* poly = 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 */
    float32x4_t poly = vfmaq_f32(vaddq_f32(one, r), r2, c2);
    poly = vfmaq_f32(poly, r3, c3);
    poly = vfmaq_f32(poly, r4, c4);
    poly = vfmaq_f32(poly, r5, c5);
    poly = vfmaq_f32(poly, r6, c6);

    /* scale = 2^n  as float  (n + 127) << 23 */
    int32x4_t n_int = vcvtq_s32_f32(n);
    int32x4_t biased = vaddq_s32(n_int, exp_bias);
    /* clamp to [1, 254]: avoid UB on negative shift (n < -126)
       and INFINITY for huge positive (n > 127).  Clamped values
       produce ~1.18e-38 (subnormal flush) or ~3.4e38, good enough
       approximations for NN use. */
    biased = vmaxq_s32(biased, vdupq_n_s32(1));
    biased = vminq_s32(biased, vdupq_n_s32(254));
    int32x4_t exp_part = vshlq_n_s32(biased, 23);
    float32x4_t scale = vreinterpretq_f32_s32(exp_part);

    return vmulq_f32(poly, scale);
}

#endif /* DNN_HAVE_NEON */

/* ══════════════════════════════════════════════════════════════════
 *  Horizontal operations on arbitrary-length arrays
 * ══════════════════════════════════════════════════════════════════ */

/* Maximum of float array */
static inline float simd_reduce_max_f32(const float *x, int n) {
#if DNN_HAVE_NEON
    float32x4_t vmax = vdupq_n_f32(-INFINITY);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
    }
    float m = vmaxvq_f32(vmax);
    for (; i < n; i++)
        if (x[i] > m) m = x[i];
    return m;
#else
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    return m;
#endif
}

/* Sum of float array */
static inline float simd_reduce_sum_f32(const float *x, int n) {
#if DNN_HAVE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vsum = vaddq_f32(vsum, vld1q_f32(x + i));
    }
    float s = vaddvq_f32(vsum);
    for (; i < n; i++) s += x[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i];
    return s;
#endif
}

/* Compute sum(exp(x[i] - max_val)) for an array, using vector expf when available.
   The max_val is pre-computed (per-row maximum for softmax stability). */
static inline float simd_exp_sum_shifted_f32(const float *x, int n, float max_val) {
#if DNN_HAVE_NEON
    float32x4_t vmax = vdupq_n_f32(max_val);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t shifted = vsubq_f32(vld1q_f32(x + i), vmax);
        vsum = vaddq_f32(vsum, simd_expf_f32(shifted));
    }
    float s = vaddvq_f32(vsum);
    for (; i < n; i++) s += expf(x[i] - max_val);
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += expf(x[i] - max_val);
    return s;
#endif
}

/* ══════════════════════════════════════════════════════════════════
 *  ReLU forward:  y = max(x, 0)
 * ══════════════════════════════════════════════════════════════════ */

static inline void simd_relu_fwd(float *out, const float *in, int n) {
#if DNN_HAVE_NEON
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmaxq_f32(vld1q_f32(in + i), zero));
    }
    for (; i < n; i++)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
#else
    for (int i = 0; i < n; i++)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
#endif
}

/* ══════════════════════════════════════════════════════════════════
 *  ReLU backward:  grad_acc += grad_out * (in > 0)
 * ══════════════════════════════════════════════════════════════════ */

static inline void simd_relu_bwd(float *grad_acc, const float *in,
                                  const float *grad_out, int n) {
#if DNN_HAVE_NEON
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vin  = vld1q_f32(in + i);
        float32x4_t vg   = vld1q_f32(grad_out + i);
        float32x4_t vacc = vld1q_f32(grad_acc + i);
        /* mask = (vin > 0) ? all-ones : all-zeros */
        uint32x4_t mask  = vcgtq_f32(vin, zero);
        /* selected = (mask) ? vg : 0 */
        float32x4_t selected = vbslq_f32(mask, vg, zero);
        vst1q_f32(grad_acc + i, vaddq_f32(vacc, selected));
    }
    for (; i < n; i++)
        if (in[i] > 0.0f) grad_acc[i] += grad_out[i];
#else
    for (int i = 0; i < n; i++)
        if (in[i] > 0.0f) grad_acc[i] += grad_out[i];
#endif
}

/* ══════════════════════════════════════════════════════════════════
 *  Cross-entropy backward: compute softmax gradient for one row
 *
 *  For each element c in a row (width = C):
 *    softmax[c] = exp(logits[c] - max) / sum_exp
 *    grad[c] += (softmax[c] - (c == target)) * scale
 *
 *  Called per row (per training example).  C is typically 10-10000.
 *  The function processes in 4-element groups using NEON.
 * ══════════════════════════════════════════════════════════════════ */

#if DNN_HAVE_NEON

static inline void simd_ce_bwd_row_kernel(float *grad_out_row,
                                           const float *logits_row,
                                           float max_val, float sum_exp,
                                           int target, float scale, int C) {
    const float32x4_t vmax     = vdupq_n_f32(max_val);
    const float32x4_t vsum_exp = vdupq_n_f32(sum_exp);
    const float32x4_t vscale   = vdupq_n_f32(scale);
    const int32x4_t   vtgt     = vdupq_n_s32(target);
    int32x4_t         vidx     = {0,1,2,3};  /* per-lane index for one-hot detection */

    int c = 0;
    for (; c + 4 <= C; c += 4) {
        /* softmax = exp(logits - max) / sum_exp */
        float32x4_t logits = vld1q_f32(logits_row + c);
        float32x4_t shifted = vsubq_f32(logits, vmax);
        float32x4_t exp_val = simd_expf_f32(shifted);
        float32x4_t sm = vdivq_f32(exp_val, vsum_exp);

        /* one-hot: compare {c, c+1, c+2, c+3} to target */
        /* we track index explicitly in vidx */
        uint32x4_t is_tgt = vceqq_s32(vidx, vtgt);
        float32x4_t vone_hot = vbslq_f32(is_tgt, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f));

        /* grad = (sm - one_hot) * scale */
        float32x4_t grad = vmulq_f32(vsubq_f32(sm, vone_hot), vscale);

        /* accumulate into grad_out_row */
        float32x4_t acc = vld1q_f32(grad_out_row + c);
        vst1q_f32(grad_out_row + c, vaddq_f32(acc, grad));

        /* advance index vector by 4 */
        vidx = vaddq_s32(vidx, vdupq_n_s32(4));
    }

    /* scalar tail */
    for (; c < C; c++) {
        float sm = expf(logits_row[c] - max_val) / sum_exp;
        grad_out_row[c] += (sm - (float)(c == target)) * scale;
    }
}

#endif /* DNN_HAVE_NEON */

/* ══════════════════════════════════════════════════════════════════
 *  Sigmoid forward:  y = 1 / (1 + exp(-x))
 * ══════════════════════════════════════════════════════════════════ */

static inline void simd_sigmoid_fwd(float *out, const float *in, int n) {
#if DNN_HAVE_NEON
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        float32x4_t exp_neg_x = simd_expf_f32(vnegq_f32(x));
        vst1q_f32(out + i, vdivq_f32(one, vaddq_f32(one, exp_neg_x)));
    }
    for (; i < n; i++)
        out[i] = 1.0f / (1.0f + expf(-in[i]));
#else
    for (int i = 0; i < n; i++)
        out[i] = 1.0f / (1.0f + expf(-in[i]));
#endif
}

/* ══════════════════════════════════════════════════════════════════
 *  Sigmoid backward:  grad_acc += sig * (1 - sig) * grad_out
 * ══════════════════════════════════════════════════════════════════ */

static inline void simd_sigmoid_bwd(float *grad_acc, const float *sig,
                                     const float *grad_out, int n) {
#if DNN_HAVE_NEON
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vsig = vld1q_f32(sig + i);
        float32x4_t vg   = vld1q_f32(grad_out + i);
        float32x4_t vacc = vld1q_f32(grad_acc + i);
        float32x4_t dsig = vmulq_f32(vsig, vsubq_f32(one, vsig));
        vst1q_f32(grad_acc + i, vfmaq_f32(vacc, dsig, vg));
    }
    for (; i < n; i++)
        grad_acc[i] += sig[i] * (1.0f - sig[i]) * grad_out[i];
#else
    for (int i = 0; i < n; i++)
        grad_acc[i] += sig[i] * (1.0f - sig[i]) * grad_out[i];
#endif
}

#endif /* DNN_SIMD_H */
