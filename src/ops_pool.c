#include "ops.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
#include "broadcast.h"
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── NEON detection (reuse simd.h pattern) ── */
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
 *  Average Pooling 2D
 *
 *  Forward:  y[n][c][oh][ow] = mean( x[n][c][ih:ih+k][iw:iw+k] )
 *  Backward: dx[n][c][ih+di][iw+dj] += dy[n][c][oh][ow] / (k*k)
 *
 *  NEON fast paths for k=2 (most common for downsampling).
 * ══════════════════════════════════════════════════════════════════ */

/* ── Forward helper for k=2, stride=2 (non-overlapping 2×2 blocks) ── */
static inline void avg_pool2d_fwd_k2s2_neon(const float *row0, const float *row1,
                                             float *out_row, int W_out, int W,
                                             float inv_k2) {
#if DNN_HAVE_NEON
    float32x4_t v_inv = vdupq_n_f32(inv_k2);
    int ow = 0;

    /* Process 2 output columns per iteration: load 4 input floats from each row */
    for (; ow + 2 <= W_out; ow += 2) {
        int iw0 = ow * 2;  /* stride=2 */
        float32x4_t r0 = vld1q_f32(row0 + iw0);  /* [a,b,c,d] */
        float32x4_t r1 = vld1q_f32(row1 + iw0);  /* [e,f,g,h] */
        float32x4_t s  = vaddq_f32(r0, r1);      /* [a+e,b+f,c+g,d+h] */
        /* pair-add within low and high halves */
        float32x2_t lo = vpadd_f32(vget_low_f32(s), vget_high_f32(s));
        /* lo = [a+e+b+f, c+g+d+h] */
        float32x2_t scaled = vmul_f32(lo, vget_low_f32(v_inv));
        vst1_f32(out_row + ow, scaled);
    }

    /* scalar tail */
    for (; ow < W_out; ow++) {
        int iw0 = ow * 2;
        float sum = row0[iw0] + row0[iw0+1] + row1[iw0] + row1[iw0+1];
        out_row[ow] = sum * inv_k2;
    }
#else
    (void)row0; (void)row1; (void)out_row; (void)W_out; (void)W; (void)inv_k2;
#endif
}

/* ── Forward helper for k=2, stride=1 (overlapping 2×2 windows) ── */
static inline void avg_pool2d_fwd_k2s1_neon(const float *row0, const float *row1,
                                             float *out_row, int W_out, int W,
                                             float inv_k2) {
#if DNN_HAVE_NEON
    float32x4_t v_inv = vdupq_n_f32(inv_k2);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    int ow = 0;

    for (; ow + 4 <= W_out; ow += 4) {
        /* Load 5 floats from each row to cover 4 overlapping windows.
         *   win[0] = r0[0]+r0[1]+r1[0]+r1[1]  = s[0]+s[1]
         *   win[1] = r0[1]+r0[2]+r1[1]+r1[2]  = s[1]+s[2]
         *   win[2] = r0[2]+r0[3]+r1[2]+r1[3]  = s[2]+s[3]
         *   win[3] = r0[3]+r0[4]+r1[3]+r1[4]  = s[3]+s[4]
         * where s[i] = r0[i]+r1[i]
         */
        float32x4_t r0_0 = vld1q_f32(row0 + ow);      /* [a,b,c,d] */
        float32x4_t r0_1 = vld1q_f32(row0 + ow + 1);  /* [b,c,d,e] */
        float32x4_t r1_0 = vld1q_f32(row1 + ow);
        float32x4_t r1_1 = vld1q_f32(row1 + ow + 1);

        /* Sum adjacent pairs vertically then horizontally */
        float32x4_t s0 = vaddq_f32(r0_0, r1_0);  /* [a+e, b+f, c+g, d+h] */
        float32x4_t s1 = vaddq_f32(r0_1, r1_1);  /* [b+f, c+g, d+h, e+i] */

        /* win[i] = s0[i] + s1[i] = (r0[i]+r1[i]) + (r0[i+1]+r1[i+1]) */
        float32x4_t win = vaddq_f32(s0, s1);
        vst1q_f32(out_row + ow, vmulq_f32(win, v_inv));
    }

    /* scalar tail */
    for (; ow < W_out; ow++) {
        float sum = row0[ow] + row0[ow+1] + row1[ow] + row1[ow+1];
        out_row[ow] = sum * inv_k2;
    }
#else
    (void)row0; (void)row1; (void)out_row; (void)W_out; (void)W; (void)inv_k2;
#endif
}

/* ── Forward helper for general k (scalar, small k) ── */
static inline float avg_pool2d_fwd_general(const float *x_chan, int ih0, int iw0,
                                            int k, int stride, int W) {
    float sum = 0.0f;
    for (int di = 0; di < k; di++) {
        const float *row = x_chan + (ih0 + di) * W;
        for (int dj = 0; dj < k; dj++)
            sum += row[iw0 + dj];
    }
    return sum;
}

/* ── avg_pool2d_backward ── */

static void avg_pool2d_backward(grad_fn *fn, tensor *grad_output) {
    tensor *x = fn->inputs[0];
    int k      = *(int*)fn->saved_tensors[0];
    int stride = *(int*)fn->saved_tensors[1];

    int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int H_out = (H - k) / stride + 1;
    int W_out = (W - k) / stride + 1;

    float *gd = (float*)grad_output->data;
    float inv_k2 = 1.0f / (float)(k * k);

    if (!tensor_requires_grad(x)) return;
    float *xg = _grad_ensure(x);

    /* Backward fast paths for k=2 */
    if (k == 2 && stride == 2 && tensor_is_contiguous(x) && tensor_is_contiguous(grad_output)) {
#if DNN_HAVE_NEON
        float32x4_t v_inv = vdupq_n_f32(inv_k2);
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float *x_chan = xg + ((size_t)n * C + c) * (size_t)H * W;
                float *g_chan = gd + ((size_t)n * C + c) * (size_t)H_out * W_out;

                for (int oh = 0; oh < H_out; oh++) {
                    int ih0 = oh * 2;
                    float *row0 = x_chan + ih0 * W;
                    float *row1 = x_chan + (ih0 + 1) * W;
                    float *g_row = g_chan + oh * W_out;
                    int ow = 0;

                    for (; ow + 2 <= W_out; ow += 2) {
                        /* Load 2 gradient values, broadcast to all 4 writes */
                        float32x2_t g2 = vld1_f32(g_row + ow);         /* [g0, g1] */
                        float32x4_t g4 = vmulq_f32(vcombine_f32(g2, g2), v_inv);
                        /* g4 = [g0/4, g1/4, g0/4, g1/4] */
                        int iw0 = ow * 2;
                        /* row0[iw0+0..1] += g0/4, g1/4 */
                        float32x2_t cur0 = vld1_f32(row0 + iw0);
                        vst1_f32(row0 + iw0, vadd_f32(cur0, vget_low_f32(g4)));
                        /* row0[iw0+2..3] += g0/4, g1/4 (adjacent window) */
                        float32x2_t cur1 = vld1_f32(row0 + iw0 + 2);
                        vst1_f32(row0 + iw0 + 2, vadd_f32(cur1, vget_low_f32(g4)));
                        /* row1 */
                        cur0 = vld1_f32(row1 + iw0);
                        vst1_f32(row1 + iw0, vadd_f32(cur0, vget_low_f32(g4)));
                        cur1 = vld1_f32(row1 + iw0 + 2);
                        vst1_f32(row1 + iw0 + 2, vadd_f32(cur1, vget_low_f32(g4)));
                    }

                    /* scalar tail */
                    for (; ow < W_out; ow++) {
                        int iw0 = ow * 2;
                        float g = g_row[ow] * inv_k2;
                        row0[iw0]   += g; row0[iw0+1] += g;
                        row1[iw0]   += g; row1[iw0+1] += g;
                    }
                }
            }
        }
#else
        /* scalar k=2 stride=2 backward */
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float *x_chan = xg + ((size_t)n * C + c) * (size_t)H * W;
                float *g_chan = gd + ((size_t)n * C + c) * (size_t)H_out * W_out;
                for (int oh = 0; oh < H_out; oh++) {
                    int ih0 = oh * 2;
                    for (int ow = 0; ow < W_out; ow++) {
                        int iw0 = ow * 2;
                        float g = g_chan[oh * W_out + ow] * inv_k2;
                        x_chan[ih0 * W + iw0]     += g;
                        x_chan[ih0 * W + iw0 + 1] += g;
                        x_chan[(ih0+1) * W + iw0]   += g;
                        x_chan[(ih0+1) * W + iw0+1] += g;
                    }
                }
            }
        }
#endif
        return;
    }

    if (k == 2 && stride == 1 && tensor_is_contiguous(x) && tensor_is_contiguous(grad_output)) {
        /* Scalar k=2 stride=1 backward — each input pixel gets gradients
         * from up to 4 overlapping windows. */
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float *x_chan = xg + ((size_t)n * C + c) * (size_t)H * W;
                float *g_chan = gd + ((size_t)n * C + c) * (size_t)H_out * W_out;
                for (int oh = 0; oh < H_out; oh++) {
                    for (int ow = 0; ow < W_out; ow++) {
                        float g = g_chan[oh * W_out + ow] * inv_k2;
                        x_chan[oh * W + ow]     += g;
                        x_chan[oh * W + ow + 1] += g;
                        x_chan[(oh+1) * W + ow]   += g;
                        x_chan[(oh+1) * W + ow+1] += g;
                    }
                }
            }
        }
        return;
    }

    /* General backward: strided scatter */
    {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float *x_chan = xg + ((size_t)n * C + c) * (size_t)H * W;
                float *g_chan = gd + ((size_t)n * C + c) * (size_t)H_out * W_out;

                for (int oh = 0; oh < H_out; oh++) {
                    int ih0 = oh * stride;
                    for (int ow = 0; ow < W_out; ow++) {
                        int iw0 = ow * stride;
                        float g = g_chan[oh * W_out + ow] * inv_k2;
                        for (int di = 0; di < k; di++) {
                            int ih = ih0 + di;
                            float *row = x_chan + ih * W;
                            for (int dj = 0; dj < k; dj++)
                                row[iw0 + dj] += g;
                        }
                    }
                }
            }
        }
    }
}

/* ── tensor_avg_pool2d forward ── */

tensor *tensor_avg_pool2d(struct mem_pool *scratch, const tensor *x, int k, int stride) {
    assert(x);
    assert(x->ndim == 4 && "avg_pool2d: input must be (N, C, H, W)");
    assert(k > 0 && "avg_pool2d: kernel size must be positive");

    if (stride <= 0) stride = k;

    int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int H_out = (H - k) / stride + 1;
    int W_out = (W - k) / stride + 1;

    assert(H_out > 0 && W_out > 0 && "avg_pool2d: kernel too large for input");

    tensor *out = tensor_scratch(scratch, 4, (int[]){N, C, H_out, W_out}, 0);
    float *od = (float*)out->data;
    float *xd = (float*)x->data;

    float inv_k2 = 1.0f / (float)(k * k);

    if (!tensor_is_contiguous(x)) {
        /* Non-contiguous fallback */
        int out_numel = N * C * H_out * W_out;
        for (int i = 0; i < out_numel; i++) {
            int coord[4];
            int r = i;
            coord[3] = r % W_out; r /= W_out;
            coord[2] = r % H_out; r /= H_out;
            coord[1] = r % C;     r /= C;
            coord[0] = r;

            int oh = coord[2], ow = coord[3];
            int ih0 = oh * stride, iw0 = ow * stride;
            float sum = 0.0f;
            for (int di = 0; di < k; di++) {
                for (int dj = 0; dj < k; dj++) {
                    int ic[4] = {coord[0], coord[1], ih0 + di, iw0 + dj};
                    sum += xd[_bcast_off(x, 4, ic)];
                }
            }
            od[_flat_off(out, i)] = sum * inv_k2;
        }
        goto attach_grad;
    }

    /* Contiguous fast path */
    if (k == 2 && stride == 2) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float *x_chan = xd + ((size_t)n * C + c) * (size_t)H * W;
                float *o_chan = od + ((size_t)n * C + c) * (size_t)H_out * W_out;

                for (int oh = 0; oh < H_out; oh++) {
                    int ih0 = oh * 2;
                    const float *row0 = x_chan + ih0 * W;
                    const float *row1 = x_chan + (ih0 + 1) * W;
#if DNN_HAVE_NEON
                    avg_pool2d_fwd_k2s2_neon(row0, row1, o_chan + oh * W_out,
                                              W_out, W, inv_k2);
#else
                    for (int ow = 0; ow < W_out; ow++) {
                        int iw0 = ow * 2;
                        float sum = row0[iw0] + row0[iw0+1] + row1[iw0] + row1[iw0+1];
                        o_chan[oh * W_out + ow] = sum * inv_k2;
                    }
#endif
                }
            }
        }
    } else if (k == 2 && stride == 1) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float *x_chan = xd + ((size_t)n * C + c) * (size_t)H * W;
                float *o_chan = od + ((size_t)n * C + c) * (size_t)H_out * W_out;

                for (int oh = 0; oh < H_out; oh++) {
                    const float *row0 = x_chan + oh * W;
                    const float *row1 = x_chan + (oh + 1) * W;
#if DNN_HAVE_NEON
                    avg_pool2d_fwd_k2s1_neon(row0, row1, o_chan + oh * W_out,
                                              W_out, W, inv_k2);
#else
                    for (int ow = 0; ow < W_out; ow++) {
                        float sum = row0[ow] + row0[ow+1] + row1[ow] + row1[ow+1];
                        o_chan[oh * W_out + ow] = sum * inv_k2;
                    }
#endif
                }
            }
        }
    } else {
        /* General k, generic stride */
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float *x_chan = xd + ((size_t)n * C + c) * (size_t)H * W;
                float *o_chan = od + ((size_t)n * C + c) * (size_t)H_out * W_out;

                for (int oh = 0; oh < H_out; oh++) {
                    int ih0 = oh * stride;
                    for (int ow = 0; ow < W_out; ow++) {
                        int iw0 = ow * stride;
                        o_chan[oh * W_out + ow] =
                            avg_pool2d_fwd_general(x_chan, ih0, iw0, k, stride, W)
                            * inv_k2;
                    }
                }
            }
        }
    }

attach_grad:
    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(x)) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = avg_pool2d_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)x;

        fn->n_saved = 2;
        fn->saved_tensors = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
        int *k_saved = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *k_saved = k;
        fn->saved_tensors[0] = (tensor*)k_saved;
        int *s_saved = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *s_saved = stride;
        fn->saved_tensors[1] = (tensor*)s_saved;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
