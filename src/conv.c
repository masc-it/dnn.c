#include "conv.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
#include "pool_int.h"
#include "tensor_int.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════════
 *  BLAS header — use Accelerate on macOS, generic cblas elsewhere
 * ══════════════════════════════════════════════════════════════════ */
#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#elif defined(HAVE_CBLAS)
#  include <cblas.h>
#else
   /* fallback: our own matmul (non-BLAS build) */
#  define NO_CBLAS 1
#endif

#include <omp.h>

/* ══════════════════════════════════════════════════════════════════
 *  im2col / col2im helpers
 *
 *  Layout: col stored as (K, M) instead of (M, K).
 *
 *    M = N*H_out*W_out    (spatial positions)
 *    K = C*kH*kW          (kernel volume)
 *
 *  (K, M) layout makes im2col writes sequential — every kernel-row
 *  sweep writes consecutive floats into col.  The GEMM uses
 *  CblasTrans on both operands to get the output back to (M, out_C).
 *
 *  col2im also benefits: restructured as (c,kh,kw,n,oh,ow) so dcol
 *  reads are sequential per kernel element.  Scattered dx writes
 *  are inherent to col2im regardless of layout.
 *
 *  PyTorch's im2col fallback uses the same (K, M) convention.
 *
 *  im2col: unfold input patches into columns.
 *    input (N,C,H,W)  →  col (K, M)
 *
 *  col2im: scatter columns back into a gradient image (accumulates).
 *    dcol (K, M)  →  dx (N,C,H,W)  (zeroed before call, accumulates)
 * ══════════════════════════════════════════════════════════════════ */

static void im2col(const float *x, float *col,
                   int N, int C, int H, int W,
                   int kH, int kW, int pad, int stride) {
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int K     = C * kH * kW;
    int M     = N * H_out * W_out;

    /* pre-zero: padding regions not written below; fast sequential */
    memset(col, 0, (size_t)K * M * sizeof(float));

    /* col stored as (K, M).  Inner loop writes sequential floats.
     *
     * Loop order: (n, c, kh, oh, kw, ow)
     *   — reads input sequentially from x_row (contiguous iw values)
     *   — writes col sequentially (ow increments the M index)
     *   — padding positions remain zero from the memset above
     */
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
        int n_off = n * H_out * W_out;
        for (int c = 0; c < C; c++) {
            int k_c = c * kH * kW;  /* kernel-offset component for channel c */
            for (int kh = 0; kh < kH; kh++) {
                /* precompute oh range that yields valid ih \in [0, H) */
                int _req = pad - kh;
                int oh_min = _req > 0 ? (_req + stride - 1) / stride : 0;
                int oh_max = (H + pad - kh - 1) / stride;
                if (oh_max > H_out - 1) oh_max = H_out - 1;
                if (oh_min > oh_max) continue;

                int k_kh = k_c + kh * kW;
                for (int oh = oh_min; oh <= oh_max; oh++) {
                    int ih = oh * stride - pad + kh;
                    /* ih \in [0, H) guaranteed by oh bounds above */

                    const float *x_row = x + ((size_t)n * C + c) * H * W
                                          + (size_t)ih * W;
                    int m_base = n_off + oh * W_out;

                    for (int kw = 0; kw < kW; kw++) {
                        int k = k_kh + kw;            /* row in (K, M) */
                        int idx_base = k * M + m_base;

                        /* precompute ow range — no checks in inner loop */
                        int w0 = kw - pad;             /* iw at ow=0 */
                        int ow0 = 0, ow1 = W_out - 1;
                        if (w0 < 0)  ow0 = (-w0 + stride - 1) / stride;
                        if (w0 + (W_out - 1) * stride >= W)
                            ow1 = (W - 1 - w0) / stride;
                        if (ow0 > ow1) continue;

                        const float *xp = x_row + w0 + ow0 * stride;
                        float *cp = col + idx_base + ow0;
                        int n_ow = ow1 - ow0 + 1;
                        for (int i = 0; i < n_ow; i++) {
                            cp[i] = *xp;
                            xp  += stride;
                        }
                    }
                }
            }
        }
    }
}

static void col2im(const float *dcol, float *dx,
                   int N, int C, int H, int W,
                   int kH, int kW, int pad, int stride) {
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int M     = N * H_out * W_out;

    /* dcol is (K, M).  Read sequential by iterating k outside, m inside.
     * (oh, ow) ranges precomputed per (kh, kw) — bounds peel eliminates
     * all per-pixel conditionals in the hot path.
     */
#pragma omp parallel for
    for (int c = 0; c < C; c++) {
        int k_c = c * kH * kW * M;

        for (int kh = 0; kh < kH; kh++) {
            int k_kh = k_c + kh * kW * M;

            /* precompute oh range for this kh */
            int ih0 = -pad + kh;
            int oh0 = 0, oh1 = H_out - 1;
            if (ih0 < 0) oh0 = (-ih0 + stride - 1) / stride;
            if (ih0 + (H_out - 1) * stride >= H)
                oh1 = (H - 1 - ih0) / stride;
            if (oh0 > oh1) continue;

            for (int kw = 0; kw < kW; kw++) {
                int k_off = k_kh + kw * M;

                /* precompute ow range for this kw */
                int iw0 = -pad + kw;
                int ow0 = 0, ow1 = W_out - 1;
                if (iw0 < 0) ow0 = (-iw0 + stride - 1) / stride;
                if (iw0 + (W_out - 1) * stride >= W)
                    ow1 = (W - 1 - iw0) / stride;
                if (ow0 > ow1) continue;

                for (int n = 0; n < N; n++) {
                    for (int oh = oh0; oh <= oh1; oh++) {
                        int ih = ih0 + oh * stride;     /* guaranteed in [0,H) */
                        float *dx_row_p = dx + ((size_t)n * C + c) * H * W
                                             + (size_t)ih * W;
                        int m_base = (n * H_out + oh) * W_out;

                        const float *dc = dcol + k_off + m_base + ow0;
                        float *dxp = dx_row_p;
                        int n_ow = ow1 - ow0 + 1;
                        int iw = iw0 + ow0 * stride;
                        for (int i = 0; i < n_ow; i++) {
                            dxp[iw] += dc[i];
                            iw += stride;
                        }
                    }
                }
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Conv2D backward (im2col-based)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    int stride, pad;
    int N, C, H, W, out_C, kH, kW;
} conv2d_shape;

static void conv2d_backward(grad_fn *fn, tensor *grad_output) {
    tensor *input    = fn->inputs[0];
    tensor *weight   = fn->inputs[1];
    tensor *bias     = (tensor*)fn->saved_tensors[0];
    conv2d_shape *cs = (conv2d_shape*)fn->saved_tensors[1];

    int stride = cs->stride, pad = cs->pad;
    int N = cs->N, C = cs->C, H = cs->H, W = cs->W;
    int out_C = cs->out_C, kH = cs->kH, kW = cs->kW;
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int M = N * H_out * W_out;
    int K = C * kH * kW;

    float *wd = tensor_data_ptr(weight);
    float *gd = tensor_data_ptr(grad_output);

    /* ── d_bias: sum over (N, H_out, W_out) ── */
    if (bias && tensor_requires_grad(bias)) {
        float *bg = _grad_ensure(bias);
#pragma omp parallel for
        for (int oc = 0; oc < out_C; oc++) {
            float acc = 0.0f;
            for (int n = 0; n < N; n++)
                for (int oh = 0; oh < H_out; oh++)
                    for (int ow = 0; ow < W_out; ow++)
                        acc += gd[(n * out_C + oc) * H_out * W_out
                                  + oh * W_out + ow];
            bg[oc] += acc;
        }
    }

    /* ── d_weight: gd^T @ col^T  =  (out_C,K) ← gd^T(out_C,M) @ col^T(M,K)
     *
     *   col is (K, M).  gd is (M, out_C).
     *   Parallelize over blocks of output channels.  Each thread does
     *   a small sgemm on its slice of gd and wg.
     */
    if (tensor_requires_grad(weight)) {
        float *col = (float*)fn->saved_tensors[2];  /* reuse forward's col */
        float *wg = _grad_ensure(weight);
        int dw_block = out_C > 16 ? 16 : out_C;
#pragma omp parallel for
        for (int oc = 0; oc < out_C; oc += dw_block) {
            int oc_this = out_C - oc;
            if (oc_this > dw_block) oc_this = dw_block;
#if NO_CBLAS
            for (int oci = 0; oci < oc_this; oci++)
                for (int k = 0; k < K; k++) {
                    float sum = 0.0f;
                    for (int m = 0; m < M; m++)
                        sum += col[(size_t)k * M + m]
                             * gd[(size_t)m * out_C + oc + oci];
                    wg[(size_t)(oc + oci) * K + k] += sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        oc_this, K, M,
                        1.0f, gd + oc, out_C,
                        col, M,
                        1.0f, wg + (size_t)oc * K, K);
#endif
        }
    }

    /* ── d_input: col2im( gd(M,out_C) @ weight(out_C,K) ) ── */
    if (tensor_requires_grad(input)) {
        float *xg = _grad_ensure(input);

        size_t _dcm = mem_pool_mark(_mem_pool_scratch());
        float *dcol = _mem_pool_alloc_nz(_mem_pool_scratch(), (size_t)K * M * sizeof(float));

        /* dcol(K,M) = wd^T(K,out_C) @ gd^T(out_C,M)
         *   Each row k of dcol is independent: sum_oc wd[oc][k] * gd[:,oc].
         *   Parallelize over kernel elements — each thread writes one row.
         */
#pragma omp parallel for
        for (int k = 0; k < K; k++) {
#if NO_CBLAS
            float *dcol_row = dcol + (size_t)k * M;
            for (int m = 0; m < M; m++) {
                float sum = 0.0f;
                for (int oc = 0; oc < out_C; oc++)
                    sum += gd[(size_t)m * out_C + oc]
                         * wd[(size_t)oc * K + k];
                dcol_row[m] = sum;
            }
#else
            /* y(M) = gd(M,out_C) @ wd_col_k(out_C,)  — sgemv NoTrans
             *   gd: M×out_C, lda=out_C
             *   wd[:,k]: out_C elements, stride=K
             *   y: M elements, stride=1
             */
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        M, out_C, 1.0f, gd, out_C, wd + k, K,
                        0.0f, dcol + (size_t)k * M, 1);
#endif
        }

        col2im(dcol, xg, N, C, H, W, kH, kW, pad, stride);

        mem_pool_release(_mem_pool_scratch(), _dcm);
    }

    /* forward's col freed on next batch's mem_pool_reset — no explicit release */
}

/* ══════════════════════════════════════════════════════════════════
 *  Conv2D forward (im2col-based)
 * ══════════════════════════════════════════════════════════════════ */

tensor *tensor_conv2d(tensor *input, tensor *weight, tensor *bias,
                      int stride, int pad) {
    assert(input);
    assert(weight);
    assert(input->ndim == 4 && "conv2d: input must be (N, C, H, W)");
    assert(weight->ndim == 4 && "conv2d: weight must be (out_C, in_C, kH, kW)");
    assert(input->shape[1] == weight->shape[1] && "conv2d: in_channels mismatch");
    assert(tensor_is_contiguous(input) && "conv2d: input must be contiguous");
    assert(tensor_is_contiguous(weight) && "conv2d: weight must be contiguous");
    assert(bias == NULL
           || (bias->ndim == 1 && bias->shape[0] == weight->shape[0]));

    int N     = input->shape[0];
    int C     = input->shape[1];
    int H     = input->shape[2];
    int W     = input->shape[3];
    int out_C = weight->shape[0];
    int kH    = weight->shape[2];
    int kW    = weight->shape[3];

    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int M     = N * H_out * W_out;
    int K     = C * kH * kW;

    tensor *out = _tensor_scratch_create(4,
        (int[]){N, out_C, H_out, W_out}, 0);

    float *xd = tensor_data_ptr(input);
    float *wd = tensor_data_ptr(weight);
    float *bd = bias ? tensor_data_ptr(bias) : NULL;
    float *od = tensor_data_ptr(out);

    /* ── im2col → matmul ── */
    size_t _fcm = mem_pool_mark(_mem_pool_scratch());
    float *col = mem_scratch_alloc((size_t)K * M * sizeof(float), NULL);

    im2col(xd, col, N, C, H, W, kH, kW, pad, stride);

    /* matmul with channel-parallel sgemm/blocks
     *
     * col stored as (K, M).  wd stored as (out_C, K).
     * Split out_C into blocks (~16 channels each), run each block's
     * sgemm on a separate thread.  Each writes disjoint columns of od.
     */
#if NO_CBLAS
#pragma omp parallel for
    for (int oc = 0; oc < out_C; oc++) {
        for (int m = 0; m < M; m++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += col[(size_t)k * M + m] * wd[(size_t)oc * K + k];
            od[(size_t)m * out_C + oc] = sum + (bd ? bd[oc] : 0.0f);
        }
    }
#else
    {
        int block_c = out_C > 16 ? 16 : out_C;
        if (bd) {
            /* pre-fill od with bias then beta=1 accum */
#pragma omp parallel for
            for (int oc = 0; oc < out_C; oc += block_c) {
                int oc_this = out_C - oc;
                if (oc_this > block_c) oc_this = block_c;
                for (int m = 0; m < M; m++)
                    memcpy(&od[(size_t)m * out_C + oc], bd + oc,
                           (size_t)oc_this * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                            M, oc_this, K, 1.0f, col, M, wd + (size_t)oc * K, K,
                            1.0f, od + oc, out_C);
            }
        } else {
#pragma omp parallel for
            for (int oc = 0; oc < out_C; oc += block_c) {
                int oc_this = out_C - oc;
                if (oc_this > block_c) oc_this = block_c;
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                            M, oc_this, K, 1.0f, col, M, wd + (size_t)oc * K, K,
                            0.0f, od + oc, out_C);
            }
        }
    }
#endif

    /* ── autograd tape ── */
    int _needs_grad = dnn_grad_enabled() &&
        (tensor_requires_grad(input) || tensor_requires_grad(weight) ||
         (bias && tensor_requires_grad(bias)));

    if (_needs_grad) {
        /* keep col live — backward d_weight reuses it; release after backward */
        grad_fn *fn = _grad_fn_create();
        fn->backward = conv2d_backward;
        fn->n_inputs = 2;
        fn->inputs = mem_scratch_alloc(2 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)input;
        fn->inputs[1] = (tensor*)weight;

        fn->n_saved = 4;
        fn->saved_tensors = mem_scratch_alloc(4 * sizeof(void*), NULL);
        fn->saved_tensors[0] = (tensor*)bias;

        conv2d_shape *cs = mem_scratch_alloc(sizeof(conv2d_shape), NULL);
        cs->stride = stride;
        cs->pad    = pad;
        cs->N      = N;
        cs->C      = C;
        cs->H      = H;
        cs->W      = W;
        cs->out_C  = out_C;
        cs->kH     = kH;
        cs->kW     = kW;
        fn->saved_tensors[1] = (void*)cs;

        fn->saved_tensors[2] = (tensor*)col;             /* fwd col buffer */
        fn->saved_tensors[3] = (void*)(uintptr_t)_fcm;  /* scratch mark to release after bwd */

        out->requires_grad = 1;
        out->grad_fn = fn;
    } else {
        mem_pool_release(_mem_pool_scratch(), _fcm);
    }

    return out;
}
