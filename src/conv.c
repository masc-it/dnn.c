#include "conv.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
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

/* ══════════════════════════════════════════════════════════════════
 *  im2col / col2im helpers
 *
 *  im2col: unfold input patches into columns.
 *    input (N,C,H,W)  →  col (M, K)   where M = N*H_out*W_out, K = C*kH*kW
 *
 *  col2im: scatter columns back into a gradient image (accumulates).
 *    dcol (M, K)  →  dx (N,C,H,W)  (zeroed before call, accumulates into dx)
 * ══════════════════════════════════════════════════════════════════ */

static void im2col(const float *x, float *col,
                   int N, int C, int H, int W,
                   int kH, int kW, int pad, int stride) {
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int K     = C * kH * kW;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int row_start = (n * H_out * W_out + oh * W_out + ow) * K;
                int h_start   = oh * stride - pad;
                int w_start   = ow * stride - pad;

                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = h_start + kh;

                        /* entire kh-row is out of bounds → fill with zeros */
                        if (ih < 0 || ih >= H) {
                            memset(&col[row_start + c * kH * kW + kh * kW],
                                   0, kW * sizeof(float));
                            continue;
                        }

                        for (int kw = 0; kw < kW; kw++) {
                            int iw = w_start + kw;
                            int off = row_start + c * kH * kW + kh * kW + kw;
                            if (iw >= 0 && iw < W)
                                col[off] = x[(n * C + c) * H * W + ih * W + iw];
                            else
                                col[off] = 0.0f;
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
    int K     = C * kH * kW;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int row_start = (n * H_out * W_out + oh * W_out + ow) * K;
                int h_start   = oh * stride - pad;
                int w_start   = ow * stride - pad;

                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = h_start + kh;
                        if (ih < 0 || ih >= H) continue;

                        for (int kw = 0; kw < kW; kw++) {
                            int iw = w_start + kw;
                            if (iw < 0 || iw >= W) continue;

                            int col_off = row_start + c * kH * kW + kh * kW + kw;
                            dx[(n * C + c) * H * W + ih * W + iw] += dcol[col_off];
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

    float *xd = tensor_data_ptr(input);
    float *wd = tensor_data_ptr(weight);
    float *gd = tensor_data_ptr(grad_output);

    /* ── d_bias: sum over (N, H_out, W_out) ── */
    if (bias && tensor_requires_grad(bias)) {
        float *bg = _grad_ensure(bias);
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < out_C; oc++) {
                float acc = 0.0f;
                for (int oh = 0; oh < H_out; oh++)
                    for (int ow = 0; ow < W_out; ow++)
                        acc += gd[(n * out_C + oc) * H_out * W_out
                                  + oh * W_out + ow];
                bg[oc] += acc;
            }
        }
    }

    /* ── d_weight: gd^T @ col (out_C,K) = gd^T(out_C,M) @ col(M,K) ──
     *
     *   weight layout is (out_C, K), so computing gd^T @ col directly
     *   produces (out_C, K) — matches wg storage without transposition.
     */
    if (tensor_requires_grad(weight)) {
        float *col = malloc((size_t)M * K * sizeof(float));
        if (!col) { fprintf(stderr, "conv2d_backward: malloc(col) failed\n"); abort(); }
        im2col(xd, col, N, C, H, W, kH, kW, pad, stride);

        float *wg = _grad_ensure(weight);
#if NO_CBLAS
        for (int oc = 0; oc < out_C; oc++)
            for (int k = 0; k < K; k++) {
                float sum = 0.0f;
                for (int m = 0; m < M; m++)
                    sum += col[(size_t)m * K + k]
                         * gd[(size_t)m * out_C + oc];
                wg[(size_t)oc * K + k] += sum;
            }
#else
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    out_C, K, M,
                    1.0f, gd, out_C,
                    col, K,
                    1.0f, wg, K);
#endif
        free(col);
    }

    /* ── d_input: col2im( gd(M,out_C) @ weight(out_C,K) ) ── */
    if (tensor_requires_grad(input)) {
        float *dcol = malloc((size_t)M * K * sizeof(float));
        if (!dcol) { fprintf(stderr, "conv2d_backward: malloc(dcol) failed\n"); abort(); }

#if NO_CBLAS
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                float sum = 0.0f;
                for (int oc = 0; oc < out_C; oc++)
                    sum += gd[(size_t)m * out_C + oc]
                         * wd[(size_t)oc * K + k];
                dcol[(size_t)m * K + k] = sum;
            }
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, K, out_C,
                    1.0f, gd, out_C,
                    wd, K,
                    0.0f, dcol, K);
#endif

        float *xg = _grad_ensure(input);
        col2im(dcol, xg, N, C, H, W, kH, kW, pad, stride);

        free(dcol);
    }
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
    float *col = malloc((size_t)M * K * sizeof(float));
    if (!col) { fprintf(stderr, "conv2d_forward: malloc(col) failed\n"); abort(); }

    im2col(xd, col, N, C, H, W, kH, kW, pad, stride);

    /* matmul: col(M,K) @ weight(K,out_C) = out(M,out_C)
     *
     * weight stored as (out_C, K).  We read wd[oc * K + k].
     * out stored as (N, out_C, H_out, W_out) — index as 2D for speed.
     */
    for (int m = 0; m < M; m++) {
        for (int oc = 0; oc < out_C; oc++) {
            float sum = bd ? bd[oc] : 0.0f;
            for (int k = 0; k < K; k++)
                sum += col[(size_t)m * K + k] * wd[(size_t)oc * K + k];
            od[(size_t)m * out_C + oc] = sum;
        }
    }

    free(col);

    /* ── autograd tape ── */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(input) || tensor_requires_grad(weight) ||
         (bias && tensor_requires_grad(bias))))
    {
        grad_fn *fn = _grad_fn_create();
        fn->backward = conv2d_backward;
        fn->n_inputs = 2;
        fn->inputs = mem_scratch_alloc(2 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)input;
        fn->inputs[1] = (tensor*)weight;

        fn->n_saved = 2;
        fn->saved_tensors = mem_scratch_alloc(2 * sizeof(void*), NULL);
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

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
