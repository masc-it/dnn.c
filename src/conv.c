#include "conv.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
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
                int k_kh = k_c + kh * kW;
                for (int oh = 0; oh < H_out; oh++) {
                    int ih = oh * stride - pad + kh;
                    if (ih < 0 || ih >= H) continue;

                    const float *x_row = x + ((size_t)n * C + c) * H * W
                                          + (size_t)ih * W;
                    int m_base = n_off + oh * W_out;  /* M-index base for this output row */

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
 *  Winograd F(2×2, 3×3) — replaces im2col+GEMM for stride=1, pad=1
 *
 *  Transforms:
 *    V = G @ w @ G^T      (weight: 3×3 → 4×4)
 *    U = B^T @ tile @ B   (input: 4×4 → 4×4)
 *    Y = A^T @ M @ A      (output: 4×4 → 2×2)
 *
 *  Standard matrices for F(2,3) from Lavin & Gray 2016:
 *    B^T = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
 *    G   = [[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]]
 *    A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]
 *
 *  Dispatch: kH==3 && kW==3 && stride==1 && pad==1.  Fallback to im2col otherwise.
 * ══════════════════════════════════════════════════════════════════ */

/* transform 3×3 weight w into 4×4 Winograd domain V = G @ w @ G^T */
static inline void winograd_weight_transform(const float w[9], float V[16]) {
    float t[4][3];
    for (int j = 0; j < 3; j++) {
        float w0 = w[j], w1 = w[3 + j], w2 = w[6 + j];
        t[0][j] = w0;
        t[1][j] = (w0 + w1 + w2) * 0.5f;
        t[2][j] = (w0 - w1 + w2) * 0.5f;
        t[3][j] = w2;
    }
    for (int i = 0; i < 4; i++) {
        float ti0 = t[i][0], ti1 = t[i][1], ti2 = t[i][2];
        V[i*4 + 0] = ti0;
        V[i*4 + 1] = (ti0 + ti1 + ti2) * 0.5f;
        V[i*4 + 2] = (ti0 - ti1 + ti2) * 0.5f;
        V[i*4 + 3] = ti2;
    }
}

/* transform 4×4 input tile into Winograd domain U = B^T @ tile @ B */
static inline void winograd_input_transform(const float tile[16], float U[16]) {
    float temp[4][4];
    for (int j = 0; j < 4; j++) {
        float t0 = tile[j], t1 = tile[4 + j], t2 = tile[8 + j], t3 = tile[12 + j];
        temp[0][j] = t0 - t2;
        temp[1][j] = t1 + t2;
        temp[2][j] = t2 - t1;
        temp[3][j] = t1 - t3;
    }
    for (int i = 0; i < 4; i++) {
        float t0 = temp[i][0], t1 = temp[i][1], t2 = temp[i][2], t3 = temp[i][3];
        U[i*4 + 0] = t0 - t2;
        U[i*4 + 1] = t1 + t2;
        U[i*4 + 2] = t2 - t1;
        U[i*4 + 3] = t1 - t3;
    }
}

/* inverse transform 4×4 M into 2×2 output Y = A^T @ M @ A */
static inline void winograd_inverse_transform(const float M[16], float Y[4]) {
    float temp[2][4];
    for (int j = 0; j < 4; j++) {
        float m0 = M[j], m1 = M[4 + j], m2 = M[8 + j], m3 = M[12 + j];
        temp[0][j] = m0 + m1 + m2;
        temp[1][j] = m1 - m2 - m3;
    }
    for (int i = 0; i < 2; i++) {
        float t0 = temp[i][0], t1 = temp[i][1], t2 = temp[i][2], t3 = temp[i][3];
        Y[i*2 + 0] = t0 + t1 + t2;
        Y[i*2 + 1] = t1 - t2 - t3;
    }
}

/* backward of input transform: d_tile = B @ dU @ B^T
 *
 *  B  = [[1, 0,  0,  0],        B^T = [[1, 0, -1, 0],
 *         [0, 1, -1,  1],               [0, 1,  1, 0],
 *         [-1,1,  1,  0],               [0,-1,  1, 0],
 *         [0, 0,  0, -1]]               [0, 1,  0,-1]]
 *
 *  B @ dU: rows of B × columns of dU
 *    row0: [1,0,0,0]      → dU[0][j]
 *    row1: [0,1,-1,1]     → dU[1][j] - dU[2][j] + dU[3][j]
 *    row2: [-1,1,1,0]     → -dU[0][j] + dU[1][j] + dU[2][j]
 *    row3: [0,0,0,-1]     → -dU[3][j]
 *
 *  (B@dU) @ B^T: rows of temp × columns of B^T
 *    B^T[:,0] = [1,0,0,0]   → t[i][0]
 *    B^T[:,1] = [0,1,-1,1]  → t[i][1] - t[i][2] + t[i][3]
 *    B^T[:,2] = [-1,1,1,0]  → -t[i][0] + t[i][1] + t[i][2]
 *    B^T[:,3] = [0,0,0,-1]  → -t[i][3]
 */
static inline void winograd_input_backward(const float dU[16], float d_tile[16]) {
    float temp[4][4];
    for (int j = 0; j < 4; j++) {
        float u0 = dU[j], u1 = dU[4 + j], u2 = dU[8 + j], u3 = dU[12 + j];
        temp[0][j] = u0;
        temp[1][j] = u1 - u2 + u3;
        temp[2][j] = -u0 + u1 + u2;
        temp[3][j] = -u3;
    }
    for (int i = 0; i < 4; i++) {
        float t0 = temp[i][0], t1 = temp[i][1], t2 = temp[i][2], t3 = temp[i][3];
        d_tile[i*4 + 0] = t0;
        d_tile[i*4 + 1] = t1 - t2 + t3;
        d_tile[i*4 + 2] = -t0 + t1 + t2;
        d_tile[i*4 + 3] = -t3;
    }
}

/* backward of inverse transform: dM = A @ dY @ A^T (4×4 from 2×2 grad) */
static inline void winograd_output_backward(const float dY[4], float dM[16]) {
    float temp[4][2];
    float dy00 = dY[0], dy01 = dY[1], dy10 = dY[2], dy11 = dY[3];
    temp[0][0] = dy00;        temp[0][1] = dy01;
    temp[1][0] = dy00 + dy10; temp[1][1] = dy01 + dy11;
    temp[2][0] = dy00 - dy10; temp[2][1] = dy01 - dy11;
    temp[3][0] = -dy10;       temp[3][1] = -dy11;
    for (int i = 0; i < 4; i++) {
        float s0 = temp[i][0], s1 = temp[i][1];
        dM[i*4 + 0] = s0;
        dM[i*4 + 1] = s0 + s1;
        dM[i*4 + 2] = s0 - s1;
        dM[i*4 + 3] = -s1;
    }
}

/* backward of weight transform: dw[3×3] = G^T @ dV[4×4] @ G */
static inline void winograd_weight_backward(const float dV[16], float dw[9]) {
    float t[3][4];
    for (int j = 0; j < 4; j++) {
        float v0 = dV[j], v1 = dV[4 + j], v2 = dV[8 + j], v3 = dV[12 + j];
        t[0][j] = v0 + 0.5f * v1 + 0.5f * v2;
        t[1][j] = 0.5f * v1 - 0.5f * v2;
        t[2][j] = 0.5f * v1 + 0.5f * v2 + v3;
    }
    for (int i = 0; i < 3; i++) {
        float ti0 = t[i][0], ti1 = t[i][1], ti2 = t[i][2], ti3 = t[i][3];
        dw[i*3 + 0] = ti0 + 0.5f * (ti1 + ti2);
        dw[i*3 + 1] = 0.5f * (ti1 - ti2);
        dw[i*3 + 2] = 0.5f * (ti1 + ti2) + ti3;
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

    /* ── d_bias: sum over (N, H_out, W_out) ──
     *
     *  gd layout: (N, out_C, H_out, W_out) contiguous.
     *  For each oc, sum over all spatial positions in all batch items.
     *  Inner ow loop is vectorized via #pragma omp simd.
     */
    if (bias && tensor_requires_grad(bias)) {
        float *bg = _grad_ensure(bias);
        size_t oc_stride = (size_t)H_out * W_out;
#pragma omp parallel for
        for (int oc = 0; oc < out_C; oc++) {
            float acc = 0.0f;
            for (int n = 0; n < N; n++) {
                float *g_chan = gd + ((size_t)n * out_C + oc) * oc_stride;
                for (int oh = 0; oh < H_out; oh++) {
                    float *g_row = g_chan + (size_t)oh * W_out;
                    #pragma omp simd reduction(+:acc)
                    for (int ow = 0; ow < W_out; ow++)
                        acc += g_row[ow];
                }
            }
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
        /* Disable nested OMP: each thread calls threaded BLAS */
        int prev_level = omp_get_max_active_levels();
        omp_set_max_active_levels(1);
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
        omp_set_max_active_levels(prev_level);
    }

    /* ── d_input: col2im( gd(M,out_C) @ weight(out_C,K) ) ── */
    if (tensor_requires_grad(input)) {
        float *xg = _grad_ensure(input);

        size_t _dcm = mem_pool_mark(fn->pool);
        float *dcol = _mem_pool_alloc(fn->pool, (size_t)K * M * sizeof(float), NULL);

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

        mem_pool_release(fn->pool, _dcm);
    }

    /* forward's col freed on next batch's mem_pool_reset — no explicit release */
}

/* ══════════════════════════════════════════════════════════════════
 *  Winograd Conv2D backward (for stride=1, pad=1, 3×3)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    int N, C, H, W, out_C;
} winograd_shape;

static void winograd_conv2d_backward(grad_fn *fn, tensor *grad_output) {
    tensor *input    = fn->inputs[0];
    tensor *weight   = fn->inputs[1];
    tensor *bias     = (tensor*)fn->saved_tensors[0];
    winograd_shape *ws = (winograd_shape*)fn->saved_tensors[1];
    float *V_pre     = (float*)fn->saved_tensors[2];

    int N = ws->N, C = ws->C, H = ws->H, W = ws->W, out_C = ws->out_C;
    int H_out = H, W_out = W;  /* stride=1, pad=1 */
    int H_tiles = (H_out + 1) / 2, W_tiles = (W_out + 1) / 2;

    float *xd = tensor_data_ptr(input);
    float *gd = tensor_data_ptr(grad_output);
    int need_input = tensor_requires_grad(input);
    int need_weight = tensor_requires_grad(weight);
    int need_bias = bias && tensor_requires_grad(bias);

    float *xg = need_input ? _grad_ensure(input) : NULL;
    float *wg = need_weight ? _grad_ensure(weight) : NULL;

    /* ── d_bias: sum over (N, H_out, W_out) ── */
    if (need_bias) {
        float *bg = _grad_ensure(bias);
        size_t oc_stride = (size_t)H_out * W_out;
#pragma omp parallel for
        for (int oc = 0; oc < out_C; oc++) {
            float acc = 0.0f;
            for (int n = 0; n < N; n++) {
                float *g_chan = gd + ((size_t)n * out_C + oc) * oc_stride;
                for (int oh = 0; oh < H_out; oh++) {
                    float *g_row = g_chan + (size_t)oh * W_out;
                    #pragma omp simd reduction(+:acc)
                    for (int ow = 0; ow < W_out; ow++) acc += g_row[ow];
                }
            }
            bg[oc] += acc;
        }
    }

    /* ── d_weight: per-thread V-domain gradient accumulators ── */
    int dV_sz = (size_t)out_C * C * 16;
    float *dV = NULL;
    int n_threads = 0;
    if (need_weight) {
        n_threads = omp_get_max_threads();
        dV = _mem_pool_alloc(fn->pool, (size_t)n_threads * dV_sz * sizeof(float), NULL);
        memset(dV, 0, (size_t)n_threads * dV_sz * sizeof(float));
    }

    /* Parallelize over all tiles (N × H_tiles × W_tiles) */
    int n_tile_idx = H_tiles * W_tiles;
#pragma omp parallel for
    for (int tid = 0; tid < N * n_tile_idx; tid++) {
        /* Per-thread buffers: use heap if C*16 or out_C*16 exceeds stack-friendly size */
        float U_all_stack[64 * 16];
        float dM_all_stack[64 * 16];
        float *U_all = (C * 16 <= 64 * 16) ? U_all_stack : (float*)calloc((size_t)C * 16, sizeof(float));
        float *dM_all = (out_C * 16 <= 64 * 16) ? dM_all_stack : (float*)calloc((size_t)out_C * 16, sizeof(float));

        int n  = tid / n_tile_idx;
        int t  = tid % n_tile_idx;
        int tx = t % W_tiles;
        int ty = t / W_tiles;

        float tile_flat[16], dU[16], d_tile[16], tile_Y[4];

        int oh0 = ty * 2, ow0 = tx * 2;
        int ih_start = oh0 - 1, iw_start = ow0 - 1;

        /* ── Step 1: Precompute U[ic] for all input channels ── */
        for (int ic = 0; ic < C; ic++) {
            float *x_chan = xd + ((size_t)n * C + ic) * (size_t)H * W;
            for (int di = 0; di < 4; di++) {
                for (int dj = 0; dj < 4; dj++) {
                    int ih = ih_start + di, iw = iw_start + dj;
                    tile_flat[di*4 + dj] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        ? x_chan[ih * W + iw] : 0.0f;
                }
            }
            winograd_input_transform(tile_flat, U_all + ic * 16);
        }

        /* ── Step 2: For each oc, compute dM, accumulate dV ── */
        for (int oc = 0; oc < out_C; oc++) {
            float *g_slice = gd + ((size_t)n * out_C + oc) * (size_t)H_out * W_out;
            tile_Y[0] = g_slice[oh0 * W_out + ow0];
            tile_Y[1] = (ow0 + 1 < W_out) ? g_slice[oh0 * W_out + ow0 + 1] : 0.0f;
            tile_Y[2] = (oh0 + 1 < H_out) ? g_slice[(oh0 + 1) * W_out + ow0] : 0.0f;
            tile_Y[3] = (oh0 + 1 < H_out && ow0 + 1 < W_out)
                ? g_slice[(oh0 + 1) * W_out + ow0 + 1] : 0.0f;
            winograd_output_backward(tile_Y, dM_all + oc * 16);

            if (dV) {
                int thr = omp_get_thread_num();
                float *dV_local = dV + (size_t)thr * dV_sz;
                float *dM_oc = dM_all + oc * 16;
                for (int ic = 0; ic < C; ic++) {
                    float *U_ic = U_all + ic * 16;
                    float *dV_ocic = dV_local + ((size_t)oc * C + ic) * 16;
                    for (int k = 0; k < 16; k++)
                        dV_ocic[k] += U_ic[k] * dM_oc[k];
                }
            }
        }

        /* Free heap-allocated per-thread buffers */
        if (U_all != U_all_stack) free(U_all);
        if (dM_all != dM_all_stack) free(dM_all);

        /* ── Step 3: d_input scatter ── */
        if (xg) {
            for (int ic = 0; ic < C; ic++) {
                memset(dU, 0, sizeof(float) * 16);
                for (int oc = 0; oc < out_C; oc++) {
                    const float *V = V_pre + ((size_t)oc * C + ic) * 16;
                    float *dM_oc = dM_all + oc * 16;
                    for (int k = 0; k < 16; k++)
                        dU[k] += dM_oc[k] * V[k];
                }
                winograd_input_backward(dU, d_tile);
                float *x_chan_g = xg + ((size_t)n * C + ic) * (size_t)H * W;
                for (int di = 0; di < 4; di++) {
                    for (int dj = 0; dj < 4; dj++) {
                        int ih = ih_start + di, iw = iw_start + dj;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            x_chan_g[ih * W + iw] += d_tile[di*4 + dj];
                    }
                }
            }
        }
    }

    /* ── Finalize d_weight: reduce per-thread dV, then G^T @ dV @ G ── */
    if (wg) {
        /* Sum per-thread accumulators into thread 0's buffer */
        for (int thr = 1; thr < n_threads; thr++) {
            float *base = dV + (size_t)thr * dV_sz;
            for (int k = 0; k < dV_sz; k++)
                dV[k] += base[k];
        }
        /* dw = G^T @ dV @ G */
        float dw_flat[9];
        for (int oc = 0; oc < out_C; oc++) {
            for (int ic = 0; ic < C; ic++) {
                winograd_weight_backward(dV + ((size_t)oc * C + ic) * 16, dw_flat);
                float *wg_ocic = wg + ((size_t)oc * C + ic) * 9;
                for (int k = 0; k < 9; k++)
                    wg_ocic[k] += dw_flat[k];
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Conv2D forward (im2col-based)
 * ══════════════════════════════════════════════════════════════════ */

tensor *tensor_conv2d(struct mem_pool *scratch, tensor *input, tensor *weight, tensor *bias,
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

    tensor *out = tensor_scratch(scratch, 4,
        (int[]){N, out_C, H_out, W_out}, 0);

    float *xd = tensor_data_ptr(input);
    float *wd = tensor_data_ptr(weight);
    float *bd = bias ? tensor_data_ptr(bias) : NULL;
    float *od = tensor_data_ptr(out);

    /* dispatch: Winograd for 3×3 stride=1 pad=1, im2col+GEMM otherwise */
    int use_winograd = (kH == 3 && kW == 3 && stride == 1 && pad == 1);

    if (use_winograd) {
        /* ═══════════════════════════════════════════════════
         *  Winograd F(2×2, 3×3) forward
         * ═══════════════════════════════════════════════════ */
        int H_tiles = (H_out + 1) / 2, W_tiles = (W_out + 1) / 2;

        /* Pre-transform weights: V[oc][ic][4×4] */
        float *V = _mem_pool_alloc(scratch, (size_t)out_C * C * 16 * sizeof(float), NULL);
        for (int oc = 0; oc < out_C; oc++) {
            for (int ic = 0; ic < C; ic++) {
                const float *w_ocic = wd + ((size_t)oc * C + ic) * 9;
                winograd_weight_transform(w_ocic, V + ((size_t)oc * C + ic) * 16);
            }
        }

        /* Precompute per-oc bias offsets for write phase */
        float *bias_off = NULL;
        if (bd) { bias_off = _mem_pool_alloc(scratch, out_C * sizeof(float), NULL);
                  for (int oc = 0; oc < out_C; oc++) bias_off[oc] = bd[oc]; }

        /* Process tiles — parallel over batch */
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            float U[16], tile_flat[16], y_tile[4];

            for (int ty = 0; ty < H_tiles; ty++) {
                for (int tx = 0; tx < W_tiles; tx++) {
                    int oh0 = ty * 2, ow0 = tx * 2;
                    int ih_start = oh0 - 1, iw_start = ow0 - 1;

                    /* Precompute U for all channels (C=1 most common, worst case C up to 64) */
                    for (int ic = 0; ic < C; ic++) {
                        float *x_chan = xd + ((size_t)n * C + ic) * (size_t)H * W;
                        for (int di = 0; di < 4; di++) {
                            for (int dj = 0; dj < 4; dj++) {
                                int ih = ih_start + di, iw = iw_start + dj;
                                tile_flat[di*4 + dj] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    ? x_chan[ih * W + iw] : 0.0f;
                            }
                        }
                        winograd_input_transform(tile_flat, U);

                        /* M += U ⊙ V[oc][ic] for each oc */
                        for (int oc = 0; oc < out_C; oc++) {
                            /* Note: M is NOT zeroed between oc — each oc writes disjoint memory.
                               Use direct sdot-style accumulation into y_tile to avoid M copy. */
                            float M_local[16];
                            const float *V_ocic = V + ((size_t)oc * C + ic) * 16;
                            for (int k = 0; k < 16; k++)
                                M_local[k] = U[k] * V_ocic[k];
                            winograd_inverse_transform(M_local, y_tile);

                            float *y_chan = od + ((size_t)n * out_C + oc) * (size_t)H_out * W_out;
                            int oh1 = oh0 + 1 < H_out ? oh0 + 1 : oh0;
                            int ow1 = ow0 + 1 < W_out ? ow0 + 1 : ow0;
                            float b = bias_off ? bias_off[oc] : 0.0f;
                            /* accumulate if multiple ic */
                            if (ic == 0) {
                                y_chan[oh0 * W_out + ow0] = y_tile[0] + b;
                                if (oh1 > oh0) y_chan[oh1 * W_out + ow0] = y_tile[2] + b;
                                if (ow1 > ow0) y_chan[oh0 * W_out + ow1] = y_tile[1] + b;
                                if (oh1 > oh0 && ow1 > ow0) y_chan[oh1 * W_out + ow1] = y_tile[3] + b;
                            } else {
                                y_chan[oh0 * W_out + ow0] += y_tile[0];
                                if (oh1 > oh0) y_chan[oh1 * W_out + ow0] += y_tile[2];
                                if (ow1 > ow0) y_chan[oh0 * W_out + ow1] += y_tile[1];
                                if (oh1 > oh0 && ow1 > ow0) y_chan[oh1 * W_out + ow1] += y_tile[3];
                            }
                        }
                    }
                }
            }
        }

        /* ── autograd tape (Winograd) ── */
        int _needs_grad = dnn_grad_enabled() &&
            (tensor_requires_grad(input) || tensor_requires_grad(weight) ||
             (bias && tensor_requires_grad(bias)));

        if (_needs_grad) {
            grad_fn *fn = _grad_fn_create(scratch);
            fn->backward = winograd_conv2d_backward;
            fn->n_inputs = 2;
            fn->inputs = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
            fn->inputs[0] = (tensor*)input;
            fn->inputs[1] = (tensor*)weight;

            fn->n_saved = 3;
            fn->saved_tensors = _mem_pool_alloc(scratch, 3 * sizeof(void*), NULL);
            fn->saved_tensors[0] = (tensor*)bias;

            winograd_shape *ws = _mem_pool_alloc(scratch, sizeof(winograd_shape), NULL);
            ws->N = N; ws->C = C; ws->H = H; ws->W = W; ws->out_C = out_C;
            fn->saved_tensors[1] = (void*)ws;

            fn->saved_tensors[2] = (void*)V;  /* keep V for backward */

            out->requires_grad = 1;
            out->grad_fn = fn;
        }
    } else {
        /* ═══════════════════════════════════════════════════
         *  im2col + GEMM forward (general case)
         * ═══════════════════════════════════════════════════ */
        int M = N * H_out * W_out;
        int K = C * kH * kW;

        size_t _fcm = mem_pool_mark(scratch);
        float *col = _mem_pool_alloc(scratch, (size_t)K * M * sizeof(float), NULL);

        im2col(xd, col, N, C, H, W, kH, kW, pad, stride);

        /* Disable nested OMP before BLAS sections: each thread calls threaded BLAS */
        int _prev_omp_level = omp_get_max_active_levels();
        omp_set_max_active_levels(1);
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
        omp_set_max_active_levels(_prev_omp_level);

        /* ── autograd tape (im2col) ── */
        int _needs_grad = dnn_grad_enabled() &&
            (tensor_requires_grad(input) || tensor_requires_grad(weight) ||
             (bias && tensor_requires_grad(bias)));

        if (_needs_grad) {
            grad_fn *fn = _grad_fn_create(scratch);
            fn->backward = conv2d_backward;
            fn->n_inputs = 2;
            fn->inputs = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
            fn->inputs[0] = (tensor*)input;
            fn->inputs[1] = (tensor*)weight;

            fn->n_saved = 4;
            fn->saved_tensors = _mem_pool_alloc(scratch, 4 * sizeof(void*), NULL);
            fn->saved_tensors[0] = (tensor*)bias;

            conv2d_shape *cs = _mem_pool_alloc(scratch, sizeof(conv2d_shape), NULL);
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

            fn->saved_tensors[2] = (tensor*)col;
            fn->saved_tensors[3] = (void*)(uintptr_t)_fcm;

            out->requires_grad = 1;
            out->grad_fn = fn;
        } else {
            mem_pool_release(scratch, _fcm);
        }
    }

    return out;
}
