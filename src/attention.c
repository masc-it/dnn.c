#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "autograd_int.h"
#include "broadcast.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* BLAS for fast matmul */
#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#elif defined(HAVE_CBLAS)
#  include <cblas.h>
#else
#  define NO_CBLAS 1
#endif

#include "simd.h"

/* ── Tuning constants ── */

#include "attention.h"  /* DNN_ATTENTION_TILE_ROWS */

#ifndef DNN_ATTENTION_DENSE_THRESHOLD
#define DNN_ATTENTION_DENSE_THRESHOLD 0   /* 0 = always use triangular */
#endif

/* Uncomment to enable dense reference path for debugging */
/* #define DNN_ATTENTION_DENSE_REF 1 */


/* ══════════════════════════════════════════════════════════════════
 *  Attention mode helpers (visible length, tile S)
 * ══════════════════════════════════════════════════════════════════ */

static inline int attention_visible_len(attention_mode mode, int i,
                                         int prefix_len) {
    switch (mode) {
    case ATTENTION_CAUSAL:
        return i + 1;
    case ATTENTION_PREFIX_LM:
        return (i < prefix_len) ? prefix_len : i + 1;
    default:
        assert(0 && "unknown attention mode");
        return i + 1;
    }
}

static inline int attention_tile_S(attention_mode mode, int r0, int r1,
                                    int prefix_len) {
    (void)r0;
    switch (mode) {
    case ATTENTION_CAUSAL:
        return r1;
    case ATTENTION_PREFIX_LM:
        return (r1 <= prefix_len) ? prefix_len : r1;
    default:
        assert(0 && "unknown attention mode");
        return r1;
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Row-prefix softmax helpers (no tile-row notion, just [0,visible))
 * ══════════════════════════════════════════════════════════════════ */

/* Forward: online max + sum_exp over visible prefix [0, visible_len) */
static void attention_softmax_row_prefix(const float *scores,
                                          float *p,
                                          int visible_len) {
    float mx = -INFINITY;
    float se = 0.0f;
#if DNN_HAVE_NEON
    {
        float32x4_t vmx = vdupq_n_f32(-INFINITY);
        float32x4_t vse = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 4 <= visible_len; j += 4) {
            float32x4_t v = vld1q_f32(scores + j);
            float group_max = vmaxvq_f32(v);
            if (group_max > mx) {
                vse = vmulq_f32(vse, vdupq_n_f32(expf(mx - group_max)));
                mx = group_max;
                vmx = vdupq_n_f32(mx);
            }
            float32x4_t shifted = vsubq_f32(v, vmx);
            vse = vaddq_f32(vse, simd_expf_f32(shifted));
        }
        se = vaddvq_f32(vse);
        for (; j < visible_len; j++) {
            float old_mx = mx;
            if (scores[j] > mx) mx = scores[j];
            if (mx != old_mx) se *= expf(old_mx - mx);
            se += expf(scores[j] - mx);
        }
    }
#else
    for (int j = 0; j < visible_len; j++) {
        float old_mx = mx;
        if (scores[j] > mx) mx = scores[j];
        if (mx != old_mx) se *= expf(old_mx - mx);
        se += expf(scores[j] - mx);
    }
#endif

    float inv_se = 1.0f / se;
#if DNN_HAVE_NEON
    {
        float32x4_t vmx = vdupq_n_f32(mx);
        float32x4_t vinv_se = vdupq_n_f32(inv_se);
        int j = 0;
        for (; j + 4 <= visible_len; j += 4) {
            float32x4_t v = vld1q_f32(scores + j);
            float32x4_t exp_v = simd_expf_f32(vsubq_f32(v, vmx));
            vst1q_f32(p + j, vmulq_f32(exp_v, vinv_se));
        }
        for (; j < visible_len; j++)
            p[j] = expf(scores[j] - mx) * inv_se;
    }
#else
    for (int j = 0; j < visible_len; j++)
        p[j] = expf(scores[j] - mx) * inv_se;
#endif
}

/* Backward: dS = P * (dP - dot) * scale over visible prefix.
 * Caller zeroes tail [visible_len, S). */
static void attention_softmax_bwd_row_prefix(const float *p,
                                              const float *dp,
                                              float *ds,
                                              int visible_len,
                                              float scale) {
    float dot;
#if DNN_HAVE_NEON
    {
        float32x4_t vdot = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 4 <= visible_len; j += 4) {
            vdot = vfmaq_f32(vdot, vld1q_f32(p + j), vld1q_f32(dp + j));
        }
        dot = vaddvq_f32(vdot);
        for (; j < visible_len; j++) dot += p[j] * dp[j];
    }
#else
    dot = 0.0f;
    for (int j = 0; j < visible_len; j++)
        dot += p[j] * dp[j];
#endif

#if DNN_HAVE_NEON
    {
        float32x4_t vscale = vdupq_n_f32(scale);
        float32x4_t vdot = vdupq_n_f32(dot);
        int j = 0;
        for (; j + 4 <= visible_len; j += 4) {
            float32x4_t vp  = vld1q_f32(p + j);
            float32x4_t vdp = vld1q_f32(dp + j);
            float32x4_t vds = vmulq_f32(vp, vmulq_f32(vsubq_f32(vdp, vdot), vscale));
            vst1q_f32(ds + j, vds);
        }
        for (; j < visible_len; j++)
            ds[j] = p[j] * (dp[j] - dot) * scale;
    }
#else
    for (int j = 0; j < visible_len; j++)
        ds[j] = p[j] * (dp[j] - dot) * scale;
#endif
}


/* ══════════════════════════════════════════════════════════════════
 *  Dense reference forward (guarded by DNN_ATTENTION_DENSE_REF)
 * ══════════════════════════════════════════════════════════════════ */

#ifdef DNN_ATTENTION_DENSE_REF

static void attention_forward_dense(int B, int H, int N, int d,
                                    float scale,
                                    const float *qd, const float *kd, const float *vd,
                                    float *Pd, float *od,
                                    tensor *mask,
                                    float *scores_buf,
                                    int n_workers) {
    (void)n_workers;
#pragma omp parallel for collapse(2) if (B * H >= 2)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;
            int tid = omp_get_thread_num();
            float *scores = scores_buf + (size_t)tid * N * N;

            float *q_slice = (float*)qd + (size_t)bh * N * d;
            float *k_slice = (float*)kd + (size_t)bh * N * d;
            float *v_slice = (float*)vd + (size_t)bh * N * d;
            float *p_slice = Pd + (size_t)bh * N * N;
            float *o_slice = od + (size_t)bh * N * d;

#if NO_CBLAS
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < d; kk++)
                        sum += q_slice[i * d + kk] * k_slice[j * d + kk];
                    scores[i * N + j] = sum * scale;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N, N, d, scale, q_slice, d, k_slice, d,
                        0.0f, scores, N);
#endif

            if (mask) {
                float *md = (float*)mask->data;
                int mask_ndim = mask->ndim;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) {
                        int coord[4] = {b, h, i, j};
                        int m_off = _bcast_off(mask, mask_ndim, coord);
                        scores[i * N + j] += md[m_off];
                    }
            }

            for (int i = 0; i < N; i++) {
                float *row = scores + i * N;
                float *p_row = p_slice + i * N;
                int visible = i + 1;
                attention_softmax_row_prefix(row, p_row, visible);
                for (int j = visible; j < N; j++)
                    p_row[j] = 0.0f;
            }

#if NO_CBLAS
            for (int i = 0; i < N; i++)
                for (int j = 0; j < d; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < N; kk++)
                        sum += p_slice[i * N + kk] * v_slice[kk * d + j];
                    o_slice[i * d + j] = sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, d, N, 1.0f, p_slice, N, v_slice, d,
                        0.0f, o_slice, d);
#endif
        }
    }
}

#endif /* DNN_ATTENTION_DENSE_REF */


/* ══════════════════════════════════════════════════════════════════
 *  Triangular tiled forward (row-blocked prefix attention)
 * ══════════════════════════════════════════════════════════════════ */

static void attention_forward_triangular(int B, int H, int N, int d,
                                         float scale,
                                         const float *qd, const float *kd, const float *vd,
                                         float *Pd, float *od,
                                         tensor *mask,
                                         float *scores_buf,
                                         float *p_buf,
                                         int TB,
                                         attention_mode mode,
                                         int prefix_len,
                                         const int *seq_lens) {
#pragma omp parallel for collapse(2) if (B * H >= 2)
    for (int b = 0; b < B; b++) {
        int N_b = seq_lens ? seq_lens[b] : N;
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;
            int tid = omp_get_thread_num();
            float *worker_scores = scores_buf + (size_t)tid * TB * N;
            float *worker_p      = p_buf      + (size_t)tid * TB * N;

            const float *q_slice = qd + (size_t)bh * N * d;
            const float *k_slice = kd + (size_t)bh * N * d;
            const float *v_slice = vd + (size_t)bh * N * d;
            float *p_slice = Pd ? Pd + (size_t)bh * N * N : NULL;
            float *o_slice = od + (size_t)bh * N * d;

            for (int r0 = 0; r0 < N_b; r0 += TB) {
                int r1 = r0 + TB;
                if (r1 > N_b) r1 = N_b;
                int M = r1 - r0;
                int S = attention_tile_S(mode, r0, r1, prefix_len);
                if (S > N_b) S = N_b;

                float *scores = worker_scores;
                float *ptmp   = worker_p;

                /* scores = Q_tile @ K_prefix^T  [M,S] = [M,d] @ [S,d]^T */
#if NO_CBLAS
                for (int mi = 0; mi < M; mi++) {
                    int i = r0 + mi;
                    for (int j = 0; j < S; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < d; kk++)
                            sum += q_slice[i * d + kk] * k_slice[j * d + kk];
                        scores[mi * S + j] = sum * scale;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, S, d, scale,
                            q_slice + r0 * d, d,
                            k_slice, d,
                            0.0f, scores, S);
#endif

                /* Add additive mask if provided (cols 0..S-1) */
                if (mask) {
                    float *md = (float*)mask->data;
                    int mask_ndim = mask->ndim;
                    for (int mi = 0; mi < M; mi++) {
                        int i = r0 + mi;
                        for (int j = 0; j < S; j++) {
                            int coord[4] = {b, h, i, j};
                            int m_off = _bcast_off(mask, mask_ndim, coord);
                            scores[mi * S + j] += md[m_off];
                        }
                    }
                }

                /* Softmax per row: visible = attention_visible_len(mode, i, prefix_len) */
                for (int mi = 0; mi < M; mi++) {
                    int i = r0 + mi;
                    int visible = attention_visible_len(mode, i, prefix_len);
                    if (visible > N_b) visible = N_b;

                    float *row_s = scores + mi * S;
                    float *row_p = ptmp   + mi * S;

                    attention_softmax_row_prefix(row_s, row_p, visible);

                    /* Zero future columns in p_tile (j >= visible) */
                    int tail = S - visible;
                    if (tail > 0)
                        memset(row_p + visible, 0, (size_t)tail * sizeof(float));

                    /* Save row into full P matrix for backward.
                     * row_p already has visible..S-1 zeroed.  Copy all S
                     * elements so backward reads valid data, then zero
                     * pad region S..N-1 (tensor_scratch is uninitialized). */
                    if (p_slice) {
                        float *p_row = p_slice + i * N;
                        memcpy(p_row, row_p, (size_t)S * sizeof(float));
                        if (S < N)
                            memset(p_row + S, 0, (size_t)(N - S) * sizeof(float));
                    }
                }

                /* O_tile = P_tile @ V_prefix  [M,d] = [M,S] @ [S,d] */
#if NO_CBLAS
                for (int mi = 0; mi < M; mi++) {
                    int i = r0 + mi;
                    for (int j = 0; j < d; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < S; kk++)
                            sum += ptmp[mi * S + kk] * v_slice[kk * d + j];
                        o_slice[i * d + j] = sum;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, d, S, 1.0f,
                            ptmp, S,
                            v_slice, d,
                            0.0f, o_slice + r0 * d, d);
#endif
            }
            /* Zero pad region N_b..N in output */
            if (N_b < N)
                memset(o_slice + N_b * d, 0, (size_t)(N - N_b) * d * sizeof(float));
        }
    }
}


/* ══════════════════════════════════════════════════════════════════
 *  Triangular tiled backward
 * ══════════════════════════════════════════════════════════════════ */

static void attention_backward_triangular(int B, int H, int N, int d,
                                          float scale,
                                          const float *qd, const float *kd, const float *vd,
                                          const float *Pd,
                                          const float *gd,
                                          float *qg, float *kg, float *vg,
                                          float *p_buf,
                                          float *dP_buf,
                                          float *dS_buf,
                                          int TB,
                                          attention_mode mode,
                                          int prefix_len,
                                          const int *seq_lens) {
#pragma omp parallel for collapse(2) if (B * H >= 2)
    for (int b = 0; b < B; b++) {
        int N_b = seq_lens ? seq_lens[b] : N;
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;
            int tid = omp_get_thread_num();
            float *p_tile  = p_buf  + (size_t)tid * TB * N;
            float *dP_tile = dP_buf + (size_t)tid * TB * N;
            float *dS_tile = dS_buf + (size_t)tid * TB * N;

            const float *q_slice = qd + (size_t)bh * N * d;
            const float *k_slice = kd + (size_t)bh * N * d;
            const float *v_slice = vd + (size_t)bh * N * d;
            const float *p_slice = Pd + (size_t)bh * N * N;
            const float *g_slice = gd + (size_t)bh * N * d;

            float *qg_slice = qg ? qg + (size_t)bh * N * d : NULL;
            float *kg_slice = kg ? kg + (size_t)bh * N * d : NULL;
            float *vg_slice = vg ? vg + (size_t)bh * N * d : NULL;

            int qk_active = (qg || kg);

            for (int r0 = 0; r0 < N_b; r0 += TB) {
                int r1 = r0 + TB;
                if (r1 > N_b) r1 = N_b;
                int M = r1 - r0;
                int S = attention_tile_S(mode, r0, r1, prefix_len);
                if (S > N_b) S = N_b;

                /* ── Pack saved P rows (stride N) to compact (stride S) ── */
                for (int mi = 0; mi < M; mi++) {
                    int i = r0 + mi;
                    memcpy(p_tile + mi * S, p_slice + (size_t)i * N,
                           (size_t)S * sizeof(float));
                }

                /* ── dV[0:S] += P_tile^T @ dO[r0:r1] ──
                 *   [S,d] += [S,M] @ [M,d] */
                if (vg) {
#if NO_CBLAS
                    for (int i = 0; i < S; i++)
                        for (int j = 0; j < d; j++) {
                            float sum = 0.0f;
                            for (int mi = 0; mi < M; mi++)
                                sum += p_tile[mi * S + i] * g_slice[(r0 + mi) * d + j];
                            vg_slice[i * d + j] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                S, d, M, 1.0f,
                                p_tile, S,
                                g_slice + r0 * d, d,
                                1.0f, vg_slice, d);
#endif
                }

                if (!qk_active) continue;

                /* ── dP_tile = dO[r0:r1] @ V[0:S]^T ──
                 *   [M,S] = [M,d] @ [S,d]^T */
#if NO_CBLAS
                for (int mi = 0; mi < M; mi++) {
                    int i = r0 + mi;
                    for (int j = 0; j < S; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < d; kk++)
                            sum += g_slice[i * d + kk] * v_slice[j * d + kk];
                        dP_tile[mi * S + j] = sum;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, S, d, 1.0f,
                            g_slice + r0 * d, d,
                            v_slice, d,
                            0.0f, dP_tile, S);
#endif

                /* ── dS_tile = softmax_bwd(P, dP) for visible prefix ── */
                for (int mi = 0; mi < M; mi++) {
                    int i = r0 + mi;
                    int visible = attention_visible_len(mode, i, prefix_len);
                    if (visible > N_b) visible = N_b;

                    float *ds_row = dS_tile + mi * S;

                    attention_softmax_bwd_row_prefix(p_tile + mi * S,
                                                     dP_tile + mi * S,
                                                     ds_row, visible, scale);

                    /* Zero tail [visible, S) — BLAS consumes full [M,S] */
                    int tail = S - visible;
                    if (tail > 0)
                        memset(ds_row + visible, 0, (size_t)tail * sizeof(float));
                }

                /* ── dQ[r0:r1] += dS_tile @ K[0:S] ──
                 *   [M,d] += [M,S] @ [S,d] */
                if (qg) {
#if NO_CBLAS
                    for (int mi = 0; mi < M; mi++) {
                        int i = r0 + mi;
                        for (int j = 0; j < d; j++) {
                            float sum = 0.0f;
                            for (int kk = 0; kk < S; kk++)
                                sum += dS_tile[mi * S + kk] * k_slice[kk * d + j];
                            qg_slice[i * d + j] += sum;
                        }
                    }
#else
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                M, d, S, 1.0f,
                                dS_tile, S,
                                k_slice, d,
                                1.0f, qg_slice + r0 * d, d);
#endif
                }

                /* ── dK[0:S] += dS_tile^T @ Q[r0:r1] ──
                 *   [S,d] += [S,M] @ [M,d] */
                if (kg) {
#if NO_CBLAS
                    for (int i = 0; i < S; i++)
                        for (int j = 0; j < d; j++) {
                            float sum = 0.0f;
                            for (int mi = 0; mi < M; mi++)
                                sum += dS_tile[mi * S + i] * q_slice[(r0 + mi) * d + j];
                            kg_slice[i * d + j] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                S, d, M, 1.0f,
                                dS_tile, S,
                                q_slice + r0 * d, d,
                                1.0f, kg_slice, d);
#endif
                }
            }
        }
    }
}


/* ══════════════════════════════════════════════════════════════════
 *  Dense reference backward
 * ══════════════════════════════════════════════════════════════════ */

#ifdef DNN_ATTENTION_DENSE_REF

static void attention_backward_dense(int B, int H, int N, int d,
                                     float scale,
                                     const float *qd, const float *kd, const float *vd,
                                     const float *Pd,
                                     const float *gd,
                                     float *qg, float *kg, float *vg,
                                     float *dS_buf,
                                     float *dP_buf) {
#pragma omp parallel for collapse(2) if (B * H >= 2)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;
            int tid = omp_get_thread_num();
            float *dS = dS_buf + (size_t)tid * N * N;
            float *dP = dP_buf + (size_t)tid * N * N;

            const float *q_slice = qd + (size_t)bh * N * d;
            const float *k_slice = kd + (size_t)bh * N * d;
            const float *v_slice = vd + (size_t)bh * N * d;
            const float *p_slice = Pd + (size_t)bh * N * N;
            const float *g_slice = gd + (size_t)bh * N * d;

            /* dV = P^T @ dO */
            if (vg) {
#if NO_CBLAS
                float *vg_slice = vg + bh * N * d;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < d; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < N; kk++)
                            sum += p_slice[kk * N + i] * g_slice[kk * d + j];
                        vg_slice[i * d + j] += sum;
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            N, d, N, 1.0f, p_slice, N, g_slice, d,
                            1.0f, vg + bh * N * d, d);
#endif
            }

            /* dP = dO @ V^T */
#if NO_CBLAS
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < d; kk++)
                        sum += g_slice[i * d + kk] * v_slice[j * d + kk];
                    dP[i * N + j] = sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N, N, d, 1.0f, g_slice, d, v_slice, d,
                        0.0f, dP, N);
#endif

            /* dS = causal_softmax_bwd(P, dP) */
            for (int i = 0; i < N; i++) {
                int visible = i + 1;
                attention_softmax_bwd_row_prefix(p_slice + i * N,
                                                 dP + i * N,
                                                 dS + i * N,
                                                 visible, scale);
                for (int j = visible; j < N; j++)
                    dS[i * N + j] = 0.0f;
            }

            /* dQ = dS @ K */
            if (qg) {
#if NO_CBLAS
                float *qg_slice = qg + bh * N * d;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < d; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < N; kk++)
                            sum += dS[i * N + kk] * k_slice[kk * d + j];
                        qg_slice[i * d + j] += sum;
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            N, d, N, 1.0f, dS, N, k_slice, d,
                            1.0f, qg + bh * N * d, d);
#endif
            }

            /* dK = dS^T @ Q */
            if (kg) {
#if NO_CBLAS
                float *kg_slice = kg + bh * N * d;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < d; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < N; kk++)
                            sum += dS[kk * N + i] * q_slice[kk * d + j];
                        kg_slice[i * d + j] += sum;
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            N, d, N, 1.0f, dS, N, q_slice, d,
                            1.0f, kg + bh * N * d, d);
#endif
            }
        }
    }
}

#endif /* DNN_ATTENTION_DENSE_REF */


/* ── Forward declaration of backward callback ── */
static void attention_backward(grad_fn *fn, tensor *grad_output);


/* ══════════════════════════════════════════════════════════════════
 *  Backward (grad_fn callback)
 * ══════════════════════════════════════════════════════════════════ */

static void attention_backward(grad_fn *fn, tensor *grad_output);

/* ══════════════════════════════════════════════════════════════════
 *  Public API: tensor_attention_ex
 * ══════════════════════════════════════════════════════════════════ */

tensor *tensor_attention_ex(struct mem_pool *scratch,
                            tensor *q, tensor *k, tensor *v,
                            tensor *mask,
                            attention_mode mode,
                            int prefix_len,
                            const int *seq_lens) {
    assert(q && k && v);
    assert(q->ndim == 4 && k->ndim == 4 && v->ndim == 4);
    assert(tensor_is_contiguous(q) && tensor_is_contiguous(k) && tensor_is_contiguous(v));

    int B = q->shape[0], H = q->shape[1], N = q->shape[2], d = q->shape[3];
    assert(k->shape[0] == B && k->shape[1] == H && k->shape[2] == N && k->shape[3] == d);
    assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == N && v->shape[3] == d);

    if (mode == ATTENTION_PREFIX_LM)
        assert(prefix_len > 0 && prefix_len <= N);
    if (mode == ATTENTION_CAUSAL)
        assert(prefix_len == 0);

    float scale = 1.0f / sqrtf((float)d);

    /* Output: always allocated */
    tensor *out = tensor_scratch(scratch, 4, (int[]){B, H, N, d}, 0);
    float *od = (float*)out->data;

    /* Determine if we need to save P for backward */
    int needs_grad = dnn_grad_enabled() &&
        (tensor_requires_grad(q) || tensor_requires_grad(k) || tensor_requires_grad(v));

    /* P: full softmax output [B,H,N,N] — only allocated when training */
    tensor *P = needs_grad ? tensor_scratch(scratch, 4, (int[]){B, H, N, N}, 0) : NULL;
    float *Pd = P ? (float*)P->data : NULL;

    float *qd = (float*)q->data;
    float *kd = (float*)k->data;
    float *vd = (float*)v->data;

    /* Per-worker tile buffers: TB × N (not N×N) */
    int n_workers = omp_get_max_threads();
    int TB = DNN_ATTENTION_TILE_ROWS;
    if (TB > N) TB = N;

    size_t tile_elems = (size_t)n_workers * (size_t)TB * (size_t)N;
    float *scores_buf = _mem_pool_alloc(scratch, tile_elems * sizeof(float), NULL);
    float *p_buf      = _mem_pool_alloc(scratch, tile_elems * sizeof(float), NULL);

#ifdef DNN_ATTENTION_DENSE_REF
    /* Dense reference uses N×N buffers */
    float *dense_scores = _mem_pool_alloc(scratch, (size_t)n_workers * N * N * sizeof(float), NULL);
    (void)scores_buf; (void)p_buf;
    attention_forward_dense(B, H, N, d, scale, qd, kd, vd, Pd, od, mask,
                            dense_scores, n_workers);
#else
    attention_forward_triangular(B, H, N, d, scale, qd, kd, vd, Pd, od, mask,
                                 scores_buf, p_buf, TB,
                                 mode, prefix_len, seq_lens);
#endif

    /* ── Autograd tape ── */
    if (needs_grad) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = attention_backward;
        fn->n_inputs = mask ? 4 : 3;
        fn->inputs = _mem_pool_alloc(scratch, (mask ? 4 : 3) * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)q;
        fn->inputs[1] = (tensor*)k;
        fn->inputs[2] = (tensor*)v;
        if (mask) fn->inputs[3] = (tensor*)mask;

        int n_saved = seq_lens ? 6 : 5;
        fn->n_saved = n_saved;
        fn->saved_tensors = _mem_pool_alloc(scratch, n_saved * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = P;

        float *scale_saved = _mem_pool_alloc(scratch, sizeof(float), NULL);
        *scale_saved = scale;
        fn->saved_tensors[1] = (tensor*)scale_saved;

        int *mask_flag = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *mask_flag = (mask != NULL) ? 1 : 0;
        fn->saved_tensors[2] = (tensor*)mask_flag;

        int *mode_saved = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *mode_saved = (int)mode;
        fn->saved_tensors[3] = (tensor*)mode_saved;

        int *prefix_len_saved = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *prefix_len_saved = prefix_len;
        fn->saved_tensors[4] = (tensor*)prefix_len_saved;

        if (seq_lens) {
            int *seq_lens_saved = _mem_pool_alloc(scratch, B * sizeof(int), NULL);
            memcpy(seq_lens_saved, seq_lens, B * sizeof(int));
            fn->saved_tensors[5] = (tensor*)seq_lens_saved;
        }

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}


/* ══════════════════════════════════════════════════════════════════
 *  Public API: tensor_attention (backward-compat wrapper)
 * ══════════════════════════════════════════════════════════════════ */

tensor *tensor_attention(struct mem_pool *scratch, tensor *q, tensor *k, tensor *v, tensor *mask) {
    return tensor_attention_ex(scratch, q, k, v, mask, ATTENTION_CAUSAL, 0, NULL);
}


/* ══════════════════════════════════════════════════════════════════
 *  Backward (grad_fn callback) — handles both causal and prefix-LM
 * ══════════════════════════════════════════════════════════════════ */

static void attention_backward(grad_fn *fn, tensor *grad_output) {
    tensor *q = fn->inputs[0];
    tensor *k = fn->inputs[1];
    tensor *v = fn->inputs[2];
    tensor *P = fn->saved_tensors[0];
    float   scale = *(float*)fn->saved_tensors[1];
    int     mask_present = *(int*)fn->saved_tensors[2];

    /* Mode, prefix_len, and seq_lens saved by tensor_attention_ex */
    attention_mode mode = ATTENTION_CAUSAL;
    int prefix_len = 0;
    const int *seq_lens = NULL;
    if (fn->n_saved >= 5) {
        mode = (attention_mode)(*(int*)fn->saved_tensors[3]);
        prefix_len = *(int*)fn->saved_tensors[4];
    }
    if (fn->n_saved >= 6) {
        seq_lens = (const int*)fn->saved_tensors[5];
    }

    int B = q->shape[0], H = q->shape[1], N = q->shape[2], d = q->shape[3];

    float *qd = (float*)q->data;
    float *kd = (float*)k->data;
    float *vd = (float*)v->data;
    float *Pd = (float*)P->data;
    float *gd = (float*)grad_output->data;

    int need_q = (q->grad_fn || q->requires_grad);
    int need_k = (k->grad_fn || k->requires_grad);
    int need_v = (v->grad_fn || v->requires_grad);

    float *qg = need_q ? _grad_ensure(q) : NULL;
    float *kg = need_k ? _grad_ensure(k) : NULL;
    float *vg = need_v ? _grad_ensure(v) : NULL;

    (void)mask_present;

    int n_workers = omp_get_max_threads();
    int TB = DNN_ATTENTION_TILE_ROWS;
    if (TB > N) TB = N;

#ifdef DNN_ATTENTION_DENSE_REF
    (void)n_workers; (void)TB;
    float *dS_buf = _mem_pool_alloc(fn->pool, (size_t)n_workers * (size_t)N * N * sizeof(float), NULL);
    float *dP_buf = _mem_pool_alloc(fn->pool, (size_t)n_workers * (size_t)N * N * sizeof(float), NULL);
    attention_backward_dense(B, H, N, d, scale, qd, kd, vd, Pd, gd,
                             qg, kg, vg, dS_buf, dP_buf);
#else
    size_t tile_elems = (size_t)n_workers * (size_t)TB * (size_t)N;
    float *p_buf  = _mem_pool_alloc(fn->pool, tile_elems * sizeof(float), NULL);
    float *dP_buf = _mem_pool_alloc(fn->pool, tile_elems * sizeof(float), NULL);
    float *dS_buf = _mem_pool_alloc(fn->pool, tile_elems * sizeof(float), NULL);

    attention_backward_triangular(B, H, N, d, scale, qd, kd, vd, Pd, gd,
                                  qg, kg, vg,
                                  p_buf, dP_buf, dS_buf, TB,
                                  mode, prefix_len, seq_lens);
#endif
}
