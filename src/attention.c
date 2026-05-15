#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
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


/* ── attention_backward ──
 *
 * Gradient for fused scaled dot-product attention with causal masking.
 *
 * Forward:
 *   S[b,h] = Q[b,h] @ K[b,h]^T * scale                [N, N]
 *   P[b,h] = causal_softmax(S[b,h])                    [N, N]
 *   O[b,h] = P[b,h] @ V[b,h]                           [N, d]
 *
 * Backward (per batch-head slice):
 *   dV  = P^T @ dO                                     [N, d]
 *   dP  = dO @ V^T                                     [N, N]
 *   dS  = causal_softmax_bwd(P, dP)                    [N, N]
 *   dQ  = (dS * scale) @ K                             [N, d]
 *   dK  = (dS^T * scale) @ Q                           [N, d]
 *
 * All tensors: Q, K, V, O shape [B, H, N, d_head]
 * Saved: P (softmax output) shape [B, H, N, N]
 * Saved: scale (float)
 * Saved: mask_present (int, 0/1)
 */

static void attention_backward(grad_fn *fn, tensor *grad_output) {
    tensor *q = fn->inputs[0];
    tensor *k = fn->inputs[1];
    tensor *v = fn->inputs[2];
    tensor *P = fn->saved_tensors[0];   /* causal softmax output [B, H, N, N] */
    float   scale = *(float*)fn->saved_tensors[1];
    int     mask_present = *(int*)fn->saved_tensors[2];

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

    /* Scratch buffers per batch-head iteration */
    float *dS = _mem_pool_alloc(fn->pool, (size_t)N * N * sizeof(float), NULL);
    float *dP = _mem_pool_alloc(fn->pool, (size_t)N * N * sizeof(float), NULL);

    (void)mask_present;  /* reserved for future mask-gradient path */

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;

            float *q_slice = qd + bh * N * d;
            float *k_slice = kd + bh * N * d;
            float *v_slice = vd + bh * N * d;
            float *p_slice = Pd + bh * N * N;
            float *g_slice = gd + bh * N * d;

            /* ── dV = P^T @ dO  [N, d] ──
             *   V_grad[b,h] += P[b,h]^T @ dO[b,h]
             *   P^T is [N,N], dO is [N,d]
             *   sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, d, N, 1, P, N, dO, d, 1, V_grad, d)
             */
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

            /* ── dP = dO @ V^T  [N, N] ──
             *   dP[b,h] = dO[b,h] @ V[b,h]^T
             *   dO is [N,d], V^T is [d,N]
             *   sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, d, 1, dO, d, V, d, 0, dP, N)
             */
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

            /* ── dS = causal_softmax_bwd(P, dP)  [N, N] ──
             *
             *   For each row i:
             *     dot_i = sum_{j <= i} P[i][j] * dP[i][j]
             *     for j <= i: dS[i][j] = P[i][j] * (dP[i][j] - dot_i)
             *     for j > i:  dS[i][j] = 0
             */
            for (int i = 0; i < N; i++) {
                float *p_row = p_slice + i * N;
                float *dp_row = dP + i * N;
                float *ds_row = dS + i * N;

                /* dot_i = sum_{j <= i} P[i][j] * dP[i][j] */
                float dot;
#if DNN_HAVE_NEON
                {
                    float32x4_t vdot = vdupq_n_f32(0.0f);
                    int j = 0;
                    for (; j + 4 <= i; j += 4) {
                        vdot = vfmaq_f32(vdot, vld1q_f32(p_row + j),
                                               vld1q_f32(dp_row + j));
                    }
                    dot = vaddvq_f32(vdot);
                    for (; j <= i; j++) dot += p_row[j] * dp_row[j];
                }
#else
                dot = 0.0f;
                for (int j = 0; j <= i; j++)
                    dot += p_row[j] * dp_row[j];
#endif

                /* dS[i][j] = P[i][j] * (dP[i][j] - dot_i) * scale for j <= i
                 *   scale baked in here to avoid a separate N×N pass */
                for (int j = 0; j <= i; j++)
                    ds_row[j] = p_row[j] * (dp_row[j] - dot) * scale;
                for (int j = i + 1; j < N; j++)
                    ds_row[j] = 0.0f;
            }

            /* ── dQ = dS @ K  [N, d] ──
             *   dS is [N,N], K is [N,d]
             *   sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, d, N, 1, dS, N, K, d, 1, Q_grad, d)
             */
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

            /* ── dK = dS^T @ Q  [N, d] ──
             *   dS^T is [N,N], Q is [N,d]
             *   sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, d, N, 1, dS, N, Q, d, 1, K_grad, d)
             */
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


/* ── tensor_attention ──
 *
 * Fused scaled dot-product attention with causal masking.
 *
 *   Q, K, V — input tensors, shape [B, H, N, d_head] (4D contiguous)
 *   mask    — additive mask, shape broadcastable to [B, H, N, N]
 *             may be NULL (no additive mask, but causal mask always applied)
 *
 *   Returns O = causal_softmax( (Q @ K^T) / sqrt(d_head) + mask ) @ V
 *
 *   All inputs must be contiguous and 4D.
 *   Causal masking is implicit (never materializes the upper-triangular mask).
 *   Autograd wired: saves P (causal softmax output) and scale for backward.
 */

tensor *tensor_attention(struct mem_pool *scratch, tensor *q, tensor *k, tensor *v, tensor *mask) {
    assert(q && k && v);
    assert(q->ndim == 4 && k->ndim == 4 && v->ndim == 4);
    assert(tensor_is_contiguous(q) && tensor_is_contiguous(k) && tensor_is_contiguous(v));

    int B = q->shape[0], H = q->shape[1], N = q->shape[2], d = q->shape[3];
    assert(k->shape[0] == B && k->shape[1] == H && k->shape[2] == N && k->shape[3] == d);
    assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == N && v->shape[3] == d);

    float scale = 1.0f / sqrtf((float)d);

    /* Output: [B, H, N, d] */
    tensor *out = tensor_scratch(scratch, 4, (int[]){B, H, N, d}, 0);
    float *od = (float*)out->data;

    /* Allocate P (causal softmax output) in scratch — saved for backward */
    tensor *P = tensor_scratch(scratch, 4, (int[]){B, H, N, N}, 0);
    float *Pd = (float*)P->data;

    float *qd = (float*)q->data;
    float *kd = (float*)k->data;
    float *vd = (float*)v->data;

    /* Temp scores buffer: [N, N] reused per batch-head */
    float *scores = _mem_pool_alloc(scratch, (size_t)N * N * sizeof(float), NULL);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;

            float *q_slice = qd + bh * N * d;
            float *k_slice = kd + bh * N * d;
            float *v_slice = vd + bh * N * d;
            float *p_slice = Pd + bh * N * N;
            float *o_slice = od + bh * N * d;

            /* ── Step 1: scores = Q @ K^T * scale  [N, N] ──
             *   Q is [N,d], K is [N,d], result is [N,N]
             *   sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, d, scale, Q, d, K, d, 0, scores, N)
             */
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

            /* ── Step 2: Add mask if provided ── */
            if (mask) {
                float *md = (float*)mask->data;
                int mask_ndim = mask->ndim;
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        int coord[4] = {b, h, i, j};
                        int m_off = _bcast_off(mask, mask_ndim, coord);
                        scores[i * N + j] += md[m_off];
                    }
                }
            }

            /* ── Step 3: Causal softmax in-place on scores → P ──
             *
             *   For each row i:
             *     mx = max_{j <= i} scores[i][j]
             *     se = sum_{j <= i} exp(scores[i][j] - mx)
             *     P[i][j] = exp(scores[i][j] - mx) / se  for j <= i
             *     P[i][j] = 0                            for j > i
             *
             *   Fused online max + sum_exp (1 pass) + NEON SIMD.
             */
            for (int i = 0; i < N; i++) {
                float *row = scores + i * N;
                float *p_row = p_slice + i * N;

                /* ── Fused pass: online max + sum_exp ── */
                float mx = -INFINITY;
                float se = 0.0f;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    for (; j + 4 <= i + 1; j += 4) {
                        float32x4_t v = vld1q_f32(row + j);
                        float group_max = vmaxvq_f32(v);
                        if (group_max > mx) {
                            se *= expf(mx - group_max);
                            mx = group_max;
                        }
                        float32x4_t shifted = vsubq_f32(v, vdupq_n_f32(mx));
                        se += vaddvq_f32(simd_expf_f32(shifted));
                    }
                    for (; j <= i; j++) {
                        float old_mx = mx;
                        if (row[j] > mx) mx = row[j];
                        if (mx != old_mx) se *= expf(old_mx - mx);
                        se += expf(row[j] - mx);
                    }
                }
#else
                for (int j = 0; j <= i; j++) {
                    float old_mx = mx;
                    if (row[j] > mx) mx = row[j];
                    if (mx != old_mx) se *= expf(old_mx - mx);
                    se += expf(row[j] - mx);
                }
#endif

                /* ── Write softmax weights (1 pass) ── */
                float inv_se = 1.0f / se;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    float32x4_t vmx = vdupq_n_f32(mx);
                    float32x4_t vinv_se = vdupq_n_f32(inv_se);
                    for (; j + 4 <= i + 1; j += 4) {
                        float32x4_t v = vld1q_f32(row + j);
                        float32x4_t exp_v = simd_expf_f32(vsubq_f32(v, vmx));
                        vst1q_f32(p_row + j, vmulq_f32(exp_v, vinv_se));
                    }
                    for (; j <= i; j++)
                        p_row[j] = expf(row[j] - mx) * inv_se;
                }
#else
                for (int j = 0; j <= i; j++)
                    p_row[j] = expf(row[j] - mx) * inv_se;
#endif
                for (int j = i + 1; j < N; j++)
                    p_row[j] = 0.0f;
            }

            /* ── Step 4: O = P @ V  [N, d] ──
             *   P is [N,N], V is [N,d]
             *   sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, d, N, 1, P, N, V, d, 0, O, d)
             */
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

    /* ── Autograd tape ── */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(q) || tensor_requires_grad(k) || tensor_requires_grad(v))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = attention_backward;
        fn->n_inputs = mask ? 4 : 3;
        fn->inputs = _mem_pool_alloc(scratch, (mask ? 4 : 3) * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)q;
        fn->inputs[1] = (tensor*)k;
        fn->inputs[2] = (tensor*)v;
        if (mask) fn->inputs[3] = (tensor*)mask;

        fn->n_saved = 3;
        fn->saved_tensors = _mem_pool_alloc(scratch, 3 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = P;

        float *scale_saved = _mem_pool_alloc(scratch, sizeof(float), NULL);
        *scale_saved = scale;
        fn->saved_tensors[1] = (tensor*)scale_saved;

        int *mask_flag = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *mask_flag = (mask != NULL) ? 1 : 0;
        fn->saved_tensors[2] = (tensor*)mask_flag;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
