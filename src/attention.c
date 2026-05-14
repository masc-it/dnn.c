#include "attention.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "simd.h"
#include <assert.h>
#include <string.h>
#include <math.h>

#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#elif defined(HAVE_CBLAS)
#  include <cblas.h>
#else
#  define NO_CBLAS 1
#endif

/* ── Internal helpers ── */

/* Flatten all dimensions except last 2 into a single batch dim.
 * Returns number of batch elements.  Sets *B, *N, *d. */
static int _flatten_batch(const tensor *t, int *B, int *N, int *d) {
    int ndim = t->ndim;
    *d = t->shape[ndim - 1];
    *N = t->shape[ndim - 2];
    *B = 1;
    for (int i = 0; i < ndim - 2; i++)
        *B *= t->shape[i];
    return *B;
}

/* ── attention_backward ── */

static void attention_backward(grad_fn *fn, tensor *grad_output) {
    tensor *Q     = fn->inputs[0];
    tensor *K     = fn->inputs[1];
    tensor *V     = fn->inputs[2];
    tensor *saved_attn = fn->saved_tensors[0];  /* softmax output [B, H, N, N] */
    (void)fn->saved_tensors[1];  /* mask (unused in backward) */
    (void)fn->saved_tensors[2];  /* scale (unused — baked into forward pass) */

    int ndim = Q->ndim;
    int d_k = Q->shape[ndim - 1];
    int N   = Q->shape[ndim - 2];

    /* flatten batch dims */
    int B, _N, _d;
    _flatten_batch(Q, &B, &_N, &_d);
    (void)_N; (void)_d;

    /* pointers */
    float *qd   = (float*)Q->data + Q->offset;
    float *kd   = (float*)K->data + K->offset;
    float *vd   = (float*)V->data + V->offset;
    float *gd   = (float*)grad_output->data;
    float *attn = (float*)saved_attn->data + saved_attn->offset;

    /* strides: for flattened [B, N, d] everything is contiguous */
    int stride_q = N * d_k;   /* stride between batch items in Q, K, V, gd */
    int stride_a = N * N;     /* stride between batch items in attn */

    /* ── backward: d_attn = d_output @ V^T ──
     *   attn_grad[bi, n, s] = sum over j: gd[bi, n, j] * vd[bi, s, j]
     *   → (B, N, N) = (B, N, d) @ (B, d, N) via transpose of V
     *
     * ── dV = attn^T @ d_output ──
     *   dV[bi, s, j] = sum over n: attn[bi, n, s] * gd[bi, n, j]
     *   → (B, d, N) = (B, N, N)^T @ (B, N, d)
     *                = (B, N, N) @ ... via transpose of attn
     *
     * For each head in batch, we compute:
     *   attn_grad[b*stride_a] = gd[b*stride_q] @ V[b*stride_q]^T
     *   dV[b*stride_q]        = attn[b*stride_a]^T @ gd[b*stride_q]
     */

    /* allocate temp: attn_grad (N, N) per head */
    float *scores_grad_buf = mem_scratch_alloc((size_t)B * N * N * sizeof(float), NULL);

    for (int b = 0; b < B; b++) {
        float *q_b  = qd  + b * stride_q;
        float *k_b  = kd  + b * stride_q;
        float *v_b  = vd  + b * stride_q;
        float *g_b  = gd  + b * stride_q;
        float *a_b  = attn + b * stride_a;
        float *sg_b = scores_grad_buf + b * stride_a;

        /* ── 1. d_attn = d_output @ V^T  → (N, N) = (N, d) @ (d, N) ── */
#if NO_CBLAS
        for (int n = 0; n < N; n++)
            for (int s = 0; s < N; s++) {
                float sum = 0.0f;
                for (int j = 0; j < d_k; j++)
                    sum += g_b[n * d_k + j] * v_b[s * d_k + j];
                sg_b[n * N + s] = sum;
            }
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, N, d_k, 1.0f, g_b, d_k, v_b, d_k,
                    0.0f, sg_b, N);
#endif

        /* ── 2. Apply softmax backward to get d_scores ──
         *   d_scores[i,j] = attn[i,j] * (d_attn[i,j] - sum_k(attn[i,k] * d_attn[i,k]))
         *
         * Compute per-row dot product, then apply formula.
         */
        for (int n = 0; n < N; n++) {
            float *a_row = a_b + n * N;
            float *sg_row = sg_b + n * N;

            /* dot = sum over s: attn[n,s] * d_attn[n,s] */
            float dot = 0.0f;
#if DNN_HAVE_NEON
            float32x4_t vdot = vdupq_n_f32(0.0f);
            int s = 0;
            for (; s + 4 <= N; s += 4)
                vdot = vfmaq_f32(vdot, vld1q_f32(a_row + s),
                                          vld1q_f32(sg_row + s));
            dot = vaddvq_f32(vdot);
            for (; s < N; s++) dot += a_row[s] * sg_row[s];
#else
            for (int s = 0; s < N; s++) dot += a_row[s] * sg_row[s];
#endif

            /* d_scores[n,s] = attn[n,s] * (d_attn[n,s] - dot) */
#if DNN_HAVE_NEON
            float32x4_t vdot_vec = vdupq_n_f32(dot);
            s = 0;
            for (; s + 4 <= N; s += 4) {
                float32x4_t va   = vld1q_f32(a_row + s);
                float32x4_t vsg  = vld1q_f32(sg_row + s);
                vst1q_f32(sg_row + s, vmulq_f32(va, vsubq_f32(vsg, vdot_vec)));
            }
            for (; s < N; s++)
                sg_row[s] = a_row[s] * (sg_row[s] - dot);
#else
            for (int s = 0; s < N; s++)
                sg_row[s] = a_row[s] * (sg_row[s] - dot);
#endif
        }

        /* ── 3. Apply mask gradient (zero out masked positions) ──
         * Mask has grad 0 (mask is constant, requires_grad=0).
         * But we need to ensure future positions don't get gradient flow.
         * If mask[i,j] == -inf, then softmax output there is 0,
         * and the softmax backward already handles it correctly.
         * No extra work needed — attn is already 0 at masked positions.
         */

        /* ── 4. dQ = d_scores @ K  → (N, d) = (N, N) @ (N, d) ── */
        if (Q->grad_fn || Q->requires_grad) {
            float *qg = _grad_ensure(Q);
            float *qg_b = qg + b * stride_q;
#if NO_CBLAS
            for (int n = 0; n < N; n++)
                for (int j = 0; j < d_k; j++) {
                    float sum = 0.0f;
                    for (int s = 0; s < N; s++)
                        sum += sg_b[n * N + s] * k_b[s * d_k + j];
                    qg_b[n * d_k + j] += sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, d_k, N, 1.0f, sg_b, N, k_b, d_k,
                        1.0f, qg_b, d_k);
#endif
        }

        /* ── 5. dK = d_scores^T @ Q  → (N, d) = (N, N) @ ... ──
         *   Actually: dK = (d_scores)^T @ Q  (since scores = Q @ K^T)
         *   dK[s,j] = sum_n d_scores[n,s] * Q[n,j]
         *   → (N, d) = (N, N)^T @ (N, d) = d_scores^T @ Q
         */
        if (K->grad_fn || K->requires_grad) {
            float *kg = _grad_ensure(K);
            float *kg_b = kg + b * stride_q;
#if NO_CBLAS
            for (int s = 0; s < N; s++)
                for (int j = 0; j < d_k; j++) {
                    float sum = 0.0f;
                    for (int n = 0; n < N; n++)
                        sum += sg_b[n * N + s] * q_b[n * d_k + j];
                    kg_b[s * d_k + j] += sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        N, d_k, N, 1.0f, sg_b, N, q_b, d_k,
                        1.0f, kg_b, d_k);
#endif
        }

        /* ── 6. dV = attn^T @ d_output  → (N, d) = (N, N)^T @ (N, d) ── */
        if (V->grad_fn || V->requires_grad) {
            float *vg = _grad_ensure(V);
            float *vg_b = vg + b * stride_q;
#if NO_CBLAS
            for (int s = 0; s < N; s++)
                for (int j = 0; j < d_k; j++) {
                    float sum = 0.0f;
                    for (int n = 0; n < N; n++)
                        sum += a_b[n * N + s] * g_b[n * d_k + j];
                    vg_b[s * d_k + j] += sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        N, d_k, N, 1.0f, a_b, N, g_b, d_k,
                        1.0f, vg_b, d_k);
#endif
        }
    }
}

/* ── tensor_attention forward ── */

tensor *tensor_attention(tensor *Q, tensor *K, tensor *V,
                         tensor *mask) {
    assert(Q && K && V);
    assert(Q->ndim == K->ndim && K->ndim == V->ndim);
    assert(tensor_is_contiguous(Q) && "attention: Q must be contiguous");
    assert(tensor_is_contiguous(K) && "attention: K must be contiguous");
    assert(tensor_is_contiguous(V) && "attention: V must be contiguous");

    int ndim = Q->ndim;
    assert(ndim >= 2 && ndim <= 4 && "attention: ndim must be 2-4");

    int d_k = Q->shape[ndim - 1];
    int N   = Q->shape[ndim - 2];

    assert(K->shape[ndim - 1] == d_k && "attention: K last dim must match Q");
    assert(V->shape[ndim - 1] == d_k && "attention: V last dim must match Q");
    assert(K->shape[ndim - 2] == N && "attention: K seq len must match Q");
    assert(V->shape[ndim - 2] == N && "attention: V seq len must match Q");

    /* mask shape check: mask must be [N, N] or NULL */
    if (mask) {
        assert(mask->ndim == 2 && "attention: mask must be 2D [N, N]");
        assert(mask->shape[0] == N && mask->shape[1] == N &&
               "attention: mask shape must be [N, N]");
    }

    float scale = 1.0f / sqrtf((float)d_k);

    /* flatten batch dims */
    int B, _N, _d;
    _flatten_batch(Q, &B, &_N, &_d);

    /* allocate output: same shape as Q */
    tensor *out = _tensor_scratch_create(ndim, Q->shape, 0);
    float  *od  = (float*)out->data + out->offset;

    /* allocate attn buffer: [B, N, N] for softmax output */
    tensor *attn_t = _tensor_scratch_create(3, (int[]){B, N, N}, 0);
    float  *attn   = (float*)attn_t->data + attn_t->offset;

    /* pointers */
    float *qd = (float*)Q->data + Q->offset;
    float *kd = (float*)K->data + K->offset;
    float *vd = (float*)V->data + V->offset;
    float *md = mask ? ((float*)mask->data + mask->offset) : NULL;

    int stride_q = N * d_k;  /* bytes between batch items in Q, K, V, out */
    int stride_a = N * N;    /* bytes between batch items in attn */
    int stride_s = N * N;    /* same for scores */
    int stride_m = N;        /* row stride for mask */

    /* temp buffer for scores before softmax */
    float *scores = mem_scratch_alloc((size_t)B * N * N * sizeof(float), NULL);

    for (int b = 0; b < B; b++) {
        float *q_b  = qd + b * stride_q;
        float *k_b  = kd + b * stride_q;
        float *v_b  = vd + b * stride_q;
        float *o_b  = od + b * stride_q;
        float *s_b  = scores + b * stride_s;
        float *a_b  = attn + b * stride_a;

        /* ── scores[b] = Q[b] @ K[b]^T ──
         *   (N, N) = (N, d) @ (d, N)
         */
#if NO_CBLAS
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < d_k; k++)
                    sum += q_b[i * d_k + k] * k_b[j * d_k + k];
                s_b[i * N + j] = sum * scale;
            }
#else
        /* sgemm computes C = alpha * A * B + beta * C
         * We want: scores = Q @ K^T  (both row-major → use CblasTrans on K)
         * Q: [N, d_k], K: [N, d_k] → K^T: [d_k, N] in BLAS terms
         * sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, d_k, ...)
         */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, N, d_k, scale, q_b, d_k, k_b, d_k,
                    0.0f, s_b, N);
#endif

        /* ── Add mask ── */
        // TODO: check for optim room
        if (md) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    s_b[i * N + j] += md[i * stride_m + j];
                }
            }
        }

        /* ── Softmax over last dim ──
         *   For each row n: softmax over s (sequence positions)
         */
        for (int n = 0; n < N; n++) {
            float *row = s_b + n * N;

            /* find max */
            float mx = -INFINITY;
            for (int s = 0; s < N; s++)
                if (row[s] > mx) mx = row[s];

            /* sum of exp(x - max) */
            float se = 0.0f;
            for (int s = 0; s < N; s++)
                se += expf(row[s] - mx);

            /* softmax = exp(x - max) / sum */
            float inv_se = 1.0f / se;
            for (int s = 0; s < N; s++)
                a_b[n * N + s] = expf(row[s] - mx) * inv_se;
        }

        /* ── output[b] = attn[b] @ V[b] ──
         *   (N, d) = (N, N) @ (N, d)
         */
#if NO_CBLAS
        for (int i = 0; i < N; i++)
            for (int j = 0; j < d_k; j++) {
                float sum = 0.0f;
                for (int s = 0; s < N; s++)
                    sum += a_b[i * N + s] * v_b[s * d_k + j];
                o_b[i * d_k + j] = sum;
            }
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, d_k, N, 1.0f, a_b, N, v_b, d_k,
                    0.0f, o_b, d_k);
#endif
    }

    /* ── Autograd tape ──
     *   Attention needs grad only if at least one of Q, K, V requires grad.
     *   Mask is not differentiable (constant).
     */
    int needs_grad = dnn_grad_enabled() &&
        (tensor_requires_grad(Q) || tensor_requires_grad(K) || tensor_requires_grad(V));

    if (needs_grad) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = attention_backward;
        fn->n_inputs = 3;
        fn->inputs = mem_scratch_alloc(3 * sizeof(tensor*), NULL);
        fn->inputs[0] = Q;
        fn->inputs[1] = K;
        fn->inputs[2] = V;

        fn->n_saved = 3;
        fn->saved_tensors = mem_scratch_alloc(3 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = attn_t;   /* softmax output, needed for backward */
        fn->saved_tensors[1] = mask ? (tensor*)mask : NULL;

        float *scale_saved = mem_scratch_alloc(sizeof(float), NULL);
        *scale_saved = scale;
        fn->saved_tensors[2] = (tensor*)scale_saved;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
