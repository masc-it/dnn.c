#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_matrix_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Minimum batch iterations before using OpenMP (avoids thread overhead on tiny batches) */
#define OMP_MIN_ITERS 8

/* BLAS for fast matmul */
#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#elif defined(HAVE_CBLAS)
#  include <cblas.h>
#else
#  define NO_CBLAS 1
#endif


/* ── Internal: compute batch offset for matmul ──
 *
 * For tensor t with ndim >= 2, batch dims are t->shape[0..t->ndim-3].
 * Given a coord in the broadcast batch space (ndim_batch), compute the
 * data offset for that batch element.
 * Dims where shape[d]==1 are broadcast (always use coord=0).
 */
static int _matmul_batch_off(const tensor *t, int ndim_batch, const int *coord) {
    int off = t->offset;
    int lead = ndim_batch - (t->ndim - 2);
    for (int d = 0; d < t->ndim - 2; d++) {
        int c = t->shape[d] == 1 ? 0 : coord[lead + d];
        off += c * t->strides[d];
    }
    return off;
}


/* ── matmul_backward ── */

static void matmul_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    int na = a->ndim, nb = b->ndim;
    int K = a->shape[na - 1];
    int M = a->shape[na - 2];
    int N = b->shape[nb - 1];

    /* ── 2D case — existing optimized path ── */
    if (na == 2 && nb == 2) {
        float *ad = (float*)a->data;
        float *bd = (float*)b->data;
        float *gd = (float*)grad_output->data;

        int a_s0 = a->strides[0], a_s1 = a->strides[1];
        int b_s0 = b->strides[0], b_s1 = b->strides[1];
        int g_s0 = grad_output->strides[0];
        int a_off = a->offset, b_off = b->offset;

        int a_contig = tensor_is_contiguous(a);
        int b_contig = tensor_is_contiguous(b);
        int g_contig = tensor_is_contiguous(grad_output);

        /* da = gd @ B^T  → (M, K) */
        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
#if NO_CBLAS
            for (int i = 0; i < M; i++)
                for (int k = 0; k < K; k++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++)
                        sum += gd[i * g_s0 + j] * bd[b_off + k * b_s0 + j * b_s1];
                    ag[a_off + i * a_s0 + k * a_s1] += sum;
                }
#else
            if (a_contig && b_contig && g_contig) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, K, N, 1.0f, gd, g_s0, bd + b_off, b_s0,
                            1.0f, ag + a_off, a_s0);
            } else {
                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++) {
                        float sum = 0.0f;
                        for (int j = 0; j < N; j++)
                            sum += gd[i * g_s0 + j] * bd[b_off + k * b_s0 + j * b_s1];
                        ag[a_off + i * a_s0 + k * a_s1] += sum;
                    }
            }
#endif
        }

        /* db = A^T @ gd  → (K, N) */
        if ((b->grad_fn || b->requires_grad) && b != a) {
            float *bg = _grad_ensure(b);
#if NO_CBLAS
            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < M; i++)
                        sum += ad[a_off + i * a_s0 + k * a_s1] * gd[i * g_s0 + j];
                    bg[b_off + k * b_s0 + j * b_s1] += sum;
                }
#else
            if (a_contig && b_contig && g_contig) {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            K, N, M, 1.0f, ad + a_off, a_s0, gd, g_s0,
                            1.0f, bg + b_off, b_s0);
            } else {
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; i++)
                            sum += ad[a_off + i * a_s0 + k * a_s1] * gd[i * g_s0 + j];
                        bg[b_off + k * b_s0 + j * b_s1] += sum;
                    }
            }
#endif
        }

        /* a == b: second term da += A^T @ gd  (gd @ A^T already done in first branch) */
        if (a == b && (a->grad_fn || a->requires_grad)) {
            float *ag = _grad_ensure(a);
#if NO_CBLAS
            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < M; i++)
                        sum += ad[a_off + i * a_s0 + k * a_s1] * gd[i * g_s0 + j];
                    ag[a_off + k * a_s0 + j * a_s1] += sum;
                }
#else
            if (a_contig && g_contig) {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            K, N, M, 1.0f, ad + a_off, a_s0, gd, g_s0,
                            1.0f, ag + a_off, a_s0);
            } else {
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; i++)
                            sum += ad[a_off + i * a_s0 + k * a_s1] * gd[i * g_s0 + j];
                        ag[a_off + k * a_s0 + j * a_s1] += sum;
                    }
            }
#endif
        }
        return;
    }

    /* ── Batched case (ND) ── */
    /* Broadcast batch dims (all except last 2) */
    int a_batch_ndim = na - 2;
    int b_batch_ndim = nb - 2;
    int batch_ndim;
    int batch_shape[8];
    batch_ndim = _bcast_ndim(a_batch_ndim, a->shape, b_batch_ndim, b->shape, batch_shape);
    int batch_size = _numel(batch_ndim, batch_shape);

    int out_ndim = batch_ndim + 2;
    int need_a = (a->grad_fn || a->requires_grad);
    int need_b = (b->grad_fn || b->requires_grad);
    int a_self = (a == b);
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;
    float *gd = (float*)grad_output->data;
    float *ag = need_a ? _grad_ensure(a) : NULL;
    float *bg = need_b ? _grad_ensure(b) : NULL;

    int a_row_stride = a->strides[na - 2];   /* lda for a and da */
    int b_row_stride = b->strides[nb - 2];   /* ldb for b and db */
    int g_row_stride = grad_output->strides[out_ndim - 2]; /* ld for gd */
    int a_col_stride = a->strides[na - 1];
    int b_col_stride = b->strides[nb - 1];
    int g_col_stride = grad_output->strides[out_ndim - 1];
    int slice_contig = (a_col_stride == 1 && b_col_stride == 1 && g_col_stride == 1);

    /* Check if broadcasting is needed in backward: if yes, skip OMP (grad accumulation races) */
    int no_broadcast = 1;
    int lead_a = batch_ndim - a_batch_ndim;
    int lead_b = batch_ndim - b_batch_ndim;
    for (int d = 0; d < batch_ndim; d++) {
        int a_sz = (d >= lead_a && d - lead_a < a_batch_ndim) ? a->shape[d - lead_a] : 0;
        int b_sz = (d >= lead_b && d - lead_b < b_batch_ndim) ? b->shape[d - lead_b] : 0;
        if ((a_sz > 0 && a_sz != batch_shape[d]) || (b_sz > 0 && b_sz != batch_shape[d])) {
            no_broadcast = 0;
            break;
        }
    }

/* OMP parallelize when many small-to-medium slices (batch >= 8, each slice 1M-50M flops).
 * For tiny slices (<1M flops) OMP overhead dominates.
 * For large slices (>50M flops) BLAS threading already saturates CPU — outer OMP oversubscribes.
 */
#pragma omp parallel for if (no_broadcast && batch_size >= OMP_MIN_ITERS \
    && (long long)M * K * N >= 1000000 && (long long)M * K * N <= 50000000)
    for (int bi = 0; bi < batch_size; bi++) {
        /* Flat batch index → batch coordinate (local, OMP-private) */
        int local_coord[8];
        int tmp = bi;
        for (int d = batch_ndim - 1; d >= 0; d--) {
            local_coord[d] = tmp % batch_shape[d];
            tmp /= batch_shape[d];
        }

        int a_off = _matmul_batch_off(a, batch_ndim, local_coord);
        int b_off = _matmul_batch_off(b, batch_ndim, local_coord);
        int g_off = _matmul_batch_off(grad_output, batch_ndim, local_coord);

        /* da += gd @ B^T  → (M, K) */
        if (need_a) {
            if (slice_contig) {
#if NO_CBLAS
                for (int i = 0; i < M; i++)
                    for (int kk = 0; kk < K; kk++) {
                        float sum = 0.0f;
                        for (int j = 0; j < N; j++)
                            sum += gd[g_off + i * g_row_stride + j]
                                 * bd[b_off + kk * b_row_stride + j];
                        ag[a_off + i * a_row_stride + kk] += sum;
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, K, N, 1.0f, gd + g_off, g_row_stride,
                            bd + b_off, b_row_stride,
                            1.0f, ag + a_off, a_row_stride);
#endif
            } else {
                for (int i = 0; i < M; i++)
                    for (int kk = 0; kk < K; kk++) {
                        float sum = 0.0f;
                        for (int j = 0; j < N; j++)
                            sum += gd[g_off + i * g_row_stride + j * g_col_stride]
                                 * bd[b_off + kk * b_row_stride + j * b_col_stride];
                        ag[a_off + i * a_row_stride + kk * a_col_stride] += sum;
                    }
            }
        }

        /* db += A^T @ gd  → (K, N)  (skip if a==b, handled below) */
        if (need_b && !a_self) {
            if (slice_contig) {
#if NO_CBLAS
                for (int kk = 0; kk < K; kk++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; i++)
                            sum += ad[a_off + i * a_row_stride + kk]
                                 * gd[g_off + i * g_row_stride + j];
                        bg[b_off + kk * b_row_stride + j] += sum;
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            K, N, M, 1.0f, ad + a_off, a_row_stride,
                            gd + g_off, g_row_stride,
                            1.0f, bg + b_off, b_row_stride);
#endif
            } else {
                for (int kk = 0; kk < K; kk++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; i++)
                            sum += ad[a_off + i * a_row_stride + kk * a_col_stride]
                                 * gd[g_off + i * g_row_stride + j * g_col_stride];
                        bg[b_off + kk * b_row_stride + j * b_col_stride] += sum;
                    }
            }
        }

        /* a == b: second da term = A^T @ gd */
        if (a_self && need_a) {
            if (slice_contig) {
#if NO_CBLAS
                for (int kk = 0; kk < K; kk++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; i++)
                            sum += ad[a_off + i * a_row_stride + kk]
                                 * gd[g_off + i * g_row_stride + j];
                        ag[a_off + kk * a_row_stride + j] += sum;
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            K, N, M, 1.0f, ad + a_off, a_row_stride,
                            gd + g_off, g_row_stride,
                            1.0f, ag + a_off, a_row_stride);
#endif
            } else {
                for (int kk = 0; kk < K; kk++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; i++)
                            sum += ad[a_off + i * a_row_stride + kk * a_col_stride]
                                 * gd[g_off + i * g_row_stride + j * g_col_stride];
                        ag[a_off + kk * a_row_stride + j * a_col_stride] += sum;
                    }
            }
        }
    }
}


tensor *tensor_matmul(struct mem_pool *scratch, const tensor *a, const tensor *b) {
    assert(a && b);
    assert(a->ndim >= 2 && b->ndim >= 2 && "tensor_matmul: need at least 2D");

    int na = a->ndim, nb = b->ndim;
    int K = a->shape[na - 1];
    int M = a->shape[na - 2];
    int N = b->shape[nb - 1];
    assert(K == b->shape[nb - 2] && "tensor_matmul: inner dim mismatch");

    /* ── 2D case — existing optimized path ── */
    if (na == 2 && nb == 2) {
        tensor *out = tensor_scratch(scratch, 2, (int[]){M, N}, 0);
        float *od = (float*)out->data;
        float *ad = (float*)a->data;
        float *bd = (float*)b->data;

        int a_s0 = a->strides[0], a_s1 = a->strides[1];
        int b_s0 = b->strides[0], b_s1 = b->strides[1];
        int a_off = a->offset, b_off = b->offset;

#if NO_CBLAS
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += ad[a_off + i * a_s0 + k * a_s1]
                         * bd[b_off + k * b_s0 + j * b_s1];
                od[out->offset + i * N + j] = sum;
            }
#else
        if (tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, ad + a_off, a_s0,
                        bd + b_off, b_s0,
                        0.0f, od, N);
        } else {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                        sum += ad[a_off + i * a_s0 + k * a_s1]
                             * bd[b_off + k * b_s0 + j * b_s1];
                    od[out->offset + i * N + j] = sum;
                }
        }
#endif

        /* autograd tape */
        if (dnn_grad_enabled() &&
            (tensor_requires_grad(a) || tensor_requires_grad(b))) {
            grad_fn *fn = _grad_fn_create(scratch);
            fn->backward = matmul_backward;
            fn->n_inputs = 2;
            fn->inputs = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
            fn->inputs[0] = (tensor*)a;
            fn->inputs[1] = (tensor*)b;
            fn->n_saved = 0;
            out->requires_grad = 1;
            out->grad_fn = fn;
        }
        return out;
    }

    /* ── Batched case (ND) ── */
    /* Broadcast batch dims (all except last 2 of each) */
    int a_batch_ndim = na - 2;
    int b_batch_ndim = nb - 2;
    int batch_ndim;
    int batch_shape[8];
    batch_ndim = _bcast_ndim(a_batch_ndim, a->shape, b_batch_ndim, b->shape, batch_shape);
    int batch_size = _numel(batch_ndim, batch_shape);

    /* Output shape = [batch..., M, N] */
    int out_ndim = batch_ndim + 2;
    int out_shape[8];
    for (int i = 0; i < batch_ndim; i++) out_shape[i] = batch_shape[i];
    out_shape[batch_ndim] = M;
    out_shape[batch_ndim + 1] = N;

    tensor *out = tensor_scratch(scratch, out_ndim, out_shape, 0);
    float *od = (float*)out->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    int a_row_stride = a->strides[na - 2];   /* lda for a slice */
    int b_row_stride = b->strides[nb - 2];   /* ldb for b slice */
    int a_col_stride = a->strides[na - 1];   /* stride along last dim */
    int b_col_stride = b->strides[nb - 1];
    int slice_contig = (a_col_stride == 1 && b_col_stride == 1);

#pragma omp parallel for if (batch_size >= OMP_MIN_ITERS \
    && (long long)M * K * N >= 1000000 && (long long)M * K * N <= 50000000)
    for (int bi = 0; bi < batch_size; bi++) {
        /* Flat batch index → batch coordinate (local, OMP-private) */
        int local_coord[8];
        int tmp = bi;
        for (int d = batch_ndim - 1; d >= 0; d--) {
            local_coord[d] = tmp % batch_shape[d];
            tmp /= batch_shape[d];
        }

        int a_off = _matmul_batch_off(a, batch_ndim, local_coord);
        int b_off = _matmul_batch_off(b, batch_ndim, local_coord);

        if (slice_contig) {
            /* Contiguous slice fast path — sgemm or simple triple loop */
#if NO_CBLAS
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                        sum += ad[a_off + i * a_row_stride + k]
                             * bd[b_off + k * b_row_stride + j];
                    od[bi * M * N + i * N + j] = sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f,
                        ad + a_off, a_row_stride,
                        bd + b_off, b_row_stride,
                        0.0f, od + bi * M * N, N);
#endif
        } else {
            /* Non-contiguous slice — full stride-based access */
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                        sum += ad[a_off + i * a_row_stride + k * a_col_stride]
                             * bd[b_off + k * b_row_stride + j * b_col_stride];
                    od[bi * M * N + i * N + j] = sum;
                }
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = matmul_backward;
        fn->n_inputs = 2;
        fn->inputs = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)a;
        fn->inputs[1] = (tensor*)b;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}


/* ═══════════════════════════════════════════════════════════════════════
 * Fused matmul + bias:  out = a @ b + bias
 *
 * Avoids allocating the intermediate (a @ b) tensor that separate
 * tensor_matmul + tensor_add would create.  The bias is added in-place
 * into the matmul output immediately after compute — one fewer scratch
 * allocation, one fewer full read of the intermediate, and one fewer
 * autograd node.
 *
 * bias must be 1-D with shape == {b->shape[b->ndim-1]} (i.e. last dim
 * of the matmul result).
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── matmul_add_backward ──
 *
 * Reuses matmul_backward for the a and b grads, then adds the bias
 * gradient (sum over all dims except the last).
 */
/* ── matmul_add_backward ──
 *
 * Reuses matmul_backward for the a and b grads, then adds the bias
 * gradient (sum over all dims except the last).
 * When trans_b != 0, forward was a @ b^T + bias (tied lm_head).
 */
static void matmul_add_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a     = fn->inputs[0];
    tensor *b     = fn->inputs[1];
    tensor *bias  = fn->inputs[2];
    int trans_b   = *(int*)fn->saved_tensors[0];

    /* ── d(matmul) — mirrors matmul_backward but adjusts for trans_b ── */
    {
        int na = a->ndim, nb = b->ndim;
        int K = a->shape[na - 1];                     /* inner dim */
        int M = a->shape[na - 2];                     /* a rows per batch */
        int N = trans_b ? b->shape[nb - 2]            /* out cols from b rows */
                       : b->shape[nb - 1];            /* out cols from b cols */

        /* When trans_b, b is [N, K] in memory.  Forward was a @ b^T.
         *   da = grad @ b          → (M,K) = (M,N) @ (N,K)
         *   db = grad^T @ a        → (N,K) = (N,M) @ (M,K)
         * When !trans_b, b is [K, N] in memory. Forward was a @ b.
         *   da = grad @ b^T        → (M,K) = (M,N) @ (N,K)
         *   db = a^T @ grad        → (K,N) = (K,M) @ (M,N)
         */

        /* ── 2D fast path ── */
        if (na == 2 && nb == 2) {
            float *ad = (float*)a->data;
            float *bd = (float*)b->data;
            float *gd = (float*)grad_output->data;
            int a_s0 = a->strides[0], a_s1 = a->strides[1];
            int g_s0 = grad_output->strides[0];
            int a_off = a->offset;
            (void)a_s1;

            if (a->grad_fn || a->requires_grad) {
                float *ag = _grad_ensure(a);
                if (trans_b) {
                    /* da = grad @ b  — CblasNoTrans on both */
                    int ld_g = tensor_is_contiguous(grad_output) ? N : g_s0;
                    int ld_b = tensor_is_contiguous(b) ? K : b->strides[0];
                    int ld_a = tensor_is_contiguous(a) ? K : a_s0;
#if NO_CBLAS
                    for (int i = 0; i < M; i++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int j = 0; j < N; j++)
                                sum += gd[i * ld_g + j] * bd[j * ld_b + k];
                            ag[a_off + i * ld_a + k] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                M, K, N, 1.0f, gd, ld_g,
                                bd, ld_b,
                                1.0f, ag + a_off, ld_a);
#endif
                } else {
                    /* da = grad @ b^T — CblasTrans on b */
                    int b_off = b->offset;
                    int b_s0 = b->strides[0];
#if NO_CBLAS
                    for (int i = 0; i < M; i++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int j = 0; j < N; j++)
                                sum += gd[i * g_s0 + j] * bd[b_off + k * b_s0 + j];
                            ag[a_off + i * a_s0 + k] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                M, K, N, 1.0f, gd, g_s0,
                                bd + b_off, b_s0,
                                1.0f, ag + a_off, a_s0);
#endif
                }
            }

            if ((b->grad_fn || b->requires_grad) && b != a) {
                float *bg = _grad_ensure(b);
                int b_off = b->offset;
                if (trans_b) {
                    /* db = grad^T @ a  — CblasTrans on grad, CblasNoTrans on a */
                    int ld_g = tensor_is_contiguous(grad_output) ? N : g_s0;
                    int ld_a = tensor_is_contiguous(a) ? K : a_s0;
                    int ld_b = tensor_is_contiguous(b) ? K : b->strides[0];
#if NO_CBLAS
                    for (int j = 0; j < N; j++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int i = 0; i < M; i++)
                                sum += gd[i * ld_g + j] * ad[a_off + i * ld_a + k];
                            bg[b_off + j * ld_b + k] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                N, K, M, 1.0f, gd, ld_g,
                                ad + a_off, ld_a,
                                1.0f, bg + b_off, ld_b);
#endif
                } else {
                    /* db = a^T @ grad  — CblasTrans on a, CblasNoTrans on grad */
                    int b_s0 = b->strides[0];
#if NO_CBLAS
                    for (int k = 0; k < K; k++)
                        for (int j = 0; j < N; j++) {
                            float sum = 0.0f;
                            for (int i = 0; i < M; i++)
                                sum += ad[a_off + i * a_s0 + k] * gd[i * g_s0 + j];
                            bg[b_off + k * b_s0 + j] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                K, N, M, 1.0f, ad + a_off, a_s0,
                                gd, g_s0,
                                1.0f, bg + b_off, b_s0);
#endif
                }
            }

            if (a == b && (a->grad_fn || a->requires_grad)) {
                float *ag = _grad_ensure(a);
                if (trans_b) {
                    int ld_g = tensor_is_contiguous(grad_output) ? N : g_s0;
                    int ld_a = tensor_is_contiguous(a) ? K : a_s0;
#if NO_CBLAS
                    for (int j = 0; j < N; j++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int i = 0; i < M; i++)
                                sum += gd[i * ld_g + j] * ad[a_off + i * ld_a + k];
                            ag[a_off + j * ld_a + k] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                N, K, M, 1.0f, gd, ld_g,
                                ad + a_off, ld_a,
                                1.0f, ag + a_off, ld_a);
#endif
                } else {
#if NO_CBLAS
                    int b_s0 = b->strides[0];
                    int b_off = b->offset;
                    (void)b_s0; (void)b_off;
                    for (int k = 0; k < K; k++)
                        for (int j = 0; j < N; j++) {
                            float sum = 0.0f;
                            for (int i = 0; i < M; i++)
                                sum += ad[a_off + i * a_s0 + k] * gd[i * g_s0 + j];
                            ag[a_off + k * a_s0 + j] += sum;
                        }
#else
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                K, N, M, 1.0f, ad + a_off, a_s0,
                                gd, g_s0,
                                1.0f, ag + a_off, a_s0);
#endif
                }
            }
        } else {
            /* Batched path — same logic as matmul_backward */
            int a_batch_ndim = na - 2;
            int b_batch_ndim = nb - 2;
            int batch_ndim;
            int batch_shape[8];
            batch_ndim = _bcast_ndim(a_batch_ndim, a->shape, b_batch_ndim, b->shape, batch_shape);
            int batch_size = _numel(batch_ndim, batch_shape);
            int out_ndim = batch_ndim + 2;
            int need_a = (a->grad_fn || a->requires_grad);
            int need_b = (b->grad_fn || b->requires_grad);
            int a_self = (a == b);
            float *ad = (float*)a->data;
            float *bd = (float*)b->data;
            float *gd = (float*)grad_output->data;
            float *ag = need_a ? _grad_ensure(a) : NULL;
            float *bg = need_b ? _grad_ensure(b) : NULL;
            int a_row_stride = a->strides[na - 2];
            int b_row_stride = b->strides[nb - 2];  /* leading dim / first of last-2 dims */
            int g_row_stride = grad_output->strides[out_ndim - 2];
            int a_col_stride = a->strides[na - 1];
            int b_col_stride = b->strides[nb - 1];  /* last-dim stride */
            int g_col_stride = grad_output->strides[out_ndim - 1];
            int slice_contig = (a_col_stride == 1 && b_col_stride == 1
                                && g_col_stride == 1);

            int no_broadcast = 1;
            int lead_a = batch_ndim - a_batch_ndim;
            int lead_b_tmp = batch_ndim - b_batch_ndim;
            for (int d = 0; d < batch_ndim; d++) {
                int a_sz = (d >= lead_a && d - lead_a < a_batch_ndim) ? a->shape[d - lead_a] : 0;
                int b_sz = (d >= lead_b_tmp && d - lead_b_tmp < b_batch_ndim) ? b->shape[d - lead_b_tmp] : 0;
                if ((a_sz > 0 && a_sz != batch_shape[d]) || (b_sz > 0 && b_sz != batch_shape[d])) {
                    no_broadcast = 0;
                    break;
                }
            }

#pragma omp parallel for if (no_broadcast && batch_size >= OMP_MIN_ITERS \
    && (long long)M * K * N >= 1000000 && (long long)M * K * N <= 50000000)
            for (int bi = 0; bi < batch_size; bi++) {
                int local_coord[8];
                int tmp = bi;
                for (int d = batch_ndim - 1; d >= 0; d--) {
                    local_coord[d] = tmp % batch_shape[d];
                    tmp /= batch_shape[d];
                }
                int a_off = _matmul_batch_off(a, batch_ndim, local_coord);
                int b_off = _matmul_batch_off(b, batch_ndim, local_coord);
                int g_off = _matmul_batch_off(grad_output, batch_ndim, local_coord);

                if (need_a) {
                    if (trans_b) {
                        /* da = grad @ b  — NoTrans on both */
                        if (slice_contig) {
#if NO_CBLAS
                            for (int i = 0; i < M; i++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int j = 0; j < N; j++)
                                        sum += gd[g_off + i * g_row_stride + j]
                                             * bd[b_off + j * b_row_stride + kk];
                                    ag[a_off + i * a_row_stride + kk] += sum;
                                }
#else
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                        M, K, N, 1.0f, gd + g_off, g_row_stride,
                                        bd + b_off, b_row_stride,
                                        1.0f, ag + a_off, a_row_stride);
#endif
                        } else {
                            for (int i = 0; i < M; i++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int j = 0; j < N; j++)
                                        sum += gd[g_off + i * g_row_stride + j * g_col_stride]
                                             * bd[b_off + j * b_row_stride + kk * b_col_stride];
                                    ag[a_off + i * a_row_stride + kk * a_col_stride] += sum;
                                }
                        }
                    } else {
                        /* da = grad @ b^T — CblasTrans on b */
                        if (slice_contig) {
#if NO_CBLAS
                            for (int i = 0; i < M; i++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int j = 0; j < N; j++)
                                        sum += gd[g_off + i * g_row_stride + j]
                                             * bd[b_off + kk * b_row_stride + j];
                                    ag[a_off + i * a_row_stride + kk] += sum;
                                }
#else
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                        M, K, N, 1.0f, gd + g_off, g_row_stride,
                                        bd + b_off, b_row_stride,
                                        1.0f, ag + a_off, a_row_stride);
#endif
                        } else {
                            for (int i = 0; i < M; i++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int j = 0; j < N; j++)
                                        sum += gd[g_off + i * g_row_stride + j * g_col_stride]
                                             * bd[b_off + kk * b_row_stride + j * b_col_stride];
                                    ag[a_off + i * a_row_stride + kk * a_col_stride] += sum;
                                }
                        }
                    }
                }

                if (need_b && !a_self) {
                    if (trans_b) {
                        /* db = grad^T @ a — Trans on grad, NoTrans on a */
                        if (slice_contig) {
#if NO_CBLAS
                            for (int j = 0; j < N; j++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += gd[g_off + i * g_row_stride + j]
                                             * ad[a_off + i * a_row_stride + kk];
                                    bg[b_off + j * b_row_stride + kk] += sum;
                                }
#else
                            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                        N, K, M, 1.0f, gd + g_off, g_row_stride,
                                        ad + a_off, a_row_stride,
                                        1.0f, bg + b_off, b_row_stride);
#endif
                        } else {
                            for (int j = 0; j < N; j++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += gd[g_off + i * g_row_stride + j * g_col_stride]
                                             * ad[a_off + i * a_row_stride + kk * a_col_stride];
                                    bg[b_off + j * b_row_stride + kk * b_col_stride] += sum;
                                }
                        }
                    } else {
                        /* db = a^T @ grad — Trans on a, NoTrans on grad */
                        if (slice_contig) {
#if NO_CBLAS
                            for (int kk = 0; kk < K; kk++)
                                for (int j = 0; j < N; j++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += ad[a_off + i * a_row_stride + kk]
                                             * gd[g_off + i * g_row_stride + j];
                                    bg[b_off + kk * b_row_stride + j] += sum;
                                }
#else
                            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                        K, N, M, 1.0f, ad + a_off, a_row_stride,
                                        gd + g_off, g_row_stride,
                                        1.0f, bg + b_off, b_row_stride);
#endif
                        } else {
                            for (int kk = 0; kk < K; kk++)
                                for (int j = 0; j < N; j++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += ad[a_off + i * a_row_stride + kk * a_col_stride]
                                             * gd[g_off + i * g_row_stride + j * g_col_stride];
                                    bg[b_off + kk * b_row_stride + j * b_col_stride] += sum;
                                }
                        }
                    }
                }

                if (a_self && need_a) {
                    if (trans_b) {
                        /* db = grad^T @ a  (same as db path for trans_b) */
                        if (slice_contig) {
#if NO_CBLAS
                            for (int j = 0; j < N; j++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += gd[g_off + i * g_row_stride + j]
                                             * ad[a_off + i * a_row_stride + kk];
                                    ag[a_off + j * b_row_stride + kk] += sum;
                                }
#else
                            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                        N, K, M, 1.0f, gd + g_off, g_row_stride,
                                        ad + a_off, a_row_stride,
                                        1.0f, ag + a_off, b_row_stride);
#endif
                        } else {
                            for (int j = 0; j < N; j++)
                                for (int kk = 0; kk < K; kk++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += gd[g_off + i * g_row_stride + j * g_col_stride]
                                             * ad[a_off + i * a_row_stride + kk * a_col_stride];
                                    ag[a_off + j * b_row_stride + kk * b_col_stride] += sum;
                                }
                        }
                    } else {
                        /* db = a^T @ grad (second term for a==b) */
                        if (slice_contig) {
#if NO_CBLAS
                            for (int kk = 0; kk < K; kk++)
                                for (int j = 0; j < N; j++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += ad[a_off + i * a_row_stride + kk]
                                             * gd[g_off + i * g_row_stride + j];
                                    ag[a_off + kk * a_row_stride + j] += sum;
                                }
#else
                            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                        K, N, M, 1.0f, ad + a_off, a_row_stride,
                                        gd + g_off, g_row_stride,
                                        1.0f, ag + a_off, a_row_stride);
#endif
                        } else {
                            for (int kk = 0; kk < K; kk++)
                                for (int j = 0; j < N; j++) {
                                    float sum = 0.0f;
                                    for (int i = 0; i < M; i++)
                                        sum += ad[a_off + i * a_row_stride + kk * a_col_stride]
                                             * gd[g_off + i * g_row_stride + j * g_col_stride];
                                    ag[a_off + kk * a_row_stride + j * a_col_stride] += sum;
                                }
                        }
                    }
                }
            }
        }
    }

    /* ── d(bias) = sum over all dims except last ──
     *
     * grad_output is contiguous [batch..., N].  Bias is [N].
     * Sum every element of grad_output whose flat index maps to the
     * same last-dim position.
     */
    if (bias->grad_fn || bias->requires_grad) {
        float *bg = _grad_ensure(bias);
        int N_bias  = bias->shape[0];
        int total   = tensor_numel(grad_output);
        int rows    = total / N_bias;
        float *gd   = (float*)grad_output->data;
        int boff    = bias->offset;
#pragma omp parallel for if (N_bias * rows >= 4096)
        for (int c = 0; c < N_bias; c++) {
            float sum = 0.0f;
#pragma omp simd reduction(+:sum)
            for (int r = 0; r < rows; r++)
                sum += gd[r * N_bias + c];
            bg[boff + c] += sum;
        }
    }
}


tensor *tensor_matmul_add(struct mem_pool *scratch, const tensor *a,
                          const tensor *b, int trans_b, const tensor *bias) {
    assert(a && b && bias);
    assert(a->ndim >= 2 && b->ndim >= 2 && "tensor_matmul_add: need at least 2D");
    assert(bias->ndim == 1 && "tensor_matmul_add: bias must be 1-D");

    int na = a->ndim, nb = b->ndim;
    int K = a->shape[na - 1];                      /* inner dim */
    int M = a->shape[na - 2];                      /* a rows (batch) */
    int N = trans_b ? b->shape[nb - 2]              /* out cols from b rows */
                   : b->shape[nb - 1];              /* out cols from b cols */

    /* Inner dim check */
    int b_inner = trans_b ? b->shape[nb - 1] : b->shape[nb - 2];
    assert(K == b_inner && "tensor_matmul_add: inner dim mismatch");
    assert(bias->shape[0] == N && "tensor_matmul_add: bias size != out cols");

    /* ── 2D case ── */
    if (na == 2 && nb == 2) {
        tensor *out = tensor_scratch(scratch, 2, (int[]){M, N}, 0);
        float *od = (float*)out->data;
        float *ad = (float*)a->data;
        float *bd = (float*)b->data;
        float *bdata = (float*)bias->data + bias->offset;

        int a_s0 = a->strides[0], a_s1 = a->strides[1];
        int a_off = a->offset;
        (void)a_s1;

        if (trans_b) {
            /* a @ b^T  where b is [N, K] */
            int b_row_stride = b->strides[0];
            int b_off = b->offset;
#if NO_CBLAS
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                        sum += ad[a_off + i * a_s0 + k]
                             * bd[b_off + j * b_row_stride + k];
                    od[out->offset + i * N + j] = sum + bdata[j];
                }
#else
            if (tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, N, K, 1.0f, ad + a_off, a_s0,
                            bd + b_off, b_row_stride,
                            0.0f, od, N);
            } else {
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++)
                            sum += ad[a_off + i * a_s0 + k]
                                 * bd[b_off + j * b_row_stride + k];
                        od[out->offset + i * N + j] = sum;
                    }
            }
            for (int i = 0; i < M * N; i++) od[i] += bdata[i % N];
#endif
        } else {
            /* a @ b  where b is [K, N] */
            int b_s0 = b->strides[0], b_s1 = b->strides[1];
            int b_off = b->offset;
#if NO_CBLAS
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                        sum += ad[a_off + i * a_s0 + k * a_s1]
                             * bd[b_off + k * b_s0 + j * b_s1];
                    od[out->offset + i * N + j] = sum + bdata[j];
                }
#else
            if (tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0f, ad + a_off, a_s0,
                            bd + b_off, b_s0,
                            0.0f, od, N);
            } else {
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++)
                            sum += ad[a_off + i * a_s0 + k * a_s1]
                                 * bd[b_off + k * b_s0 + j * b_s1];
                        od[out->offset + i * N + j] = sum;
                    }
            }
            for (int i = 0; i < M * N; i++) od[i] += bdata[i % N];
#endif
        }

        /* autograd tape */
        if (dnn_grad_enabled() &&
            (tensor_requires_grad(a) || tensor_requires_grad(b)
             || tensor_requires_grad(bias))) {
            grad_fn *fn = _grad_fn_create(scratch);
            fn->backward = matmul_add_backward;
            fn->n_inputs = 3;
            fn->inputs = _mem_pool_alloc(scratch, 3 * sizeof(tensor*), NULL);
            fn->inputs[0] = (tensor*)a;
            fn->inputs[1] = (tensor*)b;
            fn->inputs[2] = (tensor*)bias;
            fn->n_saved = 1;
            fn->saved_tensors = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
            int *saved_trans = _mem_pool_alloc(scratch, sizeof(int), NULL);
            *saved_trans = trans_b;
            fn->saved_tensors[0] = (tensor*)saved_trans;
            out->requires_grad = 1;
            out->grad_fn = fn;
        }
        return out;
    }

    /* ── Batched case (ND) ── */
    int a_batch_ndim = na - 2;
    int b_batch_ndim = nb - 2;
    int batch_ndim;
    int batch_shape[8];
    batch_ndim = _bcast_ndim(a_batch_ndim, a->shape, b_batch_ndim, b->shape, batch_shape);
    int batch_size = _numel(batch_ndim, batch_shape);

    int out_ndim = batch_ndim + 2;
    int out_shape[8];
    for (int i = 0; i < batch_ndim; i++) out_shape[i] = batch_shape[i];
    out_shape[batch_ndim] = M;
    out_shape[batch_ndim + 1] = N;

    tensor *out = tensor_scratch(scratch, out_ndim, out_shape, 0);
    float *od = (float*)out->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;
    float *bdata = (float*)bias->data + bias->offset;

    int a_row_stride = a->strides[na - 2];
    int a_col_stride = a->strides[na - 1];
    /* b is [..., K, N] (standard) or [..., N, K] (trans_b).
     * b->strides[nb-2] is the leading dim in both cases.
     * b->strides[nb-1] is the stride along the last dim (unit stride if contig). */
    int b_row_stride = b->strides[nb - 2];  /* BLAS leading dim */
    int b_col_stride = b->strides[nb - 1];  /* last-dim stride */

    int slice_contig = (a_col_stride == 1 && b_col_stride == 1);

#pragma omp parallel for if (batch_size >= OMP_MIN_ITERS \
    && (long long)M * K * N >= 1000000 && (long long)M * K * N <= 50000000)
    for (int bi = 0; bi < batch_size; bi++) {
        int local_coord[8];
        int tmp = bi;
        for (int d = batch_ndim - 1; d >= 0; d--) {
            local_coord[d] = tmp % batch_shape[d];
            tmp /= batch_shape[d];
        }

        int a_off = _matmul_batch_off(a, batch_ndim, local_coord);
        int b_off = _matmul_batch_off(b, batch_ndim, local_coord);

        if (slice_contig) {
            if (trans_b) {
#if NO_CBLAS
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++)
                            sum += ad[a_off + i * a_row_stride + k]
                                 * bd[b_off + j * b_row_stride + k];
                        od[bi * M * N + i * N + j] = sum + bdata[j];
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, N, K, 1.0f,
                            ad + a_off, a_row_stride,
                            bd + b_off, b_row_stride,
                            0.0f, od + bi * M * N, N);
                for (int i = 0; i < M * N; i++)
                    od[bi * M * N + i] += bdata[i % N];
#endif
            } else {
#if NO_CBLAS
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++)
                            sum += ad[a_off + i * a_row_stride + k]
                                 * bd[b_off + k * b_row_stride + j];
                        od[bi * M * N + i * N + j] = sum + bdata[j];
                    }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0f,
                            ad + a_off, a_row_stride,
                            bd + b_off, b_row_stride,
                            0.0f, od + bi * M * N, N);
                for (int i = 0; i < M * N; i++)
                    od[bi * M * N + i] += bdata[i % N];
#endif
            }
        } else {
            if (trans_b) {
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++)
                            sum += ad[a_off + i * a_row_stride + k * a_col_stride]
                                 * bd[b_off + j * b_row_stride + k * b_col_stride];
                        od[bi * M * N + i * N + j] = sum + bdata[j];
                    }
            } else {
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++)
                            sum += ad[a_off + i * a_row_stride + k * a_col_stride]
                                 * bd[b_off + k * b_row_stride + j * b_col_stride];
                        od[bi * M * N + i * N + j] = sum + bdata[j];
                    }
            }
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b)
         || tensor_requires_grad(bias))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = matmul_add_backward;
        fn->n_inputs = 3;
        fn->inputs = _mem_pool_alloc(scratch, 3 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)a;
        fn->inputs[1] = (tensor*)b;
        fn->inputs[2] = (tensor*)bias;
        fn->n_saved = 1;
        fn->saved_tensors = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        int *saved_trans = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *saved_trans = trans_b;
        fn->saved_tensors[0] = (tensor*)saved_trans;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
