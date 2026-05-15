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
