#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_matrix_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* ── matmul_backward ── */

static void matmul_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    float *ad = (float*)a->data;
    float *bd = (float*)b->data;
    float *gd = (float*)grad_output->data;

    int a_s0 = a->strides[0], a_s1 = a->strides[1];
    int b_s0 = b->strides[0], b_s1 = b->strides[1];
    int g_s0 = grad_output->strides[0];
    int a_off = a->offset, b_off = b->offset;

    /* da = gd @ B^T  → shape (M, K) */
    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float sum = 0.0f;
                for (int j = 0; j < N; j++)
                    sum += gd[i * g_s0 + j] * bd[b_off + k * b_s0 + j * b_s1];
                ag[a_off + i * a_s0 + k * a_s1] += sum;
            }
        }
    }

    /* db = A^T @ gd  → shape (K, N) */
    if ((b->grad_fn || b->requires_grad) && b != a) {
        float *bg = _grad_ensure(b);
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int i = 0; i < M; i++)
                    sum += ad[a_off + i * a_s0 + k * a_s1] * gd[i * g_s0 + j];
                bg[b_off + k * b_s0 + j * b_s1] += sum;
            }
        }
    }

    /* a == b: da += A^T @ dC (dC@A^T already accumulated from first branch) */
    if (a == b && (a->grad_fn || a->requires_grad)) {
        float *ag = _grad_ensure(a);
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int i = 0; i < M; i++)
                    sum += ad[a_off + i * a_s0 + k * a_s1] * gd[i * g_s0 + j];
                ag[a_off + k * a_s0 + j * a_s1] += sum;
            }
        }
    }
}

tensor *tensor_matmul(const tensor *a, const tensor *b) {
    assert(a && b);
    assert(a->ndim >= 2 && b->ndim >= 2 && "tensor_matmul: need at least 2D");
    assert(a->shape[1] == b->shape[0] && "tensor_matmul: inner dim mismatch");

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    tensor *out = _tensor_scratch_create(2, (int[]){M, N}, 0);
    float *od = (float*)out->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    int a_s0 = a->strides[0], a_s1 = a->strides[1];
    int b_s0 = b->strides[0], b_s1 = b->strides[1];
    int a_off = a->offset, b_off = b->offset;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += ad[a_off + i * a_s0 + k * a_s1]
                     * bd[b_off + k * b_s0 + j * b_s1];
            od[out->offset + i * N + j] = sum;
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = matmul_backward;
        fn->n_inputs = 2;
        fn->inputs = mem_scratch_alloc(2 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)a;
        fn->inputs[1] = (tensor*)b;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
