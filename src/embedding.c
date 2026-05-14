#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include <assert.h>
#include <string.h>

/* ── embedding_backward ──
 *
 * Gradients flow to the embedding table only (ids are not differentiable).
 *
 * d_table[ids[i], j] += d_out[i, j]
 *
 * Handles duplicate IDs via accumulation (correct scatter-add).
 */

static void embedding_backward(grad_fn *fn, tensor *grad_output) {
    tensor *table = fn->inputs[0];
    tensor *ids   = fn->saved_tensors[0];

    int   *id_data = (int*)ids->data;
    int    N       = ids->shape[0];
    int    d_model = table->shape[1];
    float *gd      = (float*)grad_output->data;
    int    g_s0    = grad_output->strides[0];

    /* only table gets gradients (ids are integers — no grad) */
    if (table->grad_fn || table->requires_grad) {
        float *tg = _grad_ensure(table);
        int t_s0 = table->strides[0];
        int t_s1 = table->strides[1];

        if (table->contiguous && grad_output->contiguous) {
            /* contiguous fast path */
            for (int i = 0; i < N; i++) {
                int idx = id_data[i];
                float *g_row = gd + i * g_s0;
                float *t_row = tg + idx * t_s0;
                for (int j = 0; j < d_model; j++)
                    t_row[j] += g_row[j];
            }
        } else {
            /* strided fallback */
            for (int i = 0; i < N; i++) {
                int idx = id_data[i];
                for (int j = 0; j < d_model; j++)
                    tg[idx * t_s0 + j * t_s1] += gd[i * g_s0 + j];
            }
        }
    }
}

/* ── tensor_embedding ──
 *
 *   out[i, j] = table[ids[i], j]
 *
 *   table — 2D float tensor [vocab_size, d_model], MUST be contiguous
 *   ids   — 1D int tensor [N] (labels, stored as int* in the data region)
 *
 *   Returns a new tensor [N, d_model] from scratch pool.
 */
tensor *tensor_embedding(const tensor *table, const tensor *ids) {
    assert(table && ids);
    assert(table->ndim == 2 && "embedding: table must be 2D");
    assert(ids->ndim == 1 && "embedding: ids must be 1D");
    assert(table->contiguous && "embedding: table must be contiguous");

    int vocab_size = table->shape[0];
    int d_model    = table->shape[1];
    int N          = ids->shape[0];
    int   *id_data = (int*)ids->data;
    float *td      = (float*)table->data;

    tensor *out = _tensor_scratch_create(2, (int[]){N, d_model}, 0);
    float  *od  = (float*)out->data;

    /* forward: copy rows from table indexed by ids */
    if (table->contiguous) {
        for (int i = 0; i < N; i++) {
            int idx = id_data[i];
            assert(idx >= 0 && idx < vocab_size && "embedding: id out of range");
            memcpy(od + i * d_model, td + idx * d_model, d_model * sizeof(float));
        }
    } else {
        /* strided table (unlikely but handle) */
        for (int i = 0; i < N; i++) {
            int idx = id_data[i];
            assert(idx >= 0 && idx < vocab_size && "embedding: id out of range");
            for (int j = 0; j < d_model; j++)
                od[i * d_model + j] = td[idx * table->strides[0] + j * table->strides[1]];
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(table)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = embedding_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)table;
        fn->n_saved = 1;
        fn->saved_tensors = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)ids;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
