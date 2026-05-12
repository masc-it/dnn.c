#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_reduce_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* ── sum_backward ── */

static void sum_backward(grad_fn *fn, tensor *grad_output) {
    tensor *t = fn->inputs[0];
    if (!(t->grad_fn || t->requires_grad)) return;

    float *tg = _grad_ensure(t);
    float *g_data = (float*)grad_output->data;
    int n = _numel(t->ndim, t->shape);

    /* broadcast grad_output (which has shape[dim]=1) back to full input shape */
    for (int i = 0; i < n; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = t->ndim - 1; d >= 0; d--) {
            coord[d] = r % t->shape[d];
            r /= t->shape[d];
        }
        tg[_flat_off(t, i)] += g_data[_bcast_off(grad_output, t->ndim, coord)];
    }
}

tensor *tensor_sum(const tensor *t, int dim) {
    assert(t);
    int ndim = t->ndim;
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "tensor_sum: dim out of range");

    /* keepdim: output shape[dim] = 1 */
    int shape_out[DNN_MAX_DIMS];
    memcpy(shape_out, t->shape, ndim * sizeof(int));
    shape_out[dim] = 1;

    tensor *out = _tensor_scratch_create(ndim, shape_out, 0);
    int out_numel = _numel(ndim, shape_out);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % shape_out[d];
            r /= shape_out[d];
        }
        /* coord[dim] is 0 — sum over all k along dim */
        float sum = 0.0f;
        for (int k = 0; k < t->shape[dim]; k++) {
            coord[dim] = k;
            sum += td[_bcast_off(t, ndim, coord)];
        }
        coord[dim] = 0;
        od[_flat_off(out, i)] = sum;
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = sum_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

tensor *tensor_mean(const tensor *t, int dim) {
    assert(t);
    float inv_n = 1.0f / (float)t->shape[dim];

    /* sum then scale: mean = sum(t, dim) * (1/n) */
    tensor *s = tensor_sum(t, dim);

    tensor *scale = _tensor_scratch_create(1, (int[]){1}, 0);
    ((float*)scale->data)[0] = inv_n;

    return tensor_mul(s, scale);
}
