#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_elem_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── add_backward ── */

static void add_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    int out_ndim = grad_output->ndim;
    int out_numel = _numel(grad_output->ndim, grad_output->shape);
    float *g_data = (float*)grad_output->data;

    /* accumulate only — backward never zeros. user calls zero_grad() before backward */
    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }

        float g = g_data[i];

        /* accumulate into any tensor that needs gradient —
           either to propagate (grad_fn) or to store on leaf (requires_grad) */
        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            ag[_bcast_off(a, out_ndim, coord)] += g;
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            bg[_bcast_off(b, out_ndim, coord)] += g;
        }
    }
}

/* ── Arithmetic ── */

tensor *tensor_add(const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_add: incompatible shapes");

    tensor *out = _tensor_scratch_create(ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim_out - 1; d >= 0; d--) {
            coord[d] = r % shape_out[d];
            r /= shape_out[d];
        }
        od[out->offset + i] = ad[_bcast_off(a, ndim_out, coord)]
                            + bd[_bcast_off(b, ndim_out, coord)];
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = add_backward;
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

/* ── sub_backward ── */

static void sub_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    /* d(a-a)/da = 0 — gradient contribution is zero */
    if (a == b) {
        if (a->requires_grad && !a->grad_fn)
            _grad_ensure(a);
        return;
    }

    int out_ndim = grad_output->ndim;
    int out_numel = _numel(grad_output->ndim, grad_output->shape);
    float *g_data = (float*)grad_output->data;

    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }

        float g = g_data[i];

        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            ag[_bcast_off(a, out_ndim, coord)] += g;
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            bg[_bcast_off(b, out_ndim, coord)] -= g;
        }
    }
}

tensor *tensor_sub(const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_sub: incompatible shapes");

    tensor *out = _tensor_scratch_create(ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim_out - 1; d >= 0; d--) {
            coord[d] = r % shape_out[d];
            r /= shape_out[d];
        }
        od[out->offset + i] = ad[_bcast_off(a, ndim_out, coord)]
                            - bd[_bcast_off(b, ndim_out, coord)];
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = sub_backward;
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

/* ── mul_backward ── */

static void mul_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    int out_ndim = grad_output->ndim;
    int out_numel = _numel(grad_output->ndim, grad_output->shape);
    float *g_data = (float*)grad_output->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }

        float g = g_data[i];
        float bv = bd[_bcast_off(b, out_ndim, coord)];
        float av = ad[_bcast_off(a, out_ndim, coord)];

        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            ag[_bcast_off(a, out_ndim, coord)] += bv * g;
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            bg[_bcast_off(b, out_ndim, coord)] += av * g;
        }
    }
}

tensor *tensor_mul(const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_mul: incompatible shapes");

    tensor *out = _tensor_scratch_create(ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim_out - 1; d >= 0; d--) {
            coord[d] = r % shape_out[d];
            r /= shape_out[d];
        }
        od[out->offset + i] = ad[_bcast_off(a, ndim_out, coord)]
                            * bd[_bcast_off(b, ndim_out, coord)];
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = mul_backward;
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

/* ── div_backward ── */

static void div_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    /* d(a/a)/da = 0 — constant 1, no gradient flows through */
    if (a == b) {
        if (a->requires_grad && !a->grad_fn)
            _grad_ensure(a);
        return;
    }

    int out_ndim = grad_output->ndim;
    int out_numel = _numel(grad_output->ndim, grad_output->shape);
    float *g_data = (float*)grad_output->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }

        float g = g_data[i];
        float av = ad[_bcast_off(a, out_ndim, coord)];
        float bv = bd[_bcast_off(b, out_ndim, coord)];

        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            ag[_bcast_off(a, out_ndim, coord)] += (1.0f / bv) * g;
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            bg[_bcast_off(b, out_ndim, coord)] += (-av / (bv * bv)) * g;
        }
    }
}

tensor *tensor_div(const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_div: incompatible shapes");

    tensor *out = _tensor_scratch_create(ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim_out - 1; d >= 0; d--) {
            coord[d] = r % shape_out[d];
            r /= shape_out[d];
        }
        od[out->offset + i] = ad[_bcast_off(a, ndim_out, coord)]
                            / bd[_bcast_off(b, ndim_out, coord)];
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = div_backward;
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

/* ── pow_backward ── */

static void pow_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    float exp = *(float*)fn->saved_tensors[0];
    int n = _numel(a->ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *ad = (float*)a->data;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        for (int i = 0; i < n; i++) {
            int off = _flat_off(a, i);
            ag[off] += exp * powf(ad[off], exp - 1.0f) * g_data[i];
        }
    }
}

tensor *tensor_pow(const tensor *t, float exp) {
    assert(t);

    tensor *out = _tensor_scratch_create(t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = t->ndim - 1; d >= 0; d--) {
            coord[d] = r % t->shape[d];
            r /= t->shape[d];
        }
        od[out->offset + i] = powf(td[_bcast_off(t, t->ndim, coord)], exp);
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = pow_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 1;
        fn->saved_tensors = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        float *exp_data = mem_scratch_alloc(sizeof(float), NULL);
        *exp_data = exp;
        fn->saved_tensors[0] = (tensor*)exp_data;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── neg_backward ── */

static void neg_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    int n = _numel(a->ndim, a->shape);
    float *g_data = (float*)grad_output->data;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        for (int i = 0; i < n; i++)
            ag[_flat_off(a, i)] -= g_data[i];
    }
}

tensor *tensor_neg(const tensor *t) {
    assert(t);

    tensor *out = _tensor_scratch_create(t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = t->ndim - 1; d >= 0; d--) {
            coord[d] = r % t->shape[d];
            r /= t->shape[d];
        }
        od[out->offset + i] = -td[_bcast_off(t, t->ndim, coord)];
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = neg_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
