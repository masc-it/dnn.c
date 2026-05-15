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

    int out_numel = tensor_numel(grad_output);
    float *g_data = tensor_data_ptr(grad_output);  /* contiguous */

    /* Fast path: both same shape as grad_output, no broadcasting */
    if (_grad_contiguous(a, grad_output) && _grad_contiguous(b, grad_output)) {
        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++) ag[a->offset + i] += g_data[i];
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++) bg[b->offset + i] += g_data[i];
        }
        return;
    }
    /* Broadcast fallback: precompute offsets (SIMD-able integer math),
     * then accumulate (scalar scatter, cannot SIMD on NEON). */
    float *ag_ptr = (a->grad_fn || a->requires_grad) ? _grad_ensure(a) : NULL;
    float *bg_ptr = (b->grad_fn || b->requires_grad) ? _grad_ensure(b) : NULL;
    int *a_offs = ag_ptr ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int *b_offs = bg_ptr ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int out_ndim = grad_output->ndim;
    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }
        if (a_offs) a_offs[i] = _bcast_off(a, out_ndim, coord);
        if (b_offs) b_offs[i] = _bcast_off(b, out_ndim, coord);
    }
    if (ag_ptr) {
        for (int i = 0; i < out_numel; i++)
            ag_ptr[a_offs[i]] += g_data[i];
    }
    if (bg_ptr) {
        for (int i = 0; i < out_numel; i++)
            bg_ptr[b_offs[i]] += g_data[i];
    }
}

/* ── tensor_add ── */

tensor *tensor_add(struct mem_pool *scratch, const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_add: incompatible shapes");

    tensor *out = tensor_scratch(scratch, ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;

    if (_same_contiguous(a, b)) {
        /* Fast path: same contiguous shape, no broadcasting */
        float *af = (float*)a->data + a->offset;
        float *bf = (float*)b->data + b->offset;
        #pragma omp simd
        for (int i = 0; i < numel; i++)
            od[i] = af[i] + bf[i];
    } else {
        /* General broadcast path */
        float *ad = (float*)a->data;
        float *bd = (float*)b->data;
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim_out - 1; d >= 0; d--) {
                coord[d] = r % shape_out[d];
                r /= shape_out[d];
            }
            int a_off = _bcast_off(a, ndim_out, coord);
            int b_off = _bcast_off(b, ndim_out, coord);
            od[out->offset + i] = ad[a_off] + bd[b_off];
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = add_backward;
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

    int out_numel = tensor_numel(grad_output);
    float *g_data = tensor_data_ptr(grad_output);

    /* Fast path */
    if (_grad_contiguous(a, grad_output) && _grad_contiguous(b, grad_output)) {
        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++) ag[a->offset + i] += g_data[i];
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++) bg[b->offset + i] -= g_data[i];
        }
        return;
    }
    float *ag_ptr = (a->grad_fn || a->requires_grad) ? _grad_ensure(a) : NULL;
    float *bg_ptr = (b->grad_fn || b->requires_grad) ? _grad_ensure(b) : NULL;
    int *a_offs = ag_ptr ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int *b_offs = bg_ptr ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int out_ndim = grad_output->ndim;
    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }
        if (a_offs) a_offs[i] = _bcast_off(a, out_ndim, coord);
        if (b_offs) b_offs[i] = _bcast_off(b, out_ndim, coord);
    }
    if (ag_ptr) {
        for (int i = 0; i < out_numel; i++)
            ag_ptr[a_offs[i]] += g_data[i];
    }
    if (bg_ptr) {
        for (int i = 0; i < out_numel; i++)
            bg_ptr[b_offs[i]] -= g_data[i];
    }
}

/* ── tensor_sub ── */

tensor *tensor_sub(struct mem_pool *scratch, const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_sub: incompatible shapes");

    tensor *out = tensor_scratch(scratch, ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;

    if (_same_contiguous(a, b)) {
        float *af = (float*)a->data + a->offset;
        float *bf = (float*)b->data + b->offset;
        #pragma omp simd
        for (int i = 0; i < numel; i++)
            od[i] = af[i] - bf[i];
    } else {
        float *ad = (float*)a->data;
        float *bd = (float*)b->data;
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim_out - 1; d >= 0; d--) {
                coord[d] = r % shape_out[d];
                r /= shape_out[d];
            }
            int a_off = _bcast_off(a, ndim_out, coord);
            int b_off = _bcast_off(b, ndim_out, coord);
            od[out->offset + i] = ad[a_off] - bd[b_off];
        }
    }

    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = sub_backward;
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

/* ── mul_backward ── */

static void mul_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    int out_numel = tensor_numel(grad_output);
    float *g_data = tensor_data_ptr(grad_output);

    /* Fast path */
    if (_grad_contiguous(a, grad_output) && _grad_contiguous(b, grad_output)) {
        float *af = (float*)a->data + a->offset;
        float *bf = (float*)b->data + b->offset;
        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++)
                ag[a->offset + i] += bf[i] * g_data[i];
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++)
                bg[b->offset + i] += af[i] * g_data[i];
        }
        return;
    }
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;
    float *ag_ptr = (a->grad_fn || a->requires_grad) ? _grad_ensure(a) : NULL;
    float *bg_ptr = (b->grad_fn || b->requires_grad) ? _grad_ensure(b) : NULL;
    int need_offs = (ag_ptr || bg_ptr);
    int *a_offs = need_offs ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int *b_offs = need_offs ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int out_ndim = grad_output->ndim;
    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }
        if (a_offs) a_offs[i] = _bcast_off(a, out_ndim, coord);
        if (b_offs) b_offs[i] = _bcast_off(b, out_ndim, coord);
    }
    if (ag_ptr) {
        for (int i = 0; i < out_numel; i++)
            ag_ptr[a_offs[i]] += bd[b_offs[i]] * g_data[i];
    }
    if (bg_ptr) {
        for (int i = 0; i < out_numel; i++)
            bg_ptr[b_offs[i]] += ad[a_offs[i]] * g_data[i];
    }
}

/* ── tensor_mul ── */

tensor *tensor_mul(struct mem_pool *scratch, const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_mul: incompatible shapes");

    tensor *out = tensor_scratch(scratch, ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;

    if (_same_contiguous(a, b)) {
        float *af = (float*)a->data + a->offset;
        float *bf = (float*)b->data + b->offset;
        #pragma omp simd
        for (int i = 0; i < numel; i++)
            od[i] = af[i] * bf[i];
    } else {
        float *ad = (float*)a->data;
        float *bd = (float*)b->data;
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim_out - 1; d >= 0; d--) {
                coord[d] = r % shape_out[d];
                r /= shape_out[d];
            }
            int a_off = _bcast_off(a, ndim_out, coord);
            int b_off = _bcast_off(b, ndim_out, coord);
            od[out->offset + i] = ad[a_off] * bd[b_off];
        }
    }

    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = mul_backward;
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

/* ── div_backward ── */

static void div_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];

    if (a == b) {
        if (a->requires_grad && !a->grad_fn)
            _grad_ensure(a);
        return;
    }

    int out_numel = tensor_numel(grad_output);
    float *g_data = tensor_data_ptr(grad_output);

    /* Fast path */
    if (_grad_contiguous(a, grad_output) && _grad_contiguous(b, grad_output)) {
        float *af = (float*)a->data + a->offset;
        float *bf = (float*)b->data + b->offset;
        if (a->grad_fn || a->requires_grad) {
            float *ag = _grad_ensure(a);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++)
                ag[a->offset + i] += (1.0f / bf[i]) * g_data[i];
        }
        if (b->grad_fn || b->requires_grad) {
            float *bg = _grad_ensure(b);
            #pragma omp simd
            for (int i = 0; i < out_numel; i++)
                bg[b->offset + i] += (-af[i] / (bf[i] * bf[i])) * g_data[i];
        }
        return;
    }
    float *ad = (float*)a->data;
    float *bd = (float*)b->data;
    float *ag_ptr = (a->grad_fn || a->requires_grad) ? _grad_ensure(a) : NULL;
    float *bg_ptr = (b->grad_fn || b->requires_grad) ? _grad_ensure(b) : NULL;
    int need_offs = (ag_ptr || bg_ptr);
    int *a_offs = need_offs ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int *b_offs = need_offs ? _mem_pool_alloc(fn->pool, out_numel * sizeof(int), NULL) : NULL;
    int out_ndim = grad_output->ndim;
    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }
        if (a_offs) a_offs[i] = _bcast_off(a, out_ndim, coord);
        if (b_offs) b_offs[i] = _bcast_off(b, out_ndim, coord);
    }
    if (ag_ptr) {
        for (int i = 0; i < out_numel; i++)
            ag_ptr[a_offs[i]] += (1.0f / bd[b_offs[i]]) * g_data[i];
    }
    if (bg_ptr) {
        for (int i = 0; i < out_numel; i++)
            bg_ptr[b_offs[i]] += (-ad[a_offs[i]] / (bd[b_offs[i]] * bd[b_offs[i]])) * g_data[i];
    }
}

/* ── tensor_div ── */

tensor *tensor_div(struct mem_pool *scratch, const tensor *a, const tensor *b) {
    assert(a && b);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(a->ndim, a->shape, b->ndim, b->shape, shape_out);
    assert(ndim_out > 0 && "tensor_div: incompatible shapes");

    tensor *out = tensor_scratch(scratch, ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;

    if (_same_contiguous(a, b)) {
        float *af = (float*)a->data + a->offset;
        float *bf = (float*)b->data + b->offset;
        #pragma omp simd
        for (int i = 0; i < numel; i++)
            od[i] = af[i] / bf[i];
    } else {
        float *ad = (float*)a->data;
        float *bd = (float*)b->data;
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim_out - 1; d >= 0; d--) {
                coord[d] = r % shape_out[d];
                r /= shape_out[d];
            }
            int a_off = _bcast_off(a, ndim_out, coord);
            int b_off = _bcast_off(b, ndim_out, coord);
            od[out->offset + i] = ad[a_off] / bd[b_off];
        }
    }

    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = div_backward;
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

/* ── pow_backward ── */

static void pow_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    float exp = *(float*)fn->saved_tensors[0];
    int n = _numel(a->ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *ad = (float*)a->data;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        if (tensor_is_contiguous(a) && tensor_is_contiguous(grad_output)) {
            float *ap = ad + a->offset;
            #pragma omp simd
            for (int i = 0; i < n; i++)
                ag[a->offset + i] += exp * powf(ap[i], exp - 1.0f) * g_data[i];
        } else {
            for (int i = 0; i < n; i++) {
                int off = _flat_off(a, i);
                ag[off] += exp * powf(ad[off], exp - 1.0f) * g_data[i];
            }
        }
    }
}

tensor *tensor_pow(struct mem_pool *scratch, const tensor *t, float exp) {
    assert(t);

    tensor *out = tensor_scratch(scratch, t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    if (tensor_is_contiguous(t)) {
        float *tp = td + t->offset;
        #pragma omp simd
        for (int i = 0; i < numel; i++)
            od[out->offset + i] = powf(tp[i], exp);
    } else {
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = t->ndim - 1; d >= 0; d--) {
                coord[d] = r % t->shape[d];
                r /= t->shape[d];
            }
            od[out->offset + i] = powf(td[_bcast_off(t, t->ndim, coord)], exp);
        }
    }

    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = pow_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 1;
        fn->saved_tensors = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        float *exp_data = _mem_pool_alloc(scratch, sizeof(float), NULL);
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
        if (tensor_is_contiguous(a) && tensor_is_contiguous(grad_output)) {
            #pragma omp simd
            for (int i = 0; i < n; i++)
                ag[a->offset + i] -= g_data[i];
        } else {
            for (int i = 0; i < n; i++)
                ag[_flat_off(a, i)] -= g_data[i];
        }
    }
}

tensor *tensor_neg(struct mem_pool *scratch, const tensor *t) {
    assert(t);

    tensor *out = tensor_scratch(scratch, t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    if (tensor_is_contiguous(t)) {
        float *tp = td + t->offset;
        #pragma omp simd
        for (int i = 0; i < numel; i++)
            od[out->offset + i] = -tp[i];
    } else {
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = t->ndim - 1; d >= 0; d--) {
                coord[d] = r % t->shape[d];
                r /= t->shape[d];
            }
            od[out->offset + i] = -td[_bcast_off(t, t->ndim, coord)];
        }
    }

    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = neg_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }
    return out;
}

/* ── tensor_triu ──
 *
 * Create [N, N] causal mask from scratch pool.
 *   Element (i,j) = -INFINITY if j >= i + diagonal,
 *                   0.0f otherwise.
 *
 * For standard causal attention (attend to self + past) use diagonal=1:
 *   j > i  → -inf (future masked)
 *   j <= i → 0    (self + past kept)
 *
 * For no-self-attention use diagonal=0:
 *   j >= i → -inf (self + future masked)
 *   j < i  → 0    (past only)
 *
 * No grad_fn attached — mask is a constant input. Gradients flow through
 * tensor_add into it but are dropped (requires_grad=0).
 */
tensor *tensor_triu(struct mem_pool *scratch, int N, int diagonal) {
    assert(N > 0);
    tensor *out = tensor_scratch(scratch, 2, (int[]){N, N}, 0);
    float *od = (float*)out->data + out->offset;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            if (j >= i + diagonal) {
                od[idx] = -INFINITY;
            } else {
                od[idx] = 0.0f;
            }
        }
    }
    return out;
}


/* ── cat_backward ──
 *
 * Gradient for concatenation: split grad_output along dim.
 * Elements with coord[dim] < a->shape[dim] flow to a, rest to b.
 * If a == b (self-concatenation), both halves accumulate into the same buffer.
 */

static void cat_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *b = fn->inputs[1];
    int dim       = *(int*)fn->saved_tensors[0];
    int ndim      = a->ndim;
    int a_sz      = a->shape[dim];
    float *gd     = (float*)grad_output->data;
    int total     = tensor_numel(grad_output);
    int need_a    = (a->grad_fn || a->requires_grad);
    int need_b    = (b->grad_fn || b->requires_grad);
    int a_self    = (a == b);
    float *ag     = need_a ? _grad_ensure(a) : NULL;
    float *bg     = need_b ? _grad_ensure(b) : NULL;

    /* ── Contiguous fast paths (avoid coord-decompose in common case) ── */

    /* Slice a portion: copy [0:a_sz) along dim from grad_output to a */
    if (need_a && a->contiguous && grad_output->contiguous) {
        int inner = 1;
        for (int d = dim + 1; d < ndim; d++) inner *= grad_output->shape[d];
        int outer = 1;
        for (int d = 0; d < dim; d++) outer *= grad_output->shape[d];
        int gs = inner;
        int as = 1;
        for (int d = dim + 1; d < ndim; d++) as *= a->shape[d];

        for (int o = 0; o < outer; o++) {
            float *g_row = gd + o * (long)grad_output->shape[dim] * gs;
            float *a_row = ag + o * (long)a_sz * as;
            for (int k = 0; k < a_sz; k++)
                for (int i = 0; i < inner; i++)
                    a_row[k * as + i] += g_row[k * gs + i];
        }
        need_a = 0;
    }

    /* Slice b portion: copy [a_sz:a_sz+b_sz) along dim from grad_output to b */
    if (need_b && b->contiguous && grad_output->contiguous && !a_self) {
        int b_sz = b->shape[dim];
        int inner = 1;
        for (int d = dim + 1; d < ndim; d++) inner *= grad_output->shape[d];
        int outer = 1;
        for (int d = 0; d < dim; d++) outer *= grad_output->shape[d];
        int gs = inner;
        int bs = 1;
        for (int d = dim + 1; d < ndim; d++) bs *= b->shape[d];

        for (int o = 0; o < outer; o++) {
            float *g_row = gd + o * (long)grad_output->shape[dim] * gs;
            float *b_row = bg + o * (long)b_sz * bs;
            for (int k = 0; k < b_sz; k++)
                for (int i = 0; i < inner; i++)
                    b_row[k * bs + i] += g_row[(a_sz + k) * gs + i];
        }
        need_b = 0;
    }

    /* Self-concatenation: after handling first half of the contiguous path
     * above (which went into ag), the second half must also go into ag. */
    if (a_self && need_a) {
        need_b = 1;
        bg = ag;
    }

    /* ── General coord-decompose fallback ── */
    {
        int coord[DNN_MAX_DIMS];
        for (int flat = 0; flat < total && (need_a || need_b); flat++) {
            int rem = flat;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = rem % grad_output->shape[d];
                rem /= grad_output->shape[d];
            }

            int c = coord[dim];
            float gv = gd[_flat_off(grad_output, flat)];

            if (c < a_sz) {
                if (need_a) {
                    int a_flat = 0;
                    for (int d = 0; d < ndim; d++)
                        a_flat = a_flat * a->shape[d] + coord[d];
                    ag[_flat_off(a, a_flat)] += gv;
                }
            } else {
                if (need_b) {
                    int bc = c - a_sz;
                    int b_flat = 0;
                    for (int d = 0; d < ndim; d++)
                        b_flat = b_flat * b->shape[d] + (d == dim ? bc : coord[d]);
                    bg[_flat_off(b, b_flat)] += gv;
                }
            }
        }
    }
}


/* ── Internal copy helper for tensor_cat ──
 *
 * Copies tensor t into contiguous output buffer od at offset out_dim_offset
 * along dim.  Handles contiguous (fast row-by-row memcpy) and strided
 * (coord-decompose) inputs.
 */

static void _cat_copy(float *od, const tensor *t, int dim,
                       int out_dim_offset, const int *out_shape) {
    int ndim    = t->ndim;
    float *td   = (float*)t->data;
    int d_sz    = t->shape[dim];
    int inner   = 1;
    int outer   = 1;
    for (int d = dim + 1; d < ndim; d++) inner *= out_shape[d];
    for (int d = 0; d < dim; d++)        outer *= out_shape[d];
    int osd = inner;  /* output stride along dim */

    if (tensor_is_contiguous(t)) {
        int tsd = 1;
        for (int d = dim + 1; d < ndim; d++) tsd *= t->shape[d];

        for (int o = 0; o < outer; o++) {
            float *dst = od + o * (long)out_shape[dim] * osd
                              + (long)out_dim_offset * osd;
            float *src = td + t->offset + o * (long)d_sz * tsd;
            memcpy(dst, src, (size_t)d_sz * tsd * sizeof(float));
        }
    } else {
        int total = tensor_numel(t);
        for (int flat = 0; flat < total; flat++) {
            int coord[DNN_MAX_DIMS];
            int rem = flat;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = rem % t->shape[d];
                rem /= t->shape[d];
            }

            int t_off = t->offset;
            for (int d = 0; d < ndim; d++)
                t_off += coord[d] * t->strides[d];

            int o_flat = 0;
            for (int d = 0; d < ndim; d++) {
                int c = coord[d];
                if (d == dim) c += out_dim_offset;
                o_flat = o_flat * out_shape[d] + c;
            }
            od[o_flat] = td[t_off];
        }
    }
}


/* ── tensor_cat ──
 *
 * Concatenate tensors a and b along dimension dim.
 *
 *   a and b must have same ndim and same shape in all dims except dim.
 *   Output shape[dim] = a->shape[dim] + b->shape[dim].
 *
 * Autograd: backward splits grad_output along dim and scatters to a, b.
 */

tensor *tensor_cat(struct mem_pool *scratch, const tensor *a, const tensor *b, int dim) {
    assert(a && b);
    assert(a->ndim == b->ndim && "tensor_cat: ndim must match");

    int ndim = a->ndim;
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "tensor_cat: dim out of range");

    for (int d = 0; d < ndim; d++) {
        if (d != dim)
            assert(a->shape[d] == b->shape[d] && "tensor_cat: shape mismatch");
    }

    int out_shape[DNN_MAX_DIMS];
    memcpy(out_shape, a->shape, ndim * sizeof(int));
    out_shape[dim] = a->shape[dim] + b->shape[dim];

    tensor *out = tensor_scratch(scratch, ndim, out_shape, 0);
    float *od = (float*)out->data;

    _cat_copy(od, a, dim, 0, out_shape);
    _cat_copy(od, b, dim, a->shape[dim], out_shape);

    /* Autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(a) || tensor_requires_grad(b))) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward  = cat_backward;
        fn->n_inputs  = 2;
        fn->inputs    = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)a;
        fn->inputs[1] = (tensor*)b;
        fn->n_saved   = 1;
        fn->saved_tensors = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        int *saved_dim = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *saved_dim = dim;
        fn->saved_tensors[0] = (tensor*)saved_dim;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
