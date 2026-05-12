#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_activation_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── relu_backward ── */

static void relu_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    int n = _numel(a->ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *ad = (float*)a->data;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        for (int i = 0; i < n; i++) {
            int off = _flat_off(a, i);
            if (ad[off] > 0.0f)
                ag[off] += g_data[i];
        }
    }
}

tensor *tensor_relu(const tensor *t) {
    assert(t);

    tensor *out = _tensor_scratch_create(t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    for (int i = 0; i < numel; i++) {
        int off = _flat_off(t, i);
        float v = td[off];
        od[out->offset + i] = v > 0.0f ? v : 0.0f;
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = relu_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

tensor *tensor_sigmoid(const tensor *t) {
    (void)t;
    return NULL;
}

/* ── softmax_backward ── */

static void softmax_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *saved_out = fn->saved_tensors[0];
    int dim = *(int*)fn->saved_tensors[1];

    int ndim = a->ndim;
    int numel = _numel(ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *sm_data = (float*)saved_out->data;
    int dim_size = a->shape[dim];
    int n_slices = numel / dim_size;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);

        /* dot = sum(sm * g, dim) for each slice */
        float *dot = mem_scratch_alloc(n_slices * sizeof(float), NULL);

        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % a->shape[d];
                r /= a->shape[d];
            }

            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= a->shape[d];
                }
            }

            dot[slice_idx] += sm_data[_flat_off(saved_out, i)] * g_data[i];
        }

        /* dL/dx_i = sm_i * (g_i - dot) */
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % a->shape[d];
                r /= a->shape[d];
            }

            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= a->shape[d];
                }
            }

            float sm_val = sm_data[_flat_off(saved_out, i)];
            int off = _flat_off(a, i);
            ag[off] += sm_val * (g_data[i] - dot[slice_idx]);
        }
    }
}

/* ── softmax forward ── */

tensor *tensor_softmax(const tensor *t, int dim) {
    assert(t);
    int ndim = t->ndim;
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "tensor_softmax: dim out of range");

    int numel = _numel(ndim, t->shape);
    int dim_size = t->shape[dim];
    int n_slices = numel / dim_size;
    float *td = (float*)t->data;

    /* per-slice max and sum of exp(x - max) */
    float *max_vals = mem_scratch_alloc(n_slices * sizeof(float), NULL);
    float *sum_vals = mem_scratch_alloc(n_slices * sizeof(float), NULL);

    for (int s = 0; s < n_slices; s++) max_vals[s] = -INFINITY;

    /* Pass 1: find max along dim for each slice */
    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % t->shape[d];
            r /= t->shape[d];
        }

        int slice_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d != dim) {
                slice_idx += coord[d] * stride;
                stride *= t->shape[d];
            }
        }

        float val = td[_bcast_off(t, ndim, coord)];
        if (val > max_vals[slice_idx]) max_vals[slice_idx] = val;
    }

    /* Pass 2: sum of exp(x - max) for each slice */
    for (int s = 0; s < n_slices; s++) sum_vals[s] = 0.0f;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % t->shape[d];
            r /= t->shape[d];
        }

        int slice_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d != dim) {
                slice_idx += coord[d] * stride;
                stride *= t->shape[d];
            }
        }

        float val = td[_bcast_off(t, ndim, coord)];
        sum_vals[slice_idx] += expf(val - max_vals[slice_idx]);
    }

    /* create output and compute softmax */
    tensor *out = _tensor_scratch_create(ndim, t->shape, 0);
    float *od = (float*)out->data;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % t->shape[d];
            r /= t->shape[d];
        }

        int slice_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d != dim) {
                slice_idx += coord[d] * stride;
                stride *= t->shape[d];
            }
        }

        float val = td[_bcast_off(t, ndim, coord)];
        od[out->offset + i] = expf(val - max_vals[slice_idx]) / sum_vals[slice_idx];
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = softmax_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 2;
        fn->saved_tensors = mem_scratch_alloc(2 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = out;
        int *dim_saved = mem_scratch_alloc(sizeof(int), NULL);
        *dim_saved = dim;
        fn->saved_tensors[1] = (tensor*)dim_saved;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── cross_entropy_backward ── */

static void cross_entropy_backward(grad_fn *fn, tensor *grad_output) {
    tensor *logits   = fn->inputs[0];
    float  *max_vals = (float*)fn->saved_tensors[0];
    float  *sum_exp  = (float*)fn->saved_tensors[1];
    tensor *target   = fn->saved_tensors[2];
    int     dim      = *(int*)fn->saved_tensors[3];
    float   inv_N    = *(float*)fn->saved_tensors[4];

    int ndim    = logits->ndim;
    int numel   = _numel(ndim, logits->shape);
    float *ld   = (float*)logits->data;
    float *gd   = (float*)grad_output->data;
    int   *td   = (int*)target->data;
    float  gout = gd[0];  /* scalar grad_output for a loss */

    if (logits->grad_fn || logits->requires_grad) {
        float *ag = _grad_ensure(logits);

        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % logits->shape[d];
                r /= logits->shape[d];
            }

            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= logits->shape[d];
                }
            }

            float val   = ld[_bcast_off(logits, ndim, coord)];
            float sm    = expf(val - max_vals[slice_idx]) / sum_exp[slice_idx];
            int is_tgt  = (coord[dim] == td[slice_idx]) ? 1 : 0;

            int off = _flat_off(logits, i);
            ag[off] += (sm - (float)is_tgt) * gout * inv_N;
        }
    }
}

/* ── cross_entropy forward ── */

tensor *tensor_cross_entropy(const tensor *logits, const tensor *target, int dim) {
    assert(logits && target);
    int ndim = logits->ndim;
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "tensor_cross_entropy: dim out of range");

    int numel    = _numel(ndim, logits->shape);
    int dim_size = logits->shape[dim];
    int n_slices = numel / dim_size;
    float *ld = (float*)logits->data;
    int   *td = (int*)target->data;

    /* per-slice max and sum of exp(x - max) */
    float *max_vals = mem_scratch_alloc(n_slices * sizeof(float), NULL);
    float *sum_exp  = mem_scratch_alloc(n_slices * sizeof(float), NULL);

    for (int s = 0; s < n_slices; s++) max_vals[s] = -INFINITY;

    /* Pass 1: find max along dim for each slice */
    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % logits->shape[d];
            r /= logits->shape[d];
        }

        int slice_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d != dim) {
                slice_idx += coord[d] * stride;
                stride *= logits->shape[d];
            }
        }

        float val = ld[_bcast_off(logits, ndim, coord)];
        if (val > max_vals[slice_idx]) max_vals[slice_idx] = val;
    }

    /* Pass 2: sum of exp(x - max) for each slice */
    for (int s = 0; s < n_slices; s++) sum_exp[s] = 0.0f;

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % logits->shape[d];
            r /= logits->shape[d];
        }

        int slice_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d != dim) {
                slice_idx += coord[d] * stride;
                stride *= logits->shape[d];
            }
        }

        float val = ld[_bcast_off(logits, ndim, coord)];
        sum_exp[slice_idx] += expf(val - max_vals[slice_idx]);
    }

    /* Pass 3: compute loss = mean(logsumexp - logits[target]) */
    float total_loss = 0.0f;
    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % logits->shape[d];
            r /= logits->shape[d];
        }

        int slice_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d != dim) {
                slice_idx += coord[d] * stride;
                stride *= logits->shape[d];
            }
        }

        if (coord[dim] == td[slice_idx]) {
            float val = ld[_bcast_off(logits, ndim, coord)];
            float lse = max_vals[slice_idx] + logf(sum_exp[slice_idx]);
            total_loss += lse - val;
        }
    }

    float inv_N = 1.0f / (float)n_slices;
    float loss_val = total_loss * inv_N;

    tensor *out = _tensor_scratch_create(1, (int[]){1}, 0);
    ((float*)out->data)[0] = loss_val;

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(logits)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = cross_entropy_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)logits;
        fn->n_saved = 5;
        fn->saved_tensors = mem_scratch_alloc(5 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)max_vals;
        fn->saved_tensors[1] = (tensor*)sum_exp;
        fn->saved_tensors[2] = (tensor*)target;
        int *dim_saved = mem_scratch_alloc(sizeof(int), NULL);
        *dim_saved = dim;
        fn->saved_tensors[3] = (tensor*)dim_saved;
        float *inv_n_saved = mem_scratch_alloc(sizeof(float), NULL);
        *inv_n_saved = inv_N;
        fn->saved_tensors[4] = (tensor*)inv_n_saved;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

tensor *tensor_tanh(const tensor *t) {
    (void)t;
    return NULL;
}
