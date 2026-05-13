#include "norm.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
#include "broadcast.h"
#include "tensor_int.h"
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── ln_backward ── */

static void ln_backward(grad_fn *fn, tensor *grad_output) {
    tensor *x      = fn->inputs[0];
    tensor *weight = (tensor*)fn->saved_tensors[0];
    tensor *bias   = (tensor*)fn->saved_tensors[1];
    float  *mean   = (float*)fn->saved_tensors[2];
    float  *rstd   = (float*)fn->saved_tensors[3];

    int ndim = x->ndim;
    int d    = x->shape[ndim - 1];
    int n    = tensor_numel(x) / d;

    float *xd = tensor_data_ptr(x);
    float *gd = (float*)grad_output->data;  /* always contiguous */
    float *wd = weight ? tensor_data_ptr(weight) : NULL;

    float inv_d = 1.0f / (float)d;

    /* dγ = sum(d_out * y, over batch dims) — y = (x - μ) * rstd */
    if (weight && tensor_requires_grad(weight)) {
        float *wg = _grad_ensure(weight);
        for (int s = 0; s < n; s++) {
            float m = mean[s], rs = rstd[s];
            #pragma omp simd
            for (int j = 0; j < d; j++) {
                float y = (xd[s * d + j] - m) * rs;
                wg[j] += gd[s * d + j] * y;
            }
        }
    }

    /* dβ = sum(d_out, over batch dims) */
    if (bias && tensor_requires_grad(bias)) {
        float *bg = _grad_ensure(bias);
        for (int s = 0; s < n; s++)
            #pragma omp simd
            for (int j = 0; j < d; j++)
                bg[j] += gd[s * d + j];
    }

    /* dx = rstd * (dy - mean(dy) - xmu * mean(dy * xmu) * rstd²) */
    if (tensor_requires_grad(x)) {
        float *xg = _grad_ensure(x);
        for (int s = 0; s < n; s++) {
            float m = mean[s], rs = rstd[s];

            float sum_dy = 0.0f, sum_dy_xmu = 0.0f;
            #pragma omp simd reduction(+:sum_dy,sum_dy_xmu)
            for (int j = 0; j < d; j++) {
                float dy = gd[s * d + j] * (wd ? wd[j] : 1.0f);
                float xmu = xd[s * d + j] - m;
                sum_dy     += dy;
                sum_dy_xmu += dy * xmu;
            }

            float mean_dy     = sum_dy * inv_d;
            float mean_dy_xmu = sum_dy_xmu * inv_d;

            #pragma omp simd
            for (int j = 0; j < d; j++) {
                float dy  = gd[s * d + j] * (wd ? wd[j] : 1.0f);
                float xmu = xd[s * d + j] - m;
                float dx  = rs * (dy - mean_dy - xmu * mean_dy_xmu * rs * rs);
                xg[s * d + j] += dx;
            }
        }
    }
}

/* ── layer_norm forward ── */

tensor *tensor_layer_norm(const tensor *x, const tensor *weight,
                          const tensor *bias, float eps) {
    assert(x);
    assert(tensor_is_contiguous(x) && "tensor_layer_norm: x must be contiguous");

    int ndim = x->ndim;
    int d    = x->shape[ndim - 1];
    int n    = tensor_numel(x) / d;  /* slices over all dims except last */

    float *xd = tensor_data_ptr((tensor*)x);

    /* mean and rstd per slice (allocated in scratch, valid until backward runs) */
    float *mean = mem_scratch_alloc(n * sizeof(float), NULL);
    float *rstd = mem_scratch_alloc(n * sizeof(float), NULL);

    /* pass 1: compute mean */
    for (int s = 0; s < n; s++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < d; j++)
            sum += xd[s * d + j];
        mean[s] = sum / (float)d;
    }

    /* pass 2: compute variance → rstd = 1/√(var + ε) */
    for (int s = 0; s < n; s++) {
        float m = mean[s];
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < d; j++) {
            float diff = xd[s * d + j] - m;
            sum += diff * diff;
        }
        rstd[s] = 1.0f / sqrtf(sum / (float)d + eps);
    }

    /* pass 3: compute output = γ * ((x - μ) * rstd) + β */
    tensor *out = _tensor_scratch_create(ndim, x->shape, 0);
    float *od   = (float*)out->data;
    float *wd   = weight ? tensor_data_ptr((tensor*)weight) : NULL;
    float *bd   = bias   ? tensor_data_ptr((tensor*)bias)   : NULL;

    for (int s = 0; s < n; s++) {
        float m = mean[s], rs = rstd[s];
        #pragma omp simd
        for (int j = 0; j < d; j++) {
            float y = (xd[s * d + j] - m) * rs;
            od[s * d + j] = y * (wd ? wd[j] : 1.0f) + (bd ? bd[j] : 0.0f);
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(x) ||
         (weight && tensor_requires_grad(weight)) ||
         (bias   && tensor_requires_grad(bias)))) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = ln_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)x;

        fn->n_saved = 4;
        fn->saved_tensors = mem_scratch_alloc(4 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)weight;  /* may be NULL */
        fn->saved_tensors[1] = (tensor*)bias;    /* may be NULL */
        fn->saved_tensors[2] = (tensor*)mean;
        fn->saved_tensors[3] = (tensor*)rstd;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
