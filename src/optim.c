#include "optim.h"
#include "pool.h"
#include "tensor_int.h"
#include <assert.h>
#include <string.h>
#include <math.h>

sgd_opt *sgd_create(tensor **params, int n_params, float lr, float momentum) {
    sgd_opt *opt = mem_params_alloc(sizeof(sgd_opt), NULL);
    opt->params   = params;
    opt->n_params = n_params;
    opt->lr       = lr;
    opt->momentum = momentum;

    /* allocate velocity buffers (one per param, same shape, zero-initialized) */
    opt->velocities = mem_params_alloc(n_params * sizeof(tensor*), NULL);
    for (int i = 0; i < n_params; i++) {
        tensor *p = params[i];
        tensor *v = tensor_zeros(p->ndim, p->shape, 0);
        opt->velocities[i] = v;
    }

    return opt;
}

void sgd_free(sgd_opt *opt) {
    /* all memory is pool-managed; no individual frees needed */
    (void)opt;
}

void sgd_zero_grad(sgd_opt *opt) {
    for (int i = 0; i < opt->n_params; i++) {
        float *g = tensor_grad(opt->params[i]);
        if (g)
            memset(g, 0, tensor_numel(opt->params[i]) * sizeof(float));
    }
}

void sgd_step(sgd_opt *opt) {
    float lr       = opt->lr;
    float momentum = opt->momentum;

    for (int i = 0; i < opt->n_params; i++) {
        tensor *p = opt->params[i];
        tensor *v = opt->velocities[i];

        int n    = tensor_numel(p);
        float *pd = tensor_data_ptr(p);
        float *vd = tensor_data_ptr(v);
        float *gd = tensor_grad(p);

        assert(gd && "sgd_step: param has no gradient (did you call backward?)");

        if (momentum == 0.0f) {
            for (int j = 0; j < n; j++)
                pd[j] -= lr * gd[j];
        } else {
            for (int j = 0; j < n; j++) {
                vd[j] = momentum * vd[j] + gd[j];
                pd[j] -= lr * vd[j];
            }
        }
    }
}

/* ── AdamW ── */

adamw_opt *adamw_create(tensor **params, int n_params, float lr,
                         float beta1, float beta2, float eps, float weight_decay) {
    adamw_opt *opt = mem_params_alloc(sizeof(adamw_opt), NULL);
    opt->params   = params;
    opt->n_params = n_params;
    opt->lr       = lr;
    opt->beta1    = beta1;
    opt->beta2    = beta2;
    opt->eps      = eps;
    opt->weight_decay = weight_decay;
    opt->t        = 0;
    opt->m        = mem_params_alloc(n_params * sizeof(tensor*), NULL);
    opt->v        = mem_params_alloc(n_params * sizeof(tensor*), NULL);

    for (int i = 0; i < n_params; i++) {
        tensor *p = params[i];
        opt->m[i] = tensor_zeros(p->ndim, p->shape, 0);
        opt->v[i] = tensor_zeros(p->ndim, p->shape, 0);
    }
    return opt;
}

void adamw_free(adamw_opt *opt) {
    (void)opt;
}

void adamw_zero_grad(adamw_opt *opt) {
    for (int i = 0; i < opt->n_params; i++) {
        float *g = tensor_grad(opt->params[i]);
        if (g)
            memset(g, 0, tensor_numel(opt->params[i]) * sizeof(float));
    }
}

void adamw_step(adamw_opt *opt) {
    opt->t++;
    float lr  = opt->lr;
    float b1  = opt->beta1;
    float b2  = opt->beta2;
    float eps = opt->eps;
    float wd  = opt->weight_decay;

    float bias_corr1 = 1.0f - powf(b1, (float)opt->t);
    float bias_corr2 = 1.0f - powf(b2, (float)opt->t);

    for (int i = 0; i < opt->n_params; i++) {
        tensor *p = opt->params[i];
        tensor *mt = opt->m[i];
        tensor *vt = opt->v[i];

        int n    = tensor_numel(p);
        float *pd = tensor_data_ptr(p);
        float *md = tensor_data_ptr(mt);
        float *vd = tensor_data_ptr(vt);
        float *gd = tensor_grad(p);

        assert(gd && "adamw_step: param has no gradient (did you call backward?)");

        for (int j = 0; j < n; j++) {
            float g = gd[j];

            /* update biased moments */
            md[j] = b1 * md[j] + (1.0f - b1) * g;
            vd[j] = b2 * vd[j] + (1.0f - b2) * g * g;

            /* step_size and denom match PyTorch order:
               step_size = lr / bias_corr1
               denom     = sqrt(v) / sqrt(bias_corr2) + eps
               param    -= step_size * m / denom
            */
            float step_size = lr / bias_corr1;
            float denom     = sqrtf(vd[j]) / sqrtf(bias_corr2) + eps;

            /* decoupled weight decay then Adam update */
            pd[j] *= 1.0f - lr * wd;
            pd[j] -= step_size * md[j] / denom;
        }
    }
}
