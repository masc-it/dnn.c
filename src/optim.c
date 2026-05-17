#include "optim.h"
#include "pool.h"
#include <assert.h>
#include <string.h>
#include <math.h>

sgd_opt *sgd_create(struct mem_pool *params_pool, tensor **params, int n_params, float lr, float momentum) {
    sgd_opt *opt = _mem_pool_alloc(params_pool, sizeof(sgd_opt), NULL);
    opt->params   = params;
    opt->n_params = n_params;
    opt->lr       = lr;
    opt->momentum = momentum;

    /* allocate velocity buffers (one per param, same shape, zero-initialized) */
    opt->velocities = _mem_pool_alloc(params_pool, n_params * sizeof(tensor*), NULL);
    for (int i = 0; i < n_params; i++) {
        tensor *p = params[i];
        tensor *v = tensor_zeros(params_pool, p->ndim, p->shape, 0);
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

adamw_opt *adamw_create(struct mem_pool *params_pool, tensor **params, int n_params, float lr,
                         float beta1, float beta2, float eps, float weight_decay) {
    adamw_opt *opt = _mem_pool_alloc(params_pool, sizeof(adamw_opt), NULL);
    opt->params   = params;
    opt->n_params = n_params;
    opt->lr       = lr;
    opt->beta1    = beta1;
    opt->beta2    = beta2;
    opt->eps      = eps;
    opt->weight_decay = weight_decay;
    opt->t        = 0;
    opt->m        = _mem_pool_alloc(params_pool, n_params * sizeof(tensor*), NULL);
    opt->v        = _mem_pool_alloc(params_pool, n_params * sizeof(tensor*), NULL);

    for (int i = 0; i < n_params; i++) {
        tensor *p = params[i];
        opt->m[i] = tensor_zeros(params_pool, p->ndim, p->shape, 0);
        opt->v[i] = tensor_zeros(params_pool, p->ndim, p->shape, 0);
    }
    return opt;
}

void adamw_free(adamw_opt *opt) {
    (void)opt;
}

void adamw_zero_grad(adamw_opt *opt) {
#pragma omp parallel for schedule(dynamic) if (opt->n_params >= 4)
    for (int i = 0; i < opt->n_params; i++) {
        float *g = tensor_grad(opt->params[i]);
        if (g)
            memset(g, 0, tensor_numel(opt->params[i]) * sizeof(float));
    }
}

/* ── LR Scheduler ── */

static float _lr_compute(const lr_scheduler *sched, int t) {
    int schedule = sched->schedule;
    float base   = sched->base_lr;
    float min_lr = sched->min_lr;
    int warmup   = sched->warmup_iters;
    int total    = sched->total_iters;
    int step_sz  = sched->step_size;
    float gamma  = sched->gamma;

    /* Linear warmup phase */
    if (warmup > 0 && t < warmup) {
        return base * (float)(t + 1) / (float)warmup;
    }

    /* Post-warmup: adjust t relative to warmup end */
    int post_t = t - warmup;
    if (post_t < 0) post_t = 0;

    switch (schedule) {
    case LR_SCHEDULE_CONSTANT:
    case LR_SCHEDULE_LINEAR_WARMUP:
        return base;

    case LR_SCHEDULE_COSINE:
    case LR_SCHEDULE_LINEAR_WARMUP_COSINE: {
        /* Cosine decay from base to min_lr over (total - warmup) steps */
        int decay_iters = total - warmup;
        if (decay_iters <= 0) return base;  /* no decay period */
        float cosine_decay = 0.5f * (1.0f + cosf((float)M_PI * post_t / decay_iters));
        return min_lr + (base - min_lr) * cosine_decay;
    }

    case LR_SCHEDULE_STEP:
        if (step_sz <= 0) return base;
        return base * powf(gamma, (float)(post_t / step_sz));

    case LR_SCHEDULE_EXPONENTIAL:
        if (post_t == 0) return base;
        return base * powf(gamma, (float)post_t);

    default:
        return base;
    }
}

lr_scheduler *lr_scheduler_create(struct mem_pool *params_pool, adamw_opt *opt, int schedule,
                                   float base_lr,
                                   int warmup_iters, int total_iters,
                                   float min_lr,
                                   int step_size, float gamma) {
    lr_scheduler *sched = _mem_pool_alloc(params_pool, sizeof(lr_scheduler), NULL);
    sched->opt          = opt;
    sched->schedule     = schedule;
    sched->base_lr      = base_lr;
    sched->warmup_iters = warmup_iters;
    sched->total_iters  = total_iters;
    sched->step_size    = step_size;
    sched->gamma        = gamma;
    sched->t            = 0;

    /* min_lr default: base_lr/10 if negative */
    sched->min_lr = (min_lr < 0.0f) ? base_lr / 10.0f : min_lr;

    /* Set initial LR on optimizer */
    opt->lr = _lr_compute(sched, 0);

    return sched;
}

void lr_scheduler_step(lr_scheduler *sched) {
    sched->t++;
    sched->opt->lr = _lr_compute(sched, sched->t);
}

float lr_scheduler_get_lr(const lr_scheduler *sched) {
    return _lr_compute(sched, sched->t);
}

void lr_scheduler_reset(lr_scheduler *sched) {
    sched->t = 0;
    sched->opt->lr = _lr_compute(sched, 0);
}

void lr_scheduler_destroy(lr_scheduler *sched) {
    (void)sched;
}

/* ── Gradient Clipping ── */

float grad_norm(tensor **params, int n_params) {
    double total_norm_sq = 0.0;
#pragma omp parallel for reduction(+:total_norm_sq) schedule(dynamic) if (n_params >= 4)
    for (int i = 0; i < n_params; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) continue;
        int n = tensor_numel(params[i]);
#pragma omp simd reduction(+:total_norm_sq)
        for (int j = 0; j < n; j++)
            total_norm_sq += (double)g[j] * (double)g[j];
    }
    return sqrtf((float)total_norm_sq);
}

float clip_grad_norm(tensor **params, int n_params, float max_norm) {
    if (max_norm <= 0.0f) return 0.0f;

    float total_norm = grad_norm(params, n_params);

    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
#pragma omp parallel for schedule(dynamic) if (n_params >= 4)
        for (int i = 0; i < n_params; i++) {
            float *g = tensor_grad(params[i]);
            if (!g) continue;
            int n = tensor_numel(params[i]);
#pragma omp simd
            for (int j = 0; j < n; j++)
                g[j] *= scale;
        }
    }

    return total_norm;
}

float clip_grad_norm_per_tensor(tensor **params, int n_params, float max_norm) {
    if (max_norm <= 0.0f) return 0.0f;
    float total_norm = grad_norm(params, n_params);
    for (int i = 0; i < n_params; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) continue;
        int n = tensor_numel(params[i]);
        double tn_sq = 0.0;
        for (int j = 0; j < n; j++)
            tn_sq += (double)g[j] * (double)g[j];
        float tn = sqrtf((float)tn_sq);
        if (tn > max_norm) {
            float scale = max_norm / tn;
            for (int j = 0; j < n; j++)
                g[j] *= scale;
        }
    }
    return total_norm;
}

void clip_grad_value(tensor **params, int n_params, float clip_value) {
    if (clip_value <= 0.0f) return;

    for (int i = 0; i < n_params; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) continue;
        int n = tensor_numel(params[i]);
        for (int j = 0; j < n; j++) {
            if (g[j] > clip_value)      g[j] =  clip_value;
            else if (g[j] < -clip_value) g[j] = -clip_value;
        }
    }
}

void adamw_step_with_lr_multipliers(adamw_opt *opt, const float *lr_mults) {
    opt->t++;
    float base_lr = opt->lr;
    float b1      = opt->beta1;
    float b2      = opt->beta2;
    float eps     = opt->eps;
    float wd      = opt->weight_decay;

    float bias_corr1 = 1.0f - powf(b1, (float)opt->t);
    float bias_corr2 = 1.0f - powf(b2, (float)opt->t);
    float bias_corr2_sqrt = sqrtf(bias_corr2);

#pragma omp parallel for schedule(dynamic) if (opt->n_params >= 4)
    for (int i = 0; i < opt->n_params; i++) {
        tensor *p = opt->params[i];
        tensor *mt = opt->m[i];
        tensor *vt = opt->v[i];

        int n     = tensor_numel(p);
        float *pd = tensor_data_ptr(p);
        float *md = tensor_data_ptr(mt);
        float *vd = tensor_data_ptr(vt);
        float *gd = tensor_grad(p);

        assert(gd && "adamw_step: param has no gradient (did you call backward?)");

        float mult = lr_mults ? lr_mults[i] : 1.0f;
        float lr = base_lr * mult;
        float step_size = lr / bias_corr1;
#pragma omp simd
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
            float denom = sqrtf(vd[j]) / bias_corr2_sqrt + eps;

            /* decoupled weight decay then Adam update */
            pd[j] *= 1.0f - lr * wd;
            pd[j] -= step_size * md[j] / denom;
        }
    }
}

void adamw_step(adamw_opt *opt) {
    adamw_step_with_lr_multipliers(opt, NULL);
}
