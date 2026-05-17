#ifndef DNN_OPTIM_H
#define DNN_OPTIM_H

#include "tensor.h"

/* ── SGD with momentum ──
 *
 *   v[t+1] = momentum * v[t] + grad
 *   param  -= lr * v[t+1]
 *
 *   momentum=0 → plain SGD (param -= lr * grad)
 */

typedef struct {
    tensor  **params;
    int       n_params;
    float     lr;
    float     momentum;
    tensor  **velocities;  /* one per param, same shape, allocated in params pool */
} sgd_opt;

sgd_opt *sgd_create(struct mem_pool *params_pool,
                     tensor **params, int n_params, float lr, float momentum);
void     sgd_free(sgd_opt *opt);
void     sgd_step(sgd_opt *opt);
void     sgd_zero_grad(sgd_opt *opt);

/* ── AdamW ──
 *
 *   m_t  = beta1*m_{t-1} + (1-beta1)*g_t
 *   v_t  = beta2*v_{t-1} + (1-beta2)*g_t²
 *   m̂    = m_t / (1 - beta1^t)
 *   v̂    = v_t / (1 - beta2^t)
 *   param = param * (1 - lr*wd) - lr * m̂ / (√v̂ + eps)
 *
 *   Matches torch.optim.AdamW exactly.
 */

typedef struct adamw_opt {
    tensor  **params;
    int       n_params;
    float     lr;
    float     beta1, beta2;
    float     eps;
    float     weight_decay;
    int       t;           /* step counter, incremented each step() */
    tensor  **m, **v;      /* first and second moment buffers */
} adamw_opt;

adamw_opt *adamw_create(struct mem_pool *params_pool,
                         tensor **params, int n_params, float lr,
                         float beta1, float beta2, float eps, float weight_decay);
void       adamw_free(adamw_opt *opt);
void       adamw_step(adamw_opt *opt);
/* AdamW step with per-parameter LR multipliers. lr_mults[i] scales opt->lr
 * for opt->params[i]. Pass 1.0 for normal LR. Weight decay is scaled by the
 * same effective group LR, matching param-group AdamW semantics. */
void       adamw_step_with_lr_multipliers(adamw_opt *opt, const float *lr_mults);
void       adamw_zero_grad(adamw_opt *opt);

/* ── Gradient Clipping ── */

/* Clip gradient norm (L2) across all params.
 *
 *   Computes total L2 norm of all gradients across all params.
 *   If total_norm > max_norm, scales all gradients by max_norm / total_norm.
 *
 *   Returns the total norm BEFORE clipping (for logging).
 *   If max_norm <= 0, no-op, returns 0.
 */
float clip_grad_norm(tensor **params, int n_params, float max_norm);

/* Per-tensor gradient clipping: clip each tensor's gradient norm
 * independently to max_norm.  Prevents a single dominant param group
 * (e.g. conv2d weight summing over spatial positions) from starving all
 * other groups.  Returns the pre-clipping global norm (for logging).
 */
float clip_grad_norm_per_tensor(tensor **params, int n_params, float max_norm);

/* Compute L2 gradient norm across all params (for logging without clipping). */
float grad_norm(tensor **params, int n_params);

/* Clip gradient values element-wise to [-clip_value, clip_value].
 *
 *   If clip_value <= 0, no-op.
 */
void clip_grad_value(tensor **params, int n_params, float clip_value);

/* ── Learning Rate Scheduler ──
 *
 * Supported schedule types:
 *
 *   LR_SCHEDULE_CONSTANT              — fixed LR (base_lr), no change
 *   LR_SCHEDULE_LINEAR_WARMUP_COSINE  — linear warmup from 0 to base_lr
 *                                       over warmup_iters, then cosine decay
 *                                       from base_lr to min_lr over remaining
 *                                       total_iters steps.
 *   LR_SCHEDULE_LINEAR_WARMUP         — linear warmup from 0 to base_lr
 *                                       over warmup_iters, then constant.
 *   LR_SCHEDULE_COSINE                — cosine decay from base_lr to min_lr
 *                                       over total_iters (no warmup).
 *   LR_SCHEDULE_STEP                  — multiply LR by gamma every step_size iters.
 *   LR_SCHEDULE_EXPONENTIAL           — multiply LR by gamma each step.
 *
 * Usage:
 *   lr_scheduler *sched = lr_scheduler_create(opt, LR_SCHEDULE_LINEAR_WARMUP_COSINE,
 *                                              base_lr, warmup_iters, total_iters, min_lr);
 *   for each training step:
 *       train_step(...)
 *       lr_scheduler_step(sched);
 */

#define LR_SCHEDULE_CONSTANT              0
#define LR_SCHEDULE_LINEAR_WARMUP_COSINE  1
#define LR_SCHEDULE_LINEAR_WARMUP         2
#define LR_SCHEDULE_COSINE                3
#define LR_SCHEDULE_STEP                  4
#define LR_SCHEDULE_EXPONENTIAL           5

/* Forward declaration (struct tag matches adamw_opt typedef) */
struct adamw_opt;

typedef struct {
    struct adamw_opt *opt;
    int        schedule;
    float      base_lr;
    float      min_lr;
    int        warmup_iters;
    int        total_iters;  /* total training steps (cosine period) */
    int        step_size;    /* step interval for step decay */
    float      gamma;        /* multiplier for step/exponential decay */
    int        t;            /* current step count */
} lr_scheduler;

/* Create LR scheduler.
 *
 *   opt        — optimizer whose lr will be adjusted
 *   schedule   — one of LR_SCHEDULE_*
 *   base_lr    — peak learning rate (after warmup, if any)
 *   warmup_iters — number of linear warmup steps (0 = no warmup)
 *   total_iters  — total training steps (for cosine period)
 *   min_lr       — minimum LR (cosine target). If < 0, defaults to base_lr/10.
 *   step_size    — step interval for step decay (ignored for other schedules)
 *   gamma        — multiplier for step/exponential decay (ignored for others)
 *
 * Stores a pointer to the optimizer; does NOT own it.
 */
lr_scheduler *lr_scheduler_create(struct mem_pool *params_pool,
                                   struct adamw_opt *opt, int schedule,
                                   float base_lr,
                                   int warmup_iters, int total_iters,
                                   float min_lr,
                                   int step_size, float gamma);

/* Advance one step: increment counter, recompute LR, update opt->lr. */
void lr_scheduler_step(lr_scheduler *sched);

/* Get current learning rate (without stepping). */
float lr_scheduler_get_lr(const lr_scheduler *sched);

/* Reset scheduler back to step 0. */
void lr_scheduler_reset(lr_scheduler *sched);

/* Destroy scheduler (no-op, pool-managed). */
void lr_scheduler_destroy(lr_scheduler *sched);

#endif /* DNN_OPTIM_H */
