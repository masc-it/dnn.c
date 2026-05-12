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

sgd_opt *sgd_create(tensor **params, int n_params, float lr, float momentum);
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

typedef struct {
    tensor  **params;
    int       n_params;
    float     lr;
    float     beta1, beta2;
    float     eps;
    float     weight_decay;
    int       t;           /* step counter, incremented each step() */
    tensor  **m, **v;      /* first and second moment buffers */
} adamw_opt;

adamw_opt *adamw_create(tensor **params, int n_params, float lr,
                         float beta1, float beta2, float eps, float weight_decay);
void       adamw_free(adamw_opt *opt);
void       adamw_step(adamw_opt *opt);
void       adamw_zero_grad(adamw_opt *opt);

#endif /* DNN_OPTIM_H */
