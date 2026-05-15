#ifndef DNN_AUTOGRAD_H
#define DNN_AUTOGRAD_H

#include "tensor.h"

struct grad_fn {
    void     (*backward)(grad_fn *fn, tensor *grad_output);
    struct mem_pool *pool;  /* scratch pool — used for temp allocs in backward */
    tensor  **inputs;
    int       n_inputs;
    tensor  **saved_tensors;
    int       n_saved;
};

/* ── Grad mode context ── */
typedef struct { int _prev; } dnn_grad_ctx;

int          dnn_grad_enabled(void);
dnn_grad_ctx dnn_no_grad_enter(void);
void         dnn_no_grad_exit(dnn_grad_ctx ctx);

/* ── Backward ── */
void dnn_backward(struct mem_pool *scratch, tensor *loss);

#endif /* DNN_AUTOGRAD_H */
