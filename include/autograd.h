#ifndef DNN_AUTOGRAD_H
#define DNN_AUTOGRAD_H

#include "tensor.h"

struct grad_fn {
    void     (*backward)(grad_fn *fn, tensor *grad_output);
    tensor  **inputs;
    int       n_inputs;
    tensor  **saved_tensors;
    int       n_saved;
};

/* ── Backward ── */
void dnn_backward(tensor *loss);

#endif /* DNN_AUTOGRAD_H */
