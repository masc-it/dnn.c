#ifndef DNN_TENSOR_INT_H
#define DNN_TENSOR_INT_H

#include "tensor.h"

/* ── Scratch alloc for intermediate tensors (exposed for ops.c) ── */
tensor *_tensor_scratch_create(int ndim, const int *shape, int requires_grad);

#endif /* DNN_TENSOR_INT_H */
