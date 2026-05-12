#ifndef DNN_AUTOGRAD_INT_H
#define DNN_AUTOGRAD_INT_H

#include "autograd.h"

/* ── Gradient function factory (exposed for ops.c) ── */
grad_fn *_grad_fn_create(void);

/* ── Gradient buffer ensure (exposed for ops.c) ── */
float   *_grad_ensure(tensor *t);

#endif /* DNN_AUTOGRAD_INT_H */
