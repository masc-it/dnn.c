#ifndef DNN_OPS_H
#define DNN_OPS_H

#include "tensor.h"

/* ── Arithmetic ── */
tensor *tensor_add(const tensor *a, const tensor *b);
tensor *tensor_sub(const tensor *a, const tensor *b);
tensor *tensor_mul(const tensor *a, const tensor *b);  /* element-wise */
tensor *tensor_div(const tensor *a, const tensor *b);

/* ── Matrix ops ── */
tensor *tensor_matmul(const tensor *a, const tensor *b);

/* ── Activations ── */
tensor *tensor_relu(const tensor *t);
tensor *tensor_sigmoid(const tensor *t);
tensor *tensor_tanh(const tensor *t);
tensor *tensor_softmax(const tensor *t, int dim);

/* ── Reduction ── */
tensor *tensor_sum(const tensor *t, int dim);
tensor *tensor_mean(const tensor *t, int dim);

/* ── Loss ── */
tensor *tensor_cross_entropy(const tensor *logits, const tensor *target, int dim);

/* ── Utility ── */
tensor *tensor_pow(const tensor *t, float exp);
tensor *tensor_neg(const tensor *t);

#endif /* DNN_OPS_H */
