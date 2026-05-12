#ifndef DNN_NN_H
#define DNN_NN_H

#include "tensor.h"

/* ── Linear (fully-connected) layer ──
 *
 *   y = x @ W + b
 *
 *   weight shape: [in_features, out_features]
 *   bias   shape: [out_features]
 */
typedef struct linear {
    tensor *weight;
    tensor *bias;
    int     in_features;
    int     out_features;
} linear;

linear  *linear_create(int in_features, int out_features);
tensor  *linear_forward(linear *l, const tensor *input);

#endif /* DNN_NN_H */
