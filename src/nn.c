#include "nn.h"
#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include <assert.h>
#include <math.h>

linear *linear_create(int in_features, int out_features) {
    assert(in_features > 0 && out_features > 0);

    linear *l = mem_params_alloc(sizeof(linear), NULL);
    l->in_features  = in_features;
    l->out_features = out_features;
    float bound = 1.0f / sqrtf((float)in_features);
    l->weight = tensor_uniform(2, (int[]){in_features, out_features}, 1, bound);
    l->bias   = tensor_uniform(1, (int[]){out_features}, 1, bound);
    return l;
}

tensor *linear_forward(linear *l, const tensor *input) {
    assert(l);
    assert(input);
    assert(input->ndim >= 2 && "linear_forward: input must be at least 2D");
    assert(input->shape[input->ndim - 1] == l->in_features
           && "linear_forward: last dim of input must match in_features");

    /* x @ W + b */
    tensor *mm = tensor_matmul(input, l->weight);
    return tensor_add(mm, l->bias);
}
