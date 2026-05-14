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

/* ── SwiGLU FFN ── */

swiglu_ffn *swiglu_ffn_create(int d_model, int intermediate_size) {
    assert(d_model > 0 && intermediate_size > 0);

    swiglu_ffn *ffn = mem_params_alloc(sizeof(swiglu_ffn), NULL);
    ffn->d_model          = d_model;
    ffn->intermediate_size = intermediate_size;
    ffn->gate_proj = linear_create(d_model, intermediate_size);
    ffn->up_proj   = linear_create(d_model, intermediate_size);
    ffn->down_proj = linear_create(intermediate_size, d_model);
    return ffn;
}

tensor *swiglu_ffn_forward(swiglu_ffn *ffn, const tensor *x) {
    assert(ffn);
    assert(x);
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == ffn->d_model);

    /* gate = SiLU(x @ W_g + b_g) */
    tensor *gate = tensor_silu(linear_forward(ffn->gate_proj, x));

    /* up = x @ W_u + b_u */
    tensor *up = linear_forward(ffn->up_proj, x);

    /* hidden = gate ⊗ up */
    tensor *hidden = tensor_mul(gate, up);

    /* out = hidden @ W_d + b_d */
    return linear_forward(ffn->down_proj, hidden);
}
