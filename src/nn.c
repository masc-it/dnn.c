#include "nn.h"
#include "ops.h"
#include "conv.h"
#include "autograd.h"
#include "pool.h"
#include <assert.h>
#include <math.h>

linear *linear_create(struct mem_pool *params, int in_features, int out_features) {
    assert(in_features > 0 && out_features > 0);

    linear *l = _mem_pool_alloc(params, sizeof(linear), NULL);
    module_init(&l->base, params, "linear");

    l->in_features  = in_features;
    l->out_features = out_features;
    float bound = 1.0f / sqrtf((float)in_features);
    l->weight = tensor_uniform(params, 2, (int[]){in_features, out_features}, 1, bound);
    l->bias   = tensor_uniform(params, 1, (int[]){out_features}, 1, bound);

    module_param(&l->base, "weight", l->weight);
    module_param(&l->base, "bias",   l->bias);
    return l;
}

tensor *linear_forward(struct mem_pool *scratch, linear *l, const tensor *input) {
    assert(l);
    assert(input);
    assert(input->ndim >= 2 && "linear_forward: input must be at least 2D");
    assert(input->shape[input->ndim - 1] == l->in_features
           && "linear_forward: last dim of input must match in_features");

    /* x @ W + b */
    tensor *mm = tensor_matmul(scratch, input, l->weight);
    return tensor_add(scratch, mm, l->bias);
}

/* ── SwiGLU FFN ── */

swiglu_ffn *swiglu_ffn_create(struct mem_pool *params, int d_model, int intermediate_size) {
    assert(d_model > 0 && intermediate_size > 0);

    swiglu_ffn *ffn = _mem_pool_alloc(params, sizeof(swiglu_ffn), NULL);
    module_init(&ffn->base, params, "swiglu_ffn");

    ffn->d_model          = d_model;
    ffn->intermediate_size = intermediate_size;

    ffn->gate_proj = linear_create(params, d_model, intermediate_size);
    module_add_child(&ffn->base, "gate_proj", &ffn->gate_proj->base);

    ffn->up_proj = linear_create(params, d_model, intermediate_size);
    module_add_child(&ffn->base, "up_proj", &ffn->up_proj->base);

    ffn->down_proj = linear_create(params, intermediate_size, d_model);
    module_add_child(&ffn->base, "down_proj", &ffn->down_proj->base);

    return ffn;
}

tensor *swiglu_ffn_forward(struct mem_pool *scratch, swiglu_ffn *ffn, const tensor *x) {
    assert(ffn);
    assert(x);
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == ffn->d_model);

    /* gate = x @ W_g + b_g  (raw, before activation) */
    tensor *gate = linear_forward(scratch, ffn->gate_proj, x);

    /* up = x @ W_u + b_u */
    tensor *up = linear_forward(scratch, ffn->up_proj, x);

    /* hidden = SiLU(gate) ⊗ up  (fused single-pass) */
    tensor *hidden = tensor_swiglu(scratch, gate, up);

    /* out = hidden @ W_d + b_d */
    return linear_forward(scratch, ffn->down_proj, hidden);
}

long long linear_num_parameters(linear *l) {
    return module_num_parameters(&l->base);
}

long long swiglu_ffn_num_parameters(swiglu_ffn *ffn) {
    return module_num_parameters(&ffn->base);
}

/* ── Conv2D ── */

conv2d *conv2d_create(struct mem_pool *params, int in_channels, int out_channels,
                       int kernel_size, int stride, int padding) {
    assert(in_channels > 0 && out_channels > 0 && kernel_size > 0);

    conv2d *c = _mem_pool_alloc(params, sizeof(conv2d), NULL);
    module_init(&c->base, params, "conv2d");

    c->in_channels  = in_channels;
    c->out_channels = out_channels;
    c->kernel_size  = kernel_size;
    c->stride       = stride;
    c->padding      = padding;

    float bound = sqrtf(6.0f / (float)(in_channels * kernel_size * kernel_size));
    c->weight = tensor_uniform(params, 4, (int[]){out_channels, in_channels, kernel_size, kernel_size}, 1, bound);
    c->bias   = tensor_zeros(params, 1, (int[]){out_channels}, 1);

    module_param(&c->base, "weight", c->weight);
    module_param(&c->base, "bias",   c->bias);
    return c;
}

tensor *conv2d_forward(struct mem_pool *scratch, conv2d *c, const tensor *input) {
    assert(c);
    assert(input);
    return tensor_conv2d(scratch, (tensor*)input, c->weight, c->bias,
                          c->stride, c->padding);
}
