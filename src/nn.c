#include "nn.h"
#include "ops.h"
#include "norm.h"
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

    /* x @ W + b  (fused — avoids intermediate tensor) */
    return tensor_matmul_add(scratch, input, l->weight, 0, l->bias);
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

/* ── LayerNorm ── */

layer_norm *layer_norm_create(struct mem_pool *params, int d, float eps) {
    assert(d > 0);

    layer_norm *ln = _mem_pool_alloc(params, sizeof(layer_norm), NULL);
    module_init(&ln->base, params, "layer_norm");

    ln->d   = d;
    ln->eps = eps;

    ln->weight = tensor_zeros(params, 1, (int[]){d}, 1);
    ln->bias   = tensor_zeros(params, 1, (int[]){d}, 1);
    float *wd = tensor_data_ptr(ln->weight);
    for (int i = 0; i < d; i++) wd[i] = 1.0f;

    module_param(&ln->base, "weight", ln->weight);
    module_param(&ln->base, "bias",   ln->bias);
    return ln;
}

tensor *layer_norm_forward(struct mem_pool *scratch, layer_norm *ln, const tensor *x) {
    assert(ln);
    assert(x);
    return tensor_layer_norm(scratch, x, ln->weight, ln->bias, ln->eps);
}

long long layer_norm_num_parameters(layer_norm *ln) {
    return module_num_parameters(&ln->base);
}

/* ── Embedding ── */

embedding *embedding_create(struct mem_pool *params, int vocab_size, int d_model) {
    assert(vocab_size > 0 && d_model > 0);

    embedding *emb = _mem_pool_alloc(params, sizeof(embedding), NULL);
    module_init(&emb->base, params, "embedding");

    emb->vocab_size = vocab_size;
    emb->d_model    = d_model;

    float bound = 1.0f / sqrtf((float)d_model);
    emb->weight = tensor_uniform(params, 2, (int[]){vocab_size, d_model}, 1, bound);

    module_param(&emb->base, "weight", emb->weight);
    return emb;
}

tensor *embedding_forward(struct mem_pool *scratch, struct mem_pool *data,
                           embedding *emb, const tensor *ids) {
    assert(emb);
    assert(ids);
    (void)data;
    return tensor_embedding(scratch, emb->weight, ids);
}

long long embedding_num_parameters(embedding *emb) {
    return module_num_parameters(&emb->base);
}
