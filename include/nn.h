#ifndef DNN_NN_H
#define DNN_NN_H

#include "tensor.h"
#include "module.h"

/* ── Linear (fully-connected) layer ──
 *
 *   y = x @ W + b
 *
 *   weight shape: [in_features, out_features]
 *   bias   shape: [out_features]
 */
typedef struct linear {
    module  base;           /* first field — cast (module*)linear safely */
    tensor *weight;
    tensor *bias;
    int     in_features;
    int     out_features;
} linear;

linear  *linear_create(struct mem_pool *params, int in_features, int out_features);
tensor  *linear_forward(struct mem_pool *scratch, linear *l, const tensor *input);

/* ── SwiGLU FFN block ──
 *
 *   SwiGLU(x) = (SiLU(x @ W_g + b_g) ⊗ (x @ W_u + b_u)) @ W_d + b_d
 *
 *   gate_proj  — linear(d_model, intermediate_size)
 *   up_proj    — linear(d_model, intermediate_size)
 *   down_proj  — linear(intermediate_size, d_model)
 *
 *   Matches Llama/Mistral FFN block.
 */
typedef struct swiglu_ffn {
    module  base;         /* first field */
    linear *gate_proj;    /* siLU-gated projection */
    linear *up_proj;      /* up projection */
    linear *down_proj;    /* down projection */
    int     d_model;
    int     intermediate_size;
} swiglu_ffn;

swiglu_ffn *swiglu_ffn_create(struct mem_pool *params_pool, int d_model, int intermediate_size);
tensor     *swiglu_ffn_forward(struct mem_pool *scratch, swiglu_ffn *ffn, const tensor *x);

/* ── LayerNorm ──
 *
 *   y = γ * (x - μ) / √(σ² + ε) + β
 *
 *   weight — [d], init 1
 *   bias   — [d], init 0
 *   eps    — small constant for numerical stability
 */
typedef struct layer_norm {
    module  base;
    tensor *weight;
    tensor *bias;
    float   eps;
    int     d;
} layer_norm;

layer_norm *layer_norm_create(struct mem_pool *params, int d, float eps);
tensor     *layer_norm_forward(struct mem_pool *scratch, layer_norm *ln, const tensor *x);

/* ── RMS Normalization (no bias) ──
 *
 *   y = x * rsqrt(mean(x²) + eps) * weight
 *
 *   weight — [d], init 1
 *   eps    — small constant
 */
typedef struct rms_norm {
    module  base;
    tensor *weight;
    float   eps;
    int     d;
} rms_norm;

rms_norm *rms_norm_create(struct mem_pool *params, int d, float eps);
tensor   *rms_norm_forward(struct mem_pool *scratch, rms_norm *rn, const tensor *x);

/* ── Embedding ──
 *
 *   out = table[ids]
 *
 *   weight — [vocab_size, d_model], uniform init
 */
typedef struct embedding {
    module  base;
    tensor *weight;
    int     vocab_size;
    int     d_model;
} embedding;

embedding *embedding_create(struct mem_pool *params, int vocab_size, int d_model);
tensor    *embedding_forward(struct mem_pool *scratch, struct mem_pool *data,
                             embedding *emb, const tensor *ids);

/* ── Conv2D layer ──
 *
 *   y = conv2d(x, weight, bias, stride, padding)
 *
 *   weight shape: [out_channels, in_channels, kH, kW]
 *   bias   shape: [out_channels]
 */
typedef struct conv2d {
    module  base;
    tensor *weight;
    tensor *bias;
    int     in_channels;
    int     out_channels;
    int     kernel_size;
    int     stride;
    int     padding;
} conv2d;

conv2d  *conv2d_create(struct mem_pool *params, int in_channels, int out_channels,
                       int kernel_size, int stride, int padding);
tensor  *conv2d_forward(struct mem_pool *scratch, conv2d *c, const tensor *input);

/* ── Parameter count helpers ── */
long long linear_num_parameters(linear *l);
long long swiglu_ffn_num_parameters(swiglu_ffn *ffn);
long long layer_norm_num_parameters(layer_norm *ln);
long long embedding_num_parameters(embedding *emb);

#endif /* DNN_NN_H */
