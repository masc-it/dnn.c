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
    linear *gate_proj;   /* siLU-gated projection */
    linear *up_proj;     /* up projection */
    linear *down_proj;   /* down projection */
    int     d_model;
    int     intermediate_size;
} swiglu_ffn;

swiglu_ffn *swiglu_ffn_create(int d_model, int intermediate_size);
tensor     *swiglu_ffn_forward(swiglu_ffn *ffn, const tensor *x);

/* ── Parameter count helpers ── */
long long linear_num_parameters(linear *l);
long long swiglu_ffn_num_parameters(swiglu_ffn *ffn);

#endif /* DNN_NN_H */
