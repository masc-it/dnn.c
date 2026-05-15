#include "transformer.h"
#include "nn.h"
#include "norm.h"
#include "ops.h"
#include "attention.h"
#include "multihead.h"
#include "rope.h"
#include "pool.h"
#include "autograd.h"
#include "optim.h"
#include "tokenizer.h"
#include "simd.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* BLAS for fast matmul in cached attention */
#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#elif defined(HAVE_CBLAS)
#  include <cblas.h>
#else
#  define NO_CBLAS 1
#endif

/* ── KV Cache ── */

kv_cache *kv_cache_create(struct mem_pool *params_pool, int B, int H, int max_seq, int d_k) {
    kv_cache *kvc = _mem_pool_alloc(params_pool, sizeof(kv_cache), NULL);
    int shape[4] = {B, H, max_seq, d_k};
    kvc->k_cache = tensor_zeros(params_pool, 4, shape, 0);
    kvc->v_cache = tensor_zeros(params_pool, 4, shape, 0);
    kvc->seq_len = 0;
    kvc->max_seq = max_seq;
    return kvc;
}

void kv_cache_append(kv_cache *kvc, const tensor *K_new, const tensor *V_new) {
    assert(kvc && "kv_cache_append: NULL cache");
    assert(K_new && V_new && "kv_cache_append: NULL tensor");
    assert(K_new->ndim == 4 && V_new->ndim == 4
           && "kv_cache_append: inputs must be 4D [B, H, N, d_k]");

    int B     = K_new->shape[0];
    int H     = K_new->shape[1];
    int N_new = K_new->shape[2];
    int D     = K_new->shape[3];

    assert(B == kvc->k_cache->shape[0] && "kv_cache_append: batch mismatch");
    assert(H == kvc->k_cache->shape[1] && "kv_cache_append: heads mismatch");
    assert(D == kvc->k_cache->shape[3] && "kv_cache_append: d_k mismatch");
    assert(N_new == V_new->shape[2] && "kv_cache_append: K/V seq len mismatch");
    assert(kvc->seq_len + N_new <= kvc->max_seq
           && "kv_cache_append: cache full");

    float *k_data = (float*)kvc->k_cache->data;
    float *v_data = (float*)kvc->v_cache->data;
    const float *k_src = (const float*)K_new->data;
    const float *v_src = (const float*)V_new->data;

    /* Cache strides (pre-allocated with max_seq along dim 2) */
    int cache_stride_h = kvc->k_cache->strides[1];  /* D * max_seq */
    int cache_stride_s = kvc->k_cache->strides[2];  /* D */
    int v_cache_stride_h = kvc->v_cache->strides[1];

    /* Source strides (K_new / V_new are contiguous from split_heads) */
    int src_stride_h = K_new->strides[1];  /* D * N_new if contiguous */
    int src_stride_b = K_new->strides[0];  /* H * D * N_new if contiguous */

    size_t row_bytes = (size_t)N_new * D * sizeof(float);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            /* Destination: cache (b, h, seq_len, 0) */
            int dst_k_off = kvc->k_cache->offset
                          + b * kvc->k_cache->strides[0]
                          + h * cache_stride_h
                          + kvc->seq_len * cache_stride_s;
            int dst_v_off = kvc->v_cache->offset
                          + b * kvc->v_cache->strides[0]
                          + h * v_cache_stride_h
                          + kvc->seq_len * cache_stride_s;

            /* Source: K_new (b, h, 0, 0) */
            int src_k_off = K_new->offset
                          + b * src_stride_b
                          + h * src_stride_h;
            int src_v_off = V_new->offset
                          + b * src_stride_b
                          + h * src_stride_h;

            memcpy(k_data + dst_k_off, k_src + src_k_off, row_bytes);
            memcpy(v_data + dst_v_off, v_src + src_v_off, row_bytes);
        }
    }

    kvc->seq_len += N_new;
}

tensor *kv_cache_get_K(struct mem_pool *scratch, kv_cache *kvc) {
    assert(kvc && "kv_cache_get_K: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_K: empty cache");
    return tensor_slice(scratch, kvc->k_cache, 2, 0, kvc->seq_len);
}

tensor *kv_cache_get_V(struct mem_pool *scratch, kv_cache *kvc) {
    assert(kvc && "kv_cache_get_V: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_V: empty cache");
    return tensor_slice(scratch, kvc->v_cache, 2, 0, kvc->seq_len);
}

/* ── Transformer Block ── */

transformer_block *transformer_block_create(struct mem_pool *params_pool, int d_model, int n_heads, int d_k,
                                             int intermediate_size) {
    assert(d_model > 0 && n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");

    transformer_block *block = _mem_pool_alloc(params_pool, sizeof(transformer_block), NULL);
    module_init(&block->base, params_pool, "transformer_block");

    block->d_model  = d_model;
    block->n_heads  = n_heads;
    block->d_k      = d_k;

    block->q_proj   = linear_create(params_pool, d_model, n_heads * d_k);
    module_add_child(&block->base, "q_proj", &block->q_proj->base);
    block->k_proj   = linear_create(params_pool, d_model, n_heads * d_k);
    module_add_child(&block->base, "k_proj", &block->k_proj->base);
    block->v_proj   = linear_create(params_pool, d_model, n_heads * d_k);
    module_add_child(&block->base, "v_proj", &block->v_proj->base);
    block->out_proj = linear_create(params_pool, n_heads * d_k, d_model);
    module_add_child(&block->base, "out_proj", &block->out_proj->base);

    /* Pre-attention and pre-FFN layer norms */
    block->attn_norm = layer_norm_create(params_pool, d_model, 1e-5f);
    module_add_child(&block->base, "attn_norm", &block->attn_norm->base);
    block->ffn_norm = layer_norm_create(params_pool, d_model, 1e-5f);
    module_add_child(&block->base, "ffn_norm", &block->ffn_norm->base);

    block->ffn = swiglu_ffn_create(params_pool, d_model, intermediate_size);
    module_add_child(&block->base, "ffn", &block->ffn->base);

    /* RoPE disabled by default */
    block->freqs_cos = NULL;
    block->freqs_sin = NULL;

    return block;
}

tensor *transformer_block_forward(struct mem_pool *scratch, transformer_block *block, const tensor *x) {
    assert(block && x);
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == block->d_model);

    /* ── Attention sublayer (pre-norm) ── */
    tensor *residual = (tensor*)x;
    tensor *h = layer_norm_forward(scratch, block->attn_norm, x);

    /* QKV projections: [B, N, d_model] → [B, N, n_heads * d_k] */
    tensor *Q = linear_forward(scratch, block->q_proj, h);
    tensor *K = linear_forward(scratch, block->k_proj, h);
    tensor *V = linear_forward(scratch, block->v_proj, h);

    /* Split heads: [B, N, H*d_k] → [B, H, N, d_k] */
    tensor *Qh = tensor_split_heads(scratch, Q, block->n_heads);
    tensor *Kh = tensor_split_heads(scratch, K, block->n_heads);
    tensor *Vh = tensor_split_heads(scratch, V, block->n_heads);

    /* ── RoPE: apply rotary position encoding to Q, K ── */
    if (block->freqs_cos && block->freqs_sin) {
        /* Slice freq tables to match current sequence length */
        int N = Qh->shape[2];
        tensor *fc = tensor_slice(scratch, block->freqs_cos, 0, 0, N);
        tensor *fs = tensor_slice(scratch, block->freqs_sin, 0, 0, N);
        Qh = tensor_rope(scratch, Qh, fc, fs);
        Kh = tensor_rope(scratch, Kh, fc, fs);
    }

    /* Fused causal attention (no extra mask needed) */
    tensor *attn_out = tensor_attention(scratch, Qh, Kh, Vh, NULL);

    /* Merge heads: [B, H, N, d_k] → [B, N, H*d_k] */
    tensor *attn_merged = tensor_merge_heads(scratch, attn_out);

    /* Output projection: [B, N, H*d_k] → [B, N, d_model] */
    tensor *attn_proj = linear_forward(scratch, block->out_proj, attn_merged);

    /* First residual: x = x + attn_proj */
    tensor *x_after_attn = tensor_add(scratch, residual, attn_proj);

    /* ── FFN sublayer (pre-norm) ── */
    residual = x_after_attn;
    h = layer_norm_forward(scratch, block->ffn_norm, x_after_attn);

    tensor *ffn_out = swiglu_ffn_forward(scratch, block->ffn, h);

    /* Second residual: return x + ffn_out */
    return tensor_add(scratch, residual, ffn_out);
}

/* ── Cached transformer block forward (eval-only, generation) ── */

tensor *transformer_block_forward_cached(struct mem_pool *scratch,
                                          transformer_block *block,
                                          const tensor *x,
                                          kv_cache *cache) {
    assert(block && x && cache);
    assert(x->contiguous && "cached forward: x must be contiguous");
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == block->d_model);

    int B     = x->shape[0];
    int N_new = x->shape[x->ndim - 2];  /* sequence dim for new tokens */
    int H     = block->n_heads;
    int d_k   = block->d_k;

    /* Pre-norm */
    tensor *h = layer_norm_forward(scratch, block->attn_norm, x);

    /* QKV projections */
    tensor *Q = linear_forward(scratch, block->q_proj, h);  /* [B, N_new, H*d_k] */
    tensor *K = linear_forward(scratch, block->k_proj, h);
    tensor *V = linear_forward(scratch, block->v_proj, h);

    /* Split heads */
    tensor *Qh = tensor_split_heads(scratch, Q, H);  /* [B, H, N_new, d_k] */
    tensor *Kh = tensor_split_heads(scratch, K, H);
    tensor *Vh = tensor_split_heads(scratch, V, H);

    /* ── RoPE: apply to Q, K with position offset ── */
    if (block->freqs_cos && block->freqs_sin) {
        /* New tokens are at positions [cache->seq_len, cache->seq_len + N_new)
         * Slice the freq tables to start at the current cache position. */
        int pos_offset = cache->seq_len;
        tensor *fc_slice = tensor_slice(scratch, block->freqs_cos, 0, pos_offset, N_new);
        tensor *fs_slice = tensor_slice(scratch, block->freqs_sin, 0, pos_offset, N_new);
        Qh = tensor_rope(scratch, Qh, fc_slice, fs_slice);
        Kh = tensor_rope(scratch, Kh, fc_slice, fs_slice);
    }

    /* Append new K/V to cache */
    kv_cache_append(cache, Kh, Vh);

    /* Get full cached K/V */
    tensor *K_full = kv_cache_get_K(scratch, cache);  /* [B, H, S, d_k] */
    tensor *V_full = kv_cache_get_V(scratch, cache);  /* [B, H, S, d_k] */

    int S = K_full->shape[2];
    float scale = 1.0f / sqrtf((float)d_k);

    /* Output allocation */
    tensor *attn_out = tensor_scratch(scratch, 4, (int[]){B, H, N_new, d_k}, 0);
    float *od = (float*)attn_out->data;
    float *qd = (float*)Qh->data;
    float *kd = (float*)K_full->data;
    float *vd = (float*)V_full->data;

    int q_sN = Qh->strides[2];
    int k_sN = K_full->strides[2];
    int v_sN = V_full->strides[2];
    int o_sN = attn_out->strides[2];

    /* Temp scores [N_new, S] */
    float *scores = _mem_pool_alloc(scratch, (size_t)N_new * S * sizeof(float), NULL);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            float *q_slice = qd + Qh->offset
                           + b * Qh->strides[0]
                           + h * Qh->strides[1];
            float *k_slice = kd + K_full->offset
                           + b * K_full->strides[0]
                           + h * K_full->strides[1];
            float *v_slice = vd + V_full->offset
                           + b * V_full->strides[0]
                           + h * V_full->strides[1];
            float *o_slice = od + attn_out->offset
                           + b * attn_out->strides[0]
                           + h * attn_out->strides[1];

            /* scores = Q @ K^T * scale  [N_new, S] */
#if NO_CBLAS
            for (int i = 0; i < N_new; i++)
                for (int j = 0; j < S; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < d_k; kk++)
                        sum += q_slice[i * q_sN + kk] * k_slice[j * k_sN + kk];
                    scores[i * S + j] = sum * scale;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N_new, S, d_k, scale, q_slice, q_sN, k_slice, k_sN,
                        0.0f, scores, S);
#endif

            /* ── Causal mask: row i attends only to positions ≤ old_seq + i ── */
            {
                int old_seq = S - N_new;
                for (int i = 0; i < N_new; i++) {
                    float *row = scores + i * S;
                    for (int j = old_seq + i + 1; j < S; j++)
                        row[j] = -INFINITY;
                }
            }

            /* Softmax over last dim — causal mask applied above
             * Fused online max + sum_exp + NEON SIMD.
             */
            for (int i = 0; i < N_new; i++) {
                float *row = scores + i * S;

                /* ── Fused pass: online max + sum_exp ── */
                float mx = -INFINITY;
                float se = 0.0f;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    for (; j + 4 <= S; j += 4) {
                        float32x4_t v = vld1q_f32(row + j);
                        float group_max = vmaxvq_f32(v);
                        if (group_max > mx) {
                            se *= expf(mx - group_max);
                            mx = group_max;
                        }
                        float32x4_t shifted = vsubq_f32(v, vdupq_n_f32(mx));
                        se += vaddvq_f32(simd_expf_f32(shifted));
                    }
                    for (; j < S; j++) {
                        float old_mx = mx;
                        if (row[j] > mx) mx = row[j];
                        if (mx != old_mx) se *= expf(old_mx - mx);
                        se += expf(row[j] - mx);
                    }
                }
#else
                for (int j = 0; j < S; j++) {
                    float old_mx = mx;
                    if (row[j] > mx) mx = row[j];
                    if (mx != old_mx) se *= expf(old_mx - mx);
                    se += expf(row[j] - mx);
                }
#endif

                /* ── Write softmax weights (1 pass) ── */
                float inv_se = 1.0f / se;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    float32x4_t vmx = vdupq_n_f32(mx);
                    float32x4_t vinv_se = vdupq_n_f32(inv_se);
                    for (; j + 4 <= S; j += 4) {
                        float32x4_t v = vld1q_f32(row + j);
                        float32x4_t exp_v = simd_expf_f32(vsubq_f32(v, vmx));
                        vst1q_f32(row + j, vmulq_f32(exp_v, vinv_se));
                    }
                    for (; j < S; j++)
                        row[j] = expf(row[j] - mx) * inv_se;
                }
#else
                for (int j = 0; j < S; j++)
                    row[j] = expf(row[j] - mx) * inv_se;
#endif
            }

            /* O = P @ V  [N_new, d_k] */
#if NO_CBLAS
            for (int i = 0; i < N_new; i++)
                for (int j = 0; j < d_k; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < S; kk++)
                        sum += scores[i * S + kk] * v_slice[kk * v_sN + j];
                    o_slice[i * o_sN + j] = sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N_new, d_k, S, 1.0f, scores, S, v_slice, v_sN,
                        0.0f, o_slice, o_sN);
#endif
        }
    }

    /* No autograd — generation is eval-only */

    /* Merge heads: [B, H, N_new, d_k] → [B, N_new, H*d_k] */
    tensor *attn_merged = tensor_merge_heads(scratch, attn_out);

    /* Output projection */
    tensor *attn_proj = linear_forward(scratch, block->out_proj, attn_merged);

    /* First residual */
    tensor *x_after_attn = tensor_add(scratch, x, attn_proj);

    /* ── FFN sublayer (pre-norm) ── */
    h = layer_norm_forward(scratch, block->ffn_norm, x_after_attn);
    tensor *ffn_out = swiglu_ffn_forward(scratch, block->ffn, h);

    /* Second residual */
    return tensor_add(scratch, x_after_attn, ffn_out);
}

/* ── Parameter count ── */

long long transformer_block_num_parameters(transformer_block *block) {
    return module_num_parameters(&block->base);
}
