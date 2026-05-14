#include "transformer.h"
#include "nn.h"
#include "norm.h"
#include "ops.h"
#include "attention.h"
#include "multihead.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd.h"
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── KV Cache ── */

kv_cache *kv_cache_create(int B, int H, int max_seq, int d_k) {
    kv_cache *kvc = mem_params_alloc(sizeof(kv_cache), NULL);
    int shape[4] = {B, H, max_seq, d_k};
    kvc->k_cache = tensor_zeros(4, shape, 0);
    kvc->v_cache = tensor_zeros(4, shape, 0);
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

tensor *kv_cache_get_K(kv_cache *kvc) {
    assert(kvc && "kv_cache_get_K: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_K: empty cache");
    return tensor_slice(kvc->k_cache, 2, 0, kvc->seq_len);
}

tensor *kv_cache_get_V(kv_cache *kvc) {
    assert(kvc && "kv_cache_get_V: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_V: empty cache");
    return tensor_slice(kvc->v_cache, 2, 0, kvc->seq_len);
}

/* ── Transformer Block ── */

transformer_block *transformer_block_create(int d_model, int n_heads, int d_k,
                                             int intermediate_size) {
    assert(d_model > 0 && n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");

    transformer_block *block = mem_params_alloc(sizeof(transformer_block), NULL);
    block->d_model  = d_model;
    block->n_heads  = n_heads;
    block->d_k      = d_k;

    block->q_proj   = linear_create(d_model, n_heads * d_k);
    block->k_proj   = linear_create(d_model, n_heads * d_k);
    block->v_proj   = linear_create(d_model, n_heads * d_k);
    block->out_proj = linear_create(n_heads * d_k, d_model);

    /* Pre-norm params: init γ=1, β=0 */
    block->attn_norm_weight = tensor_zeros(1, (int[]){d_model}, 1);
    block->attn_norm_bias   = tensor_zeros(1, (int[]){d_model}, 1);
    float *wn = tensor_data_ptr(block->attn_norm_weight);
    for (int i = 0; i < d_model; i++) wn[i] = 1.0f;

    block->ffn_norm_weight = tensor_zeros(1, (int[]){d_model}, 1);
    block->ffn_norm_bias   = tensor_zeros(1, (int[]){d_model}, 1);
    float *wf = tensor_data_ptr(block->ffn_norm_weight);
    for (int i = 0; i < d_model; i++) wf[i] = 1.0f;

    block->ffn = swiglu_ffn_create(d_model, intermediate_size);

    return block;
}

tensor *transformer_block_forward(transformer_block *block, const tensor *x) {
    assert(block && x);
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == block->d_model);

    /* ── Attention sublayer (pre-norm) ── */
    tensor *residual = (tensor*)x;
    tensor *h = tensor_layer_norm(x, block->attn_norm_weight,
                                   block->attn_norm_bias, 1e-5f);

    /* QKV projections: [B, N, d_model] → [B, N, n_heads * d_k] */
    tensor *Q = linear_forward(block->q_proj, h);
    tensor *K = linear_forward(block->k_proj, h);
    tensor *V = linear_forward(block->v_proj, h);

    /* Split heads: [B, N, H*d_k] → [B, H, N, d_k] */
    tensor *Qh = tensor_split_heads(Q, block->n_heads);
    tensor *Kh = tensor_split_heads(K, block->n_heads);
    tensor *Vh = tensor_split_heads(V, block->n_heads);

    /* Fused causal attention (no extra mask needed) */
    tensor *attn_out = tensor_attention(Qh, Kh, Vh, NULL);

    /* Merge heads: [B, H, N, d_k] → [B, N, H*d_k] */
    tensor *attn_merged = tensor_merge_heads(attn_out);

    /* Output projection: [B, N, H*d_k] → [B, N, d_model] */
    tensor *attn_proj = linear_forward(block->out_proj, attn_merged);

    /* First residual: x = x + attn_proj */
    tensor *x_after_attn = tensor_add(residual, attn_proj);

    /* ── FFN sublayer (pre-norm) ── */
    residual = x_after_attn;
    h = tensor_layer_norm(x_after_attn, block->ffn_norm_weight,
                           block->ffn_norm_bias, 1e-5f);

    tensor *ffn_out = swiglu_ffn_forward(block->ffn, h);

    /* Second residual: return x + ffn_out */
    return tensor_add(residual, ffn_out);
}

/* ── Decoder-only Language Model ── */

decoder_lm *decoder_lm_create(int vocab_size, int d_model,
                               int n_layers, int n_heads, int d_k,
                               int intermediate_size) {
    assert(vocab_size > 0 && d_model > 0 && n_layers > 0);
    assert(n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");

    decoder_lm *lm = mem_params_alloc(sizeof(decoder_lm), NULL);
    lm->d_model    = d_model;
    lm->vocab_size = vocab_size;
    lm->n_layers   = n_layers;

    /* Embedding table: [vocab_size, d_model], uniform init */
    float bound = 1.0f / sqrtf((float)d_model);
    lm->embedding_table = tensor_uniform(2, (int[]){vocab_size, d_model}, 1, bound);

    /* Transformer blocks */
    lm->blocks = mem_params_alloc(n_layers * sizeof(transformer_block*), NULL);
    for (int i = 0; i < n_layers; i++) {
        lm->blocks[i] = transformer_block_create(d_model, n_heads, d_k,
                                                   intermediate_size);
    }

    /* Final layer norm: γ=1, β=0 */
    lm->norm_weight = tensor_zeros(1, (int[]){d_model}, 1);
    lm->norm_bias   = tensor_zeros(1, (int[]){d_model}, 1);
    float *wn = tensor_data_ptr(lm->norm_weight);
    for (int i = 0; i < d_model; i++) wn[i] = 1.0f;

    /* LM head: d_model → vocab_size */
    lm->lm_head = linear_create(d_model, vocab_size);

    return lm;
}

tensor *decoder_lm_forward(decoder_lm *lm, const tensor *input_ids) {
    assert(lm && input_ids);
    assert(input_ids->ndim == 2 && "decoder_lm_forward: input_ids must be 2D [B, N]");
    assert(input_ids->contiguous && "decoder_lm_forward: input_ids must be contiguous");

    int B = input_ids->shape[0];
    int N = input_ids->shape[1];
    int D = lm->d_model;

    /* Flatten [B, N] → [B*N] for embedding lookup */
    tensor *flat_ids = tensor_flatten((tensor*)input_ids);  /* view, no data copy */

    /* Embed: [B*N] → [B*N, d_model] */
    tensor *h = tensor_embedding(lm->embedding_table, flat_ids);

    /* Reshape to [B, N, d_model] */
    h = tensor_reshape(h, 3, (int[]){B, N, D});

    /* Pass through all transformer blocks */
    for (int i = 0; i < lm->n_layers; i++) {
        h = transformer_block_forward(lm->blocks[i], h);
    }

    /* Final layer norm */
    h = tensor_layer_norm(h, lm->norm_weight, lm->norm_bias, 1e-5f);

    /* LM head: [B, N, d_model] → [B, N, vocab_size] */
    tensor *logits = linear_forward(lm->lm_head, h);

    return logits;
}
