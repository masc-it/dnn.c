#include "transformer.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd.h"
#include <assert.h>
#include <string.h>

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
