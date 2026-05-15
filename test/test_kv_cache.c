#include "dnn.h"
#include "context.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

#define EPS 1e-5f

/* helper: read a tensor element by logical coordinate (handles strided views) */
static float tget(tensor *t, int b, int h, int n, int d) {
    float *base = (float*)t->data;
    int off = t->offset
            + b * t->strides[0]
            + h * t->strides[1]
            + n * t->strides[2]
            + d * t->strides[3];
    return base[off];
}

/* helper: check tensor shape */
static void check_shape(tensor *t, int ndim, const int *exp, const char *label) {
    assert(tensor_ndim(t) == ndim && "shape ndim mismatch");
    for (int i = 0; i < ndim; i++) {
        if (tensor_shape(t, i) != exp[i]) {
            printf("    FAIL: %s shape[%d]: got %d, expected %d\n",
                   label, i, tensor_shape(t, i), exp[i]);
            assert(0);
        }
    }
}

/* ── Tests ── */

static void test_create(void) {
    printf("  test_create... ");
    kv_cache *kvc = kv_cache_create(ctx.params, 2, 4, 512, 64);

    assert(kvc->seq_len == 0);
    assert(kvc->max_seq == 512);
    assert(kvc->k_cache != NULL);
    assert(kvc->v_cache != NULL);

    int exp_shape[] = {2, 4, 512, 64};
    check_shape(kvc->k_cache, 4, exp_shape, "k_cache");
    check_shape(kvc->v_cache, 4, exp_shape, "v_cache");

    /* Verify zero-initialized via cache buffer directly */
    float *kd = (float*)kvc->k_cache->data;
    float *vd = (float*)kvc->v_cache->data;
    for (int i = 0; i < 2 * 4 * 512 * 64; i++) {
        assert(kd[i] == 0.0f && "k_cache not zeroed");
        assert(vd[i] == 0.0f && "v_cache not zeroed");
    }

    printf("OK\n");
}

static void test_append_one_token(void) {
    printf("  test_append_one_token... ");
    kv_cache *kvc = kv_cache_create(ctx.params, 1, 2, 16, 4);

    /* Create K_new: [1, 2, 1, 4] with known values */
    tensor *K_new = tensor_zeros_data(ctx.data, 4, (int[]){1, 2, 1, 4});
    tensor *V_new = tensor_zeros_data(ctx.data, 4, (int[]){1, 2, 1, 4});
    float *kd = (float*)K_new->data;
    float *vd = (float*)V_new->data;
    /* head 0: K=[1,2,3,4], V=[5,6,7,8] */
    kd[0]=1; kd[1]=2; kd[2]=3; kd[3]=4;
    vd[0]=5; vd[1]=6; vd[2]=7; vd[3]=8;
    /* head 1: K=[9,10,11,12], V=[13,14,15,16] */
    kd[4]=9; kd[5]=10; kd[6]=11; kd[7]=12;
    vd[4]=13; vd[5]=14; vd[6]=15; vd[7]=16;

    kv_cache_append(kvc, K_new, V_new);
    assert(kvc->seq_len == 1);

    /* Verify data directly in cache buffer */
    /* b=0, h=0, n=0 */
    assert(fabsf(tget(kvc->k_cache, 0, 0, 0, 0) - 1.0f) < EPS);
    assert(fabsf(tget(kvc->k_cache, 0, 0, 0, 1) - 2.0f) < EPS);
    assert(fabsf(tget(kvc->k_cache, 0, 0, 0, 2) - 3.0f) < EPS);
    assert(fabsf(tget(kvc->k_cache, 0, 0, 0, 3) - 4.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 0, 0, 0) - 5.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 0, 0, 1) - 6.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 0, 0, 2) - 7.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 0, 0, 3) - 8.0f) < EPS);

    /* b=0, h=1, n=0: offset = 0*128 + 1*64 + 0*4 = 64 */
    assert(fabsf(tget(kvc->k_cache, 0, 1, 0, 0) - 9.0f) < EPS);
    assert(fabsf(tget(kvc->k_cache, 0, 1, 0, 1) - 10.0f) < EPS);
    assert(fabsf(tget(kvc->k_cache, 0, 1, 0, 2) - 11.0f) < EPS);
    assert(fabsf(tget(kvc->k_cache, 0, 1, 0, 3) - 12.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 1, 0, 0) - 13.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 1, 0, 1) - 14.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 1, 0, 2) - 15.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 1, 0, 3) - 16.0f) < EPS);

    /* Verify get_K/V returns correct shape */
    tensor *K_view = kv_cache_get_K(ctx.scratch, kvc);
    tensor *V_view = kv_cache_get_V(ctx.scratch, kvc);
    check_shape(K_view, 4, (int[]){1, 2, 1, 4}, "K_view");
    check_shape(V_view, 4, (int[]){1, 2, 1, 4}, "V_view");

    printf("OK\n");
}

static void test_append_multiple_tokens(void) {
    printf("  test_append_multiple_tokens... ");
    kv_cache *kvc = kv_cache_create(ctx.params, 1, 1, 8, 3);

    /* Append 2 tokens: [1,1,1], [2,2,2] */
    tensor *K1 = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 2, 3});
    float *k1d = (float*)K1->data;
    k1d[0]=1; k1d[1]=1; k1d[2]=1; k1d[3]=2; k1d[4]=2; k1d[5]=2;

    tensor *V1 = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 2, 3});
    float *v1d = (float*)V1->data;
    v1d[0]=10; v1d[1]=10; v1d[2]=10; v1d[3]=20; v1d[4]=20; v1d[5]=20;

    kv_cache_append(kvc, K1, V1);
    assert(kvc->seq_len == 2);

    /* Append 1 more token: [3,3,3] */
    tensor *K2 = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 1, 3});
    float *k2d = (float*)K2->data;
    k2d[0]=3; k2d[1]=3; k2d[2]=3;

    tensor *V2 = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 1, 3});
    float *v2d = (float*)V2->data;
    v2d[0]=30; v2d[1]=30; v2d[2]=30;

    kv_cache_append(kvc, K2, V2);
    assert(kvc->seq_len == 3);

    /* Verify via tget on the cache */
    for (int d = 0; d < 3; d++) {
        assert(fabsf(tget(kvc->k_cache, 0, 0, 0, d) - 1.0f) < EPS);
        assert(fabsf(tget(kvc->k_cache, 0, 0, 1, d) - 2.0f) < EPS);
        assert(fabsf(tget(kvc->k_cache, 0, 0, 2, d) - 3.0f) < EPS);
        assert(fabsf(tget(kvc->v_cache, 0, 0, 0, d) - 10.0f) < EPS);
        assert(fabsf(tget(kvc->v_cache, 0, 0, 1, d) - 20.0f) < EPS);
        assert(fabsf(tget(kvc->v_cache, 0, 0, 2, d) - 30.0f) < EPS);
    }

    /* Verify get_K/V shape */
    tensor *K_view = kv_cache_get_K(ctx.scratch, kvc);
    tensor *V_view = kv_cache_get_V(ctx.scratch, kvc);
    check_shape(K_view, 4, (int[]){1, 1, 3, 3}, "K_view mult");
    check_shape(V_view, 4, (int[]){1, 1, 3, 3}, "V_view mult");

    printf("OK\n");
}

static void test_append_multibatch_multihead(void) {
    printf("  test_append_multibatch_multihead... ");
    kv_cache *kvc = kv_cache_create(ctx.params, 2, 3, 10, 2);

    /* Append 2 tokens at once: K/V shape [2, 3, 2, 2] */
    tensor *K = tensor_zeros_data(ctx.data, 4, (int[]){2, 3, 2, 2});
    tensor *V = tensor_zeros_data(ctx.data, 4, (int[]){2, 3, 2, 2});
    float *kd = (float*)K->data;
    float *vd = (float*)V->data;
    /* Fill with distinct values per (b, h, n, d) */
    for (int i = 0; i < 2 * 3 * 2 * 2; i++) {
        kd[i] = (float)(i + 1);
        vd[i] = (float)((i + 1) * 100);
    }

    kv_cache_append(kvc, K, V);
    assert(kvc->seq_len == 2);

    /* Spot-check values in cache buffer via tget */
    /* b=0, h=0, n=0, d=0: flat_idx 0 → value 1 */
    assert(fabsf(tget(kvc->k_cache, 0, 0, 0, 0) - 1.0f) < EPS);
    assert(fabsf(tget(kvc->k_cache, 0, 0, 0, 1) - 2.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 0, 0, 0) - 100.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 0, 0, 1) - 200.0f) < EPS);

    /* b=0, h=1, n=0, d=0: K flat_idx = 0*3*2*2 + 1*2*2 + 0*2 + 0 = 4 → value 5 */
    assert(fabsf(tget(kvc->k_cache, 0, 1, 0, 0) - 5.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 0, 1, 0, 0) - 500.0f) < EPS);

    /* b=1, h=2, n=1, d=0: K flat_idx = 1*3*2*2 + 2*2*2 + 1*2 + 0 = 12+8+2=22 → value 23 */
    assert(fabsf(tget(kvc->k_cache, 1, 2, 1, 0) - 23.0f) < EPS);
    assert(fabsf(tget(kvc->v_cache, 1, 2, 1, 0) - 2300.0f) < EPS);

    /* Verify get_K/V shape */
    tensor *K_view = kv_cache_get_K(ctx.scratch, kvc);
    tensor *V_view = kv_cache_get_V(ctx.scratch, kvc);
    check_shape(K_view, 4, (int[]){2, 3, 2, 2}, "K_view mb");
    check_shape(V_view, 4, (int[]){2, 3, 2, 2}, "V_view mb");

    printf("OK\n");
}

static void test_append_fills_to_max(void) {
    printf("  test_append_fills_to_max... ");
    kv_cache *kvc = kv_cache_create(ctx.params, 1, 1, 4, 2);

    /* Append one at a time until full */
    for (int t = 1; t <= 4; t++) {
        tensor *K = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 1, 2});
        tensor *V = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 1, 2});
        float *kd = (float*)K->data;
        float *vd = (float*)V->data;
        kd[0] = (float)t; kd[1] = (float)(t * 10);
        vd[0] = (float)(t * 100); vd[1] = (float)(t * 1000);

        kv_cache_append(kvc, K, V);
        assert(kvc->seq_len == t);
    }

    assert(kvc->seq_len == 4);

    /* Verify all values in cache buffer */
    for (int n = 0; n < 4; n++) {
        assert(fabsf(tget(kvc->k_cache, 0, 0, n, 0) - (float)(n + 1)) < EPS);
        assert(fabsf(tget(kvc->k_cache, 0, 0, n, 1) - (float)((n + 1) * 10)) < EPS);
        assert(fabsf(tget(kvc->v_cache, 0, 0, n, 0) - (float)((n + 1) * 100)) < EPS);
        assert(fabsf(tget(kvc->v_cache, 0, 0, n, 1) - (float)((n + 1) * 1000)) < EPS);
    }

    /* Verify get_K/V shape for full cache */
    tensor *K_view = kv_cache_get_K(ctx.scratch, kvc);
    tensor *V_view = kv_cache_get_V(ctx.scratch, kvc);
    check_shape(K_view, 4, (int[]){1, 1, 4, 2}, "K_view full");
    check_shape(V_view, 4, (int[]){1, 1, 4, 2}, "V_view full");

    printf("OK\n");
}

static void test_append_seq_len_grows(void) {
    printf("  test_append_seq_len_grows... ");
    kv_cache *kvc = kv_cache_create(ctx.params, 1, 1, 5, 2);

    assert(kvc->seq_len == 0);
    assert(kvc->max_seq == 5);

    tensor *K = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 1, 2});
    tensor *V = tensor_zeros_data(ctx.data, 4, (int[]){1, 1, 1, 2});
    ((float*)K->data)[0] = 1.0f; ((float*)K->data)[1] = 2.0f;
    ((float*)V->data)[0] = 3.0f; ((float*)V->data)[1] = 4.0f;

    for (int t = 1; t <= 3; t++) {
        kv_cache_append(kvc, K, V);
        assert(kvc->seq_len == t);
    }

    printf("OK\n");
}

int main(void) {
    printf("test_kv_cache:\n");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 4 * 1024 * 1024, 4 * 1024 * 1024);

    test_create();                         mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch); mem_pool_reset(ctx.data);
    test_append_one_token();               mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch); mem_pool_reset(ctx.data);
    test_append_multiple_tokens();         mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch); mem_pool_reset(ctx.data);
    test_append_multibatch_multihead();    mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch); mem_pool_reset(ctx.data);
    test_append_fills_to_max();            mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch); mem_pool_reset(ctx.data);
    test_append_seq_len_grows();           mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch); mem_pool_reset(ctx.data);

    printf("  ALL PASS\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
