#include "dnn.h"
#include "context.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

#define EPS 1e-4f

/* ── Helpers ── */

/* read element at logical index from a (possibly strided) tensor */
static float get_elem(tensor *t, int idx) {
    /* walk dims from innermost to outermost to compute offset */
    float *base = (float*)t->data + t->offset;
    if (tensor_is_contiguous(t))
        return base[idx];
    /* strided: decompose flat idx into coords, compute offset */
    int coords[DNN_MAX_DIMS];
    int rem = idx;
    for (int d = t->ndim - 1; d >= 0; d--) {
        coords[d] = rem % t->shape[d];
        rem /= t->shape[d];
    }
    int off = 0;
    for (int d = 0; d < t->ndim; d++)
        off += coords[d] * t->strides[d];
    return base[off];
}

static void check_data_ary_tol(tensor *t, const float *exp, int n, float tol, const char *label) {
    for (int i = 0; i < n; i++) {
        float v = get_elem(t, i);
        if (fabsf(v - exp[i]) > tol) {
            printf("    FAIL: %s[%d]: got %.6f, expected %.6f (tol=%.6f)\n",
                   label, i, v, exp[i], tol);
            assert(0);
        }
    }
}

static void check_data_eq(tensor *t, const float *exp, int n, const char *label) {
    check_data_ary_tol(t, exp, n, EPS, label);
}

static tensor *make_param(int ndim, const int *shape, const float *data) {
    tensor *t = tensor_zeros(ctx.params, ndim, shape, 1);
    float *td = tensor_data_ptr(t);
    memcpy(td, data, tensor_numel(t) * sizeof(float));
    return t;
}

static tensor *make_tensor(int ndim, const int *shape, const float *data) {
    return make_param(ndim, shape, data); /* same pool, just no grad tracking needed */
}

/* ── Test 1: split_heads basic ──
 *
 *   B=1, N=2, H=2, d_k=2
 *   Input [1,2,4] = [[a,b,c,d], [e,f,g,h]]
 *   Expected [1,2,2,2] = [[[a,b],[c,d]], [[e,f],[g,h]]]
 */
static void test_split_basic(void) {
    printf("  test_split_basic... ");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f};
    tensor *t = make_tensor(3, (int[]){1, 2, 4}, data);
    tensor *out = tensor_split_heads(ctx.scratch, t, 2);

    assert(tensor_ndim(out) == 4);
    assert(tensor_shape(out, 0) == 1);
    assert(tensor_shape(out, 1) == 2);  /* H */
    assert(tensor_shape(out, 2) == 2);  /* N */
    assert(tensor_shape(out, 3) == 2);  /* d_k */

    /* After transpose: [B, H, N, d_k] logical order */
    float exp[] = {1.0f, 2.0f,    /* H=0, N=0 */
                   5.0f, 6.0f,    /* H=0, N=1 */
                   3.0f, 4.0f,    /* H=1, N=0 */
                   7.0f, 8.0f};   /* H=1, N=1 */
    check_data_eq(out, exp, 8, "out");
    printf("OK\n");
}

/* ── Test 2: split_heads with B=2 ── */
static void test_split_batched(void) {
    printf("  test_split_batched... ");
    /* B=2, N=2, H=2, d_k=2 → input [2,2,4] */
    float data[] = {
        /* batch 0 */
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        /* batch 1 */
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    tensor *t = make_tensor(3, (int[]){2, 2, 4}, data);
    tensor *out = tensor_split_heads(ctx.scratch, t, 2);

    assert(tensor_ndim(out) == 4);
    assert(tensor_shape(out, 0) == 2);
    assert(tensor_shape(out, 1) == 2);
    assert(tensor_shape(out, 2) == 2);
    assert(tensor_shape(out, 3) == 2);

    /* [B,H,N,d_k] logical order:
     * B=0: H=0,N=0:[1,2], H=0,N=1:[5,6], H=1,N=0:[3,4], H=1,N=1:[7,8]
     * B=1: H=0,N=0:[9,10], H=0,N=1:[13,14], H=1,N=0:[11,12], H=1,N=1:[15,16] */
    float exp[] = {
        1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 13.0f, 14.0f, 11.0f, 12.0f, 15.0f, 16.0f
    };
    check_data_eq(out, exp, 16, "out");
    printf("OK\n");
}

/* ── Test 3: split_heads H=1 (degenerate, just reshape) ── */
static void test_split_h1(void) {
    printf("  test_split_h1... ");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor *t = make_tensor(3, (int[]){1, 2, 2}, data);  /* B=1, N=2, H=1, d_k=2 */
    tensor *out = tensor_split_heads(ctx.scratch, t, 1);

    assert(tensor_ndim(out) == 4);
    assert(tensor_shape(out, 0) == 1);
    assert(tensor_shape(out, 1) == 1);
    assert(tensor_shape(out, 2) == 2);
    assert(tensor_shape(out, 3) == 2);

    check_data_eq(out, data, 4, "out");
    printf("OK\n");
}

/* ── Test 4: merge_heads basic ──
 *
 *   Input [1,2,2,2] (B=1, H=2, N=2, d_k=2)
 *   Expected [1,2,4] with heads concatenated along last dim
 */
static void test_merge_basic(void) {
    printf("  test_merge_basic... ");
    float data[] = {1.0f, 2.0f,    /* head 0, N=0 */
                    3.0f, 4.0f,    /* head 1, N=0 */
                    5.0f, 6.0f,    /* head 0, N=1 */
                    7.0f, 8.0f};   /* head 1, N=1 */
    tensor *t = make_tensor(4, (int[]){1, 2, 2, 2}, data);
    tensor *out = tensor_merge_heads(ctx.scratch, t);

    assert(tensor_ndim(out) == 3);
    assert(tensor_shape(out, 0) == 1);
    assert(tensor_shape(out, 1) == 2);
    assert(tensor_shape(out, 2) == 4);

    /* merge: transpose(H,N) then reshape → [B,N,H*d_k] contiguous.
     * Logical walk yields: N=0:[H0,H1]=[1,2,5,6], N=1:[3,4,7,8] */
    float exp[] = {1.0f, 2.0f, 5.0f, 6.0f,
                   3.0f, 4.0f, 7.0f, 8.0f};
    check_data_eq(out, exp, 8, "out");
    printf("OK\n");
}

/* ── Test 5: merge_heads batched ── */
static void test_merge_batched(void) {
    printf("  test_merge_batched... ");
    float data[] = {
        /* batch 0 */
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        /* batch 1 */
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
    };
    tensor *t = make_tensor(4, (int[]){2, 2, 2, 2}, data);
    tensor *out = tensor_merge_heads(ctx.scratch, t);

    assert(tensor_ndim(out) == 3);
    assert(tensor_shape(out, 0) == 2);
    assert(tensor_shape(out, 1) == 2);
    assert(tensor_shape(out, 2) == 4);

    /* After merge: [B,N,H*d_k] contiguous.
     * B=0: N=0:[1,2,5,6], N=1:[3,4,7,8]
     * B=1: N=0:[9,10,13,14], N=1:[11,12,15,16] */
    float exp[] = {
        1.0f, 2.0f, 5.0f, 6.0f,  3.0f, 4.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 13.0f, 14.0f, 11.0f, 12.0f, 15.0f, 16.0f
    };
    check_data_eq(out, exp, 16, "out");
    printf("OK\n");
}

/* ── Test 6: split → merge roundtrip (identity) ── */
static void test_split_merge_roundtrip(void) {
    printf("  test_split_merge_roundtrip... ");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f,11.0f,12.0f};
    tensor *t = make_tensor(3, (int[]){1, 3, 4}, data);

    tensor *split = tensor_split_heads(ctx.scratch, t, 2);
    tensor *merged = tensor_merge_heads(ctx.scratch, split);

    assert(tensor_ndim(merged) == 3);
    assert(tensor_shape(merged, 0) == 1);
    assert(tensor_shape(merged, 1) == 3);
    assert(tensor_shape(merged, 2) == 4);

    check_data_eq(merged, data, 12, "roundtrip");
    printf("OK\n");
}

/* ── Test 7: merge → split roundtrip (identity) ── */
static void test_merge_split_roundtrip(void) {
    printf("  test_merge_split_roundtrip... ");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f,11.0f,12.0f,
                    13.0f,14.0f,15.0f,16.0f};
    tensor *t = make_tensor(4, (int[]){2, 2, 2, 2}, data);

    tensor *merged = tensor_merge_heads(ctx.scratch, t);
    tensor *split = tensor_split_heads(ctx.scratch, merged, 2);

    assert(tensor_ndim(split) == 4);
    assert(tensor_shape(split, 0) == 2);
    assert(tensor_shape(split, 1) == 2);
    assert(tensor_shape(split, 2) == 2);
    assert(tensor_shape(split, 3) == 2);

    check_data_eq(split, data, 16, "roundtrip");
    printf("OK\n");
}

/* ── Test 8: gradient flows through split_heads ── */
static void test_split_grad(void) {
    printf("  test_split_grad... ");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f};
    tensor *t = make_param(3, (int[]){1, 2, 4}, data);

    tensor *split = tensor_split_heads(ctx.scratch, t, 2);
    tensor *loss  = tensor_sum(ctx.scratch, split, -1);  /* sum over last dim → shape [1,2,2] */
    tensor *loss2 = tensor_sum(ctx.scratch, loss, -1);   /* → shape [1,2] */
    tensor *loss3 = tensor_sum(ctx.scratch, loss2, -1);  /* → scalar [1] */
    dnn_backward(ctx.scratch, loss3);

    assert(tensor_grad(t) && "d_input must be non-NULL");
    float *g = tensor_grad(t);
    int n = tensor_numel(t);
    for (int i = 0; i < n; i++)
        assert(fabsf(g[i]) > 0.0f && "grad must be non-zero");
    printf("OK\n");
}

/* ── Test 9: gradient flows through merge_heads ── */
static void test_merge_grad(void) {
    printf("  test_merge_grad... ");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f};
    tensor *t = make_param(4, (int[]){1, 2, 2, 2}, data);

    tensor *merged = tensor_merge_heads(ctx.scratch, t);
    tensor *loss   = tensor_sum(ctx.scratch, merged, -1);
    tensor *loss2  = tensor_sum(ctx.scratch, loss, -1);
    dnn_backward(ctx.scratch, loss2);

    assert(tensor_grad(t) && "d_input must be non-NULL");
    float *g = tensor_grad(t);
    int n = tensor_numel(t);
    for (int i = 0; i < n; i++)
        assert(fabsf(g[i]) > 0.0f && "grad must be non-zero");
    printf("OK\n");
}

/* ── Test 10: split + merge in no-grad mode ── */
static void test_no_grad(void) {
    printf("  test_no_grad... ");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f};
    tensor *t = make_param(3, (int[]){1, 2, 4}, data);

    dnn_grad_ctx gc = dnn_no_grad_enter();
    tensor *split = tensor_split_heads(ctx.scratch, t, 2);
    dnn_no_grad_exit(gc);

    /* In no-grad mode, grad_fn should be NULL on directly created tensors.
     * The view ops don't create new grad_fn nodes, they pass through. */
    (void)split;
    printf("OK\n");
}

/* ── Test 11: split_heads with different H/d_k ratios ── */
static void test_split_various_shapes(void) {
    printf("  test_split_various_shapes... ");

    /* H=4, d_k=3 → D=12 */
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = (float)i;
    tensor *t = make_tensor(3, (int[]){1, 2, 12}, data);
    tensor *out = tensor_split_heads(ctx.scratch, t, 4);

    assert(tensor_ndim(out) == 4);
    assert(tensor_shape(out, 1) == 4);
    assert(tensor_shape(out, 3) == 3);

    /* Verify data order: [B, N, H*d_k] → [B, H, N, d_k]
     * Input N=0: [0,1,2,3,4,5,6,7,8,9,10,11]
     *   → head 0: [0,1,2], head 1: [3,4,5], head 2: [6,7,8], head 3: [9,10,11]
     * Input N=1: [12,13,14,15,16,17,18,19,20,21,22,23]
     *   → head 0: [12,13,14], head 1: [15,16,17], head 2: [18,19,20], head 3: [21,22,23]
     * Output layout [B=1, H=4, N=2, d_k=3]:
     *   H=0: N=0=[0,1,2], N=1=[12,13,14]
     *   H=1: N=0=[3,4,5], N=1=[15,16,17]
     *   H=2: N=0=[6,7,8], N=1=[18,19,20]
     *   H=3: N=0=[9,10,11], N=1=[21,22,23]
     * In flat order: H=0,N=0, H=0,N=1, H=1,N=0, H=1,N=1, ...
     *   = [0,1,2, 12,13,14, 3,4,5, 15,16,17, 6,7,8, 18,19,20, 9,10,11, 21,22,23]
     */
    float exp[] = {0,1,2, 12,13,14,
                   3,4,5, 15,16,17,
                   6,7,8, 18,19,20,
                   9,10,11, 21,22,23};
    check_data_eq(out, exp, 24, "out");
    printf("OK\n");
}

/* ── Test 12: merge_heads with various shapes ── */
static void test_merge_various_shapes(void) {
    printf("  test_merge_various_shapes... ");
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = (float)i;
    tensor *t = make_tensor(4, (int[]){1, 4, 2, 3}, data);
    tensor *out = tensor_merge_heads(ctx.scratch, t);

    assert(tensor_ndim(out) == 3);
    assert(tensor_shape(out, 2) == 12);

    /* Reverse of above: [B,H,N,d_k] → [B,N,H*d_k]
     * Input layout: H=0,N=0, H=0,N=1, H=1,N=0, H=1,N=1, ...
     *   = [0..2, 3..5, 6..8, 9..11, 12..14, 15..17, 18..20, 21..23]
     * Output [B=1, N=2, H*d_k=12]:
     *   N=0: H0=[0,1,2] H1=[6,7,8] H2=[12,13,14] H3=[18,19,20]
     *       = [0,1,2,6,7,8,12,13,14,18,19,20]
     *   N=1: H0=[3,4,5] H1=[9,10,11] H2=[15,16,17] H3=[21,22,23]
     *       = [3,4,5,9,10,11,15,16,17,21,22,23]
     * In flat order: [0,1,2,6,7,8,12,13,14,18,19,20, 3,4,5,9,10,11,15,16,17,21,22,23]
     */
    float exp[] = {0,1,2,6,7,8,12,13,14,18,19,20,
                   3,4,5,9,10,11,15,16,17,21,22,23};
    check_data_eq(out, exp, 24, "out");
    printf("OK\n");
}

/* ── Test 13: structural — split preserves batch size ── */
static void test_split_structure(void) {
    printf("  test_split_structure... ");
    float data[12];
    tensor *t = make_tensor(3, (int[]){3, 2, 6}, data);
    tensor *out = tensor_split_heads(ctx.scratch, t, 3);

    assert(tensor_shape(out, 0) == 3);
    assert(tensor_shape(out, 1) == 3);
    assert(tensor_shape(out, 2) == 2);
    assert(tensor_shape(out, 3) == 2);
    assert(tensor_numel(out) == tensor_numel(t));
    printf("OK\n");
}

int main(void) {
    printf("test_multihead:\n");

    dnn_ctx_init(&ctx, 10 * 1024 * 1024, 50 * 1024 * 1024, 8*1024*1024);

    test_split_basic();             mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_split_batched();           mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_split_h1();                mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_merge_basic();             mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_merge_batched();           mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_split_merge_roundtrip();   mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_merge_split_roundtrip();   mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_split_grad();              mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_merge_grad();              mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_no_grad();                 mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_split_various_shapes();    mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_merge_various_shapes();    mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_split_structure();         mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);

    printf("  ALL PASS\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
