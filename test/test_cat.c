#include "dnn.h"
#include "context.h"
#include "ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

#define EPS 1e-5f

/* helper: check tensor data elements */
static void check_data_ary(tensor *t, const float *exp, int n, const char *label) {
    float *d = tensor_data_ptr(t);
    for (int i = 0; i < n; i++) {
        if (fabsf(d[i] - exp[i]) > EPS) {
            printf("    FAIL: %s[%d]: got %.6f, expected %.6f\n", label, i, d[i], exp[i]);
            assert(0);
        }
    }
}

/* helper: check tensor grad elements */
static void check_grad_ary(tensor *t, const float *exp, int n, const char *label) {
    float *g = tensor_grad(t);
    if (!g) { printf("    FAIL: %s: grad is NULL\n", label); assert(0); }
    for (int i = 0; i < n; i++) {
        if (fabsf(g[i] - exp[i]) > EPS) {
            printf("    FAIL: %s[%d]: got %.6f, expected %.6f\n", label, i, g[i], exp[i]);
            assert(0);
        }
    }
}

/* ── Forward tests ── */

static void test_cat_1d(void) {
    printf("  test_cat_1d... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 0);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){2}, 0);
    float *ad = tensor_data_ptr(a);
    float *bd = tensor_data_ptr(b);
    ad[0]=1; ad[1]=2; ad[2]=3;
    bd[0]=4; bd[1]=5;

    tensor *out = tensor_cat(ctx.scratch, a, b, 0);
    assert(tensor_ndim(out) == 1);
    assert(tensor_shape(out, 0) == 5);

    float exp[] = {1, 2, 3, 4, 5};
    check_data_ary(out, exp, 5, "out");

    /* properties */
    assert(out->contiguous && "cat output should be contiguous");
    printf("OK\n");
}

static void test_cat_2d_dim0(void) {
    printf("  test_cat_2d_dim0... ");
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 0);
    tensor *b = tensor_zeros(ctx.params, 2, (int[]){4, 3}, 0);
    float *ad = tensor_data_ptr(a);
    float *bd = tensor_data_ptr(b);
    for (int i = 0; i < 6; i++) ad[i] = (float)(i + 1);       /* [[1,2,3],[4,5,6]] */
    for (int i = 0; i < 12; i++) bd[i] = (float)(i + 7);      /* [[7,8,9],[10,11,12],[13,14,15],[16,17,18]] */

    tensor *out = tensor_cat(ctx.scratch, a, b, 0);
    assert(tensor_ndim(out) == 2);
    assert(tensor_shape(out, 0) == 6 && tensor_shape(out, 1) == 3);

    float exp[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15, 16,17,18};
    check_data_ary(out, exp, 18, "out");
    printf("OK\n");
}

static void test_cat_2d_dim1(void) {
    printf("  test_cat_2d_dim1... ");
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 0);
    tensor *b = tensor_zeros(ctx.params, 2, (int[]){3, 5}, 0);
    float *ad = tensor_data_ptr(a);
    float *bd = tensor_data_ptr(b);
    for (int i = 0; i < 6; i++) ad[i] = (float)(i + 1);        /* [[1,2],[3,4],[5,6]] */
    for (int i = 0; i < 15; i++) bd[i] = (float)(i + 7);       /* [[7..11],[12..16],[17..21]] */

    tensor *out = tensor_cat(ctx.scratch, a, b, 1);
    assert(tensor_shape(out, 0) == 3 && tensor_shape(out, 1) == 7);

    float exp[] = {1,2, 7,8,9,10,11,   3,4, 12,13,14,15,16,   5,6, 17,18,19,20,21};
    check_data_ary(out, exp, 21, "out");
    printf("OK\n");
}

static void test_cat_3d_dim1(void) {
    printf("  test_cat_3d_dim1... ");
    tensor *a = tensor_zeros(ctx.params, 3, (int[]){2, 3, 4}, 0);
    tensor *b = tensor_zeros(ctx.params, 3, (int[]){2, 7, 4}, 0);
    float *ad = tensor_data_ptr(a);
    float *bd = tensor_data_ptr(b);
    for (int i = 0; i < 24; i++) ad[i] = 1.0f;
    for (int i = 0; i < 56; i++) bd[i] = 2.0f;

    tensor *out = tensor_cat(ctx.scratch, a, b, 1);
    assert(tensor_shape(out, 0) == 2);
    assert(tensor_shape(out, 1) == 10);
    assert(tensor_shape(out, 2) == 4);
    assert(tensor_numel(out) == 80);
    float *od = tensor_data_ptr(out);

    /* verify first batch [0]: first 3*4=12 elements are 1.0, last 7*4=28 are 2.0 */
    for (int i = 0; i < 12; i++)  assert(fabsf(od[i] - 1.0f) < EPS);
    for (int i = 12; i < 40; i++) assert(fabsf(od[i] - 2.0f) < EPS);
    /* second batch [1]: same pattern */
    for (int i = 40; i < 52; i++)  assert(fabsf(od[i] - 1.0f) < EPS);
    for (int i = 52; i < 80; i++) assert(fabsf(od[i] - 2.0f) < EPS);
    printf("OK\n");
}

static void test_cat_negative_dim(void) {
    printf("  test_cat_negative_dim... ");
    /* Use compatible shapes: cat along dim=0, so dim=1 must match */
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 0);
    tensor *b = tensor_zeros(ctx.params, 2, (int[]){4, 3}, 0);

    /* dim=-2 → dim=0: cat [2,3]+[4,3] along rows → [6,3] */
    tensor *out = tensor_cat(ctx.scratch, a, b, -2);
    assert(tensor_shape(out, 0) == 6 && tensor_shape(out, 1) == 3);

    /* dim=-1 → dim=1 would require matching dim=0 shapes, not tested here */
    printf("OK\n");
}

/* ── Backward tests ── */

static void test_cat_backward_simple(void) {
    printf("  test_cat_backward_simple... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){2}, 1);
    float *ad = tensor_data_ptr(a); ad[0]=1; ad[1]=2; ad[2]=3;
    float *bd = tensor_data_ptr(b); bd[0]=4; bd[1]=5;

    tensor *out = tensor_cat(ctx.scratch, a, b, 0);
    dnn_backward(ctx.scratch, out);  /* d_out = ones */

    float exp_a[] = {1.0f, 1.0f, 1.0f};
    float exp_b[] = {1.0f, 1.0f};
    check_grad_ary(a, exp_a, 3, "a.grad");
    check_grad_ary(b, exp_b, 2, "b.grad");
    printf("OK\n");
}

static void test_cat_backward_2d_dim0(void) {
    printf("  test_cat_backward_2d_dim0... ");
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    tensor *b = tensor_zeros(ctx.params, 2, (int[]){4, 3}, 1);

    tensor *out = tensor_cat(ctx.scratch, a, b, 0);
    dnn_backward(ctx.scratch, out);

    float exp_a[] = {1,1,1, 1,1,1};
    float exp_b[] = {1,1,1, 1,1,1, 1,1,1, 1,1,1};
    check_grad_ary(a, exp_a, 6, "a.grad");
    check_grad_ary(b, exp_b, 12, "b.grad");
    printf("OK\n");
}

static void test_cat_backward_2d_dim1(void) {
    printf("  test_cat_backward_2d_dim1... ");
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 1);
    tensor *b = tensor_zeros(ctx.params, 2, (int[]){3, 5}, 1);

    tensor *out = tensor_cat(ctx.scratch, a, b, 1);
    dnn_backward(ctx.scratch, out);

    float exp_a[] = {1,1, 1,1, 1,1};
    float exp_b[] = {1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1};
    check_grad_ary(a, exp_a, 6, "a.grad");
    check_grad_ary(b, exp_b, 15, "b.grad");
    printf("OK\n");
}

static void test_cat_backward_partial_grad(void) {
    printf("  test_cat_backward_partial_grad... ");
    /* only a requires grad */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){2}, 0);
    float *ad = tensor_data_ptr(a); ad[0]=1; ad[1]=2; ad[2]=3;
    float *bd = tensor_data_ptr(b); bd[0]=4; bd[1]=5;

    tensor *out = tensor_cat(ctx.scratch, a, b, 0);
    dnn_backward(ctx.scratch, out);

    float exp_a[] = {1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 3, "a.grad");
    assert(tensor_grad(b) == NULL && "b should not have grad");
    printf("OK\n");
}

static void test_cat_backward_self(void) {
    printf("  test_cat_backward_self... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){2}, 1);
    float *ad = tensor_data_ptr(a); ad[0]=1; ad[1]=2;

    tensor *out = tensor_cat(ctx.scratch, a, a, 0);  /* self-concatenation */
    assert(tensor_shape(out, 0) == 4);
    float *od = tensor_data_ptr(out);
    assert(fabsf(od[0]-1)<EPS && fabsf(od[1]-2)<EPS && fabsf(od[2]-1)<EPS && fabsf(od[3]-2)<EPS);

    /* loss = sum(out), gradient = ones(4) */
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

    /* Each element of a appears twice in out, so gradient = 2 */
    float exp_a[] = {2.0f, 2.0f};
    check_grad_ary(a, exp_a, 2, "a.grad");
    printf("OK\n");
}

static void test_cat_backward_sum_loss(void) {
    printf("  test_cat_backward_sum_loss... ");
    /* cat -> sum loss = scalar */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){2}, 1);
    float *ad = tensor_data_ptr(a); ad[0]=1; ad[1]=2; ad[2]=3;
    float *bd = tensor_data_ptr(b); bd[0]=4; bd[1]=5;

    tensor *out = tensor_cat(ctx.scratch, a, b, 0);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);  /* scalar, keepdim */
    dnn_backward(ctx.scratch, loss);

    /* grad everywhere = 1.0f (sum backprop) */
    float exp_a[] = {1.0f, 1.0f, 1.0f};
    float exp_b[] = {1.0f, 1.0f};
    check_grad_ary(a, exp_a, 3, "a.grad");
    check_grad_ary(b, exp_b, 2, "b.grad");
    printf("OK\n");
}

static void test_cat_no_grad(void) {
    printf("  test_cat_no_grad... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){2}, 1);
    float *ad = tensor_data_ptr(a); ad[0]=1; ad[1]=2; ad[2]=3;
    float *bd = tensor_data_ptr(b); bd[0]=4; bd[1]=5;

    tensor *out;
    {
        dnn_grad_ctx gc = dnn_no_grad_enter();
        out = tensor_cat(ctx.scratch, a, b, 0);
        dnn_no_grad_exit(gc);
    }

    /* forward still works */
    assert(tensor_shape(out, 0) == 5);
    float exp[] = {1, 2, 3, 4, 5};
    check_data_ary(out, exp, 5, "out");

    /* no grad_fn on output */
    assert(out->grad_fn == NULL && "no-grad: grad_fn should be NULL");
    printf("OK\n");
}

static void test_cat_chain_matmul(void) {
    printf("  test_cat_chain_matmul... ");
    /* cat -> matmul -> loss, verify grads flow through both paths */
    tensor *a = tensor_zeros(ctx.params, 3, (int[]){2, 3, 4}, 1);
    tensor *b = tensor_zeros(ctx.params, 3, (int[]){2, 3, 4}, 1);
    float *ad = tensor_data_ptr(a);
    float *bd = tensor_data_ptr(b);
    for (int i = 0; i < 24; i++) ad[i] = 1.0f;
    for (int i = 0; i < 24; i++) bd[i] = 2.0f;

    /* linear weight */
    linear *l = linear_create(ctx.params, 4, 4);
    float *wp = tensor_data_ptr(l->weight);
    /* identity weight */
    memset(wp, 0, 16 * sizeof(float));
    for (int i = 0; i < 4; i++) wp[i * 4 + i] = 1.0f;
    float *bp = tensor_data_ptr(l->bias);
    memset(bp, 0, 4 * sizeof(float));

    tensor *cat_out = tensor_cat(ctx.scratch, a, b, 1);  /* [2, 6, 4] */
    tensor *y = linear_forward(ctx.scratch, l, cat_out);  /* [2, 6, 4] */
    dnn_backward(ctx.scratch, y);

    /* grads should flow back to both a and b */
    float *a_grad = tensor_grad(a);
    float *b_grad = tensor_grad(b);
    assert(a_grad != NULL && "a.grad should exist");
    assert(b_grad != NULL && "b.grad should exist");
    assert(fabsf(a_grad[0]) > 0.0f && "a.grad[0] non-zero");
    assert(fabsf(b_grad[0]) > 0.0f && "b.grad[0] non-zero");
    printf("OK\n");
}

int main(void) {
    printf("test_cat:\n");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 2 * 1024 * 1024, 2 * 1024 * 1024);

    /* ── Forward ── */
    test_cat_1d();                mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_2d_dim0();           mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_2d_dim1();           mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_3d_dim1();           mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_negative_dim();      mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);

    /* ── Backward ── */
    test_cat_backward_simple();   mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_backward_2d_dim0();  mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_backward_2d_dim1();  mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_backward_partial_grad(); mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_backward_self();     mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_backward_sum_loss(); mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_no_grad();           mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);
    test_cat_chain_matmul();      mem_pool_reset(ctx.params); mem_pool_reset(ctx.scratch);

    printf("  ALL PASS\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
