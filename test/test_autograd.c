#include "dnn.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#define EPS 1e-5f

/* helper: create scalar leaf with given value */
static tensor *scalar(float v, int rg) {
    tensor *t = tensor_zeros(1, (int[]){1}, rg);
    float *p = tensor_data_ptr(t);
    p[0] = v;
    return t;
}

/* helper: check scalar grad is as expected */
static void check_grad(tensor *t, float expected, const char *label) {
    float *g = tensor_grad(t);
    if (!g) { printf("    FAIL: %s: grad is NULL\n", label); assert(0); }
    if (fabsf(g[0] - expected) > EPS) {
        printf("    FAIL: %s: got %.4f, expected %.4f\n", label, g[0], expected);
        assert(0);
    }
}

/* helper: check tensor grad elements */
static void check_grad_ary(tensor *t, const float *exp, int n, const char *label) {
    float *g = tensor_grad(t);
    if (!g) { printf("    FAIL: %s: grad is NULL\n", label); assert(0); }
    for (int i = 0; i < n; i++) {
        if (fabsf(g[i] - exp[i]) > EPS) {
            printf("    FAIL: %s[%d]: got %.4f, expected %.4f\n", label, i, g[i], exp[i]);
            assert(0);
        }
    }
}

static void test_add_simple(void) {
    printf("  test_add_simple... ");
    tensor *a = scalar(1.0f, 1);
    tensor *b = scalar(2.0f, 1);
    tensor *c = tensor_add(a, b);
    dnn_backward(c);
    check_grad(a, 1.0f, "da");
    check_grad(b, 1.0f, "db");
    printf("OK\n");
}

static void test_add_chain(void) {
    printf("  test_add_chain... ");
    tensor *a = scalar(1.0f, 1);
    tensor *b = scalar(2.0f, 1);
    tensor *c = scalar(3.0f, 1);
    tensor *d = tensor_add(a, b);
    tensor *e = tensor_add(d, c);
    dnn_backward(e);
    check_grad(a, 1.0f, "da");
    check_grad(b, 1.0f, "db");
    check_grad(c, 1.0f, "dc");
    printf("OK\n");
}

static void test_add_multi_use(void) {
    printf("  test_add_multi_use... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_add(a, a);     /* a + a = 2a */
    dnn_backward(c);
    check_grad(a, 2.0f, "da");        /* d(2a)/da = 2 */
    printf("OK\n");
}

static void test_add_diamond(void) {
    printf("  test_add_diamond... ");
    /*  a ──┐        ┌── d ──┐
     *      ├── add ─┤       ├── add ── loss
     *  b ──┘        └── e ──┘
     *       c ──────┤
     * d = a + b, e = a + c, loss = d + e
     * dloss/da = dd/da + de/da = 1 + 1 = 2
     * dloss/db = 1, dloss/dc = 1
     */
    tensor *a = scalar(1.0f, 1);
    tensor *b = scalar(2.0f, 1);
    tensor *c = scalar(3.0f, 1);
    tensor *d = tensor_add(a, b);
    tensor *e = tensor_add(a, c);
    tensor *loss = tensor_add(d, e);
    dnn_backward(loss);
    check_grad(a, 2.0f, "da");
    check_grad(b, 1.0f, "db");
    check_grad(c, 1.0f, "dc");
    printf("OK\n");
}

static void test_add_broadcast(void) {
    printf("  test_add_broadcast... ");
    /* a shape [3], b shape [1] → broadcast to [3] */
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    tensor *b = tensor_zeros(1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 10.0f;

    tensor *c = tensor_add(a, b);
    dnn_backward(c);

    /* a: 3 outputs each receiving 1 grad = [1,1,1] */
    float exp_a[] = {1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 3, "da");
    /* b: 3 contributions summed = 3 */
    check_grad(b, 3.0f, "db");
    printf("OK\n");
}

static void test_no_grad(void) {
    printf("  test_no_grad... ");
    tensor *a = scalar(10.0f, 1);
    tensor *b = scalar(20.0f, 1);

    dnn_grad_ctx ctx = dnn_no_grad_enter();
    tensor *c = tensor_add(a, b);
    dnn_no_grad_exit(ctx);

    /* inside no_grad: c should NOT have grad_fn */
    assert(c->grad_fn == NULL && "c has grad_fn despite no_grad");
    printf("OK\n");
}

static void test_add_self_via_grad_fn(void) {
    printf("  test_add_self_via_grad_fn... ");
    /* Use c = tensor_add(a, a) but interpose a non-
     * trivial op so a == b check in add_backward fires */
    tensor *a = scalar(7.0f, 1);
    tensor *c = tensor_add(a, a);
    dnn_backward(c);
    check_grad(a, 2.0f, "da");
    printf("OK\n");
}

/* ── Multiplication tests ── */

static void test_mul_simple(void) {
    printf("  test_mul_simple... ");
    /* c = a * b, d = a + c (to get non-unit gout) */
    tensor *a = scalar(3.0f, 1);
    tensor *b = scalar(4.0f, 1);
    tensor *c = tensor_mul(a, b);  /* 12 */
    dnn_backward(c);
    check_grad(a, 4.0f, "da");    /* d(a*b)/da = b */
    check_grad(b, 3.0f, "db");    /* d(a*b)/db = a */
    printf("OK\n");
}

static void test_mul_chain(void) {
    printf("  test_mul_chain... ");
    /* d = a * b * c */
    tensor *a = scalar(2.0f, 1);
    tensor *b = scalar(3.0f, 1);
    tensor *c = scalar(4.0f, 1);
    tensor *d = tensor_mul(tensor_mul(a, b), c);  /* 24 */
    dnn_backward(d);
    check_grad(a, 12.0f, "da");   /* d(a*b*c)/da = b*c */
    check_grad(b, 8.0f,  "db");   /* d(a*b*c)/db = a*c */
    check_grad(c, 6.0f,  "dc");   /* d(a*b*c)/dc = a*b */
    printf("OK\n");
}

static void test_mul_self(void) {
    printf("  test_mul_self... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_mul(a, a);  /* a*a = 25 */
    dnn_backward(c);
    check_grad(a, 10.0f, "da");   /* d(a*a)/da = 2*a = 10 */
    printf("OK\n");
}

static void test_mul_broadcast(void) {
    printf("  test_mul_broadcast... ");
    /* a shape [3], b shape [1] → broadcast to [3] */
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    tensor *b = tensor_zeros(1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 2.0f; ap[1] = 3.0f; ap[2] = 4.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 10.0f;

    tensor *c = tensor_mul(a, b);
    dnn_backward(c);

    /* da = b * gout = [10, 10, 10] */
    float exp_a[] = {10.0f, 10.0f, 10.0f};
    check_grad_ary(a, exp_a, 3, "da");
    /* db = sum(a * gout) = 2+3+4 = 9 */
    check_grad(b, 9.0f, "db");
    printf("OK\n");
}

static void test_mul_with_add(void) {
    printf("  test_mul_with_add... ");
    /* mix mul and add to exercise gradient flow through both ops */
    tensor *a = scalar(2.0f, 1);
    tensor *b = scalar(3.0f, 1);
    tensor *c = scalar(4.0f, 1);
    tensor *ab = tensor_mul(a, b);   /* 6 */
    tensor *out = tensor_add(ab, c); /* 10 */
    dnn_backward(out);
    check_grad(a, 3.0f, "da");      /* d(out)/da = b */
    check_grad(b, 2.0f, "db");      /* d(out)/db = a */
    check_grad(c, 1.0f, "dc");      /* d(out)/dc = 1 */
    printf("OK\n");
}

/* ── Pow tests ── */

static void test_pow_simple(void) {
    printf("  test_pow_simple... ");
    /* c = a^2 */
    tensor *a = scalar(3.0f, 1);
    tensor *c = tensor_pow(a, 2.0f);  /* 9 */
    dnn_backward(c);
    check_grad(a, 6.0f, "da");       /* d(a^2)/da = 2*a = 6 */
    printf("OK\n");
}

static void test_pow_cube(void) {
    printf("  test_pow_cube... ");
    /* c = a^3 */
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_pow(a, 3.0f);  /* 125 */
    dnn_backward(c);
    check_grad(a, 75.0f, "da");      /* d(a^3)/da = 3*a^2 = 75 */
    printf("OK\n");
}

static void test_pow_exp1(void) {
    printf("  test_pow_exp1... ");
    /* c = a^1 — derivative is 1 */
    tensor *a = scalar(42.0f, 1);
    tensor *c = tensor_pow(a, 1.0f);
    dnn_backward(c);
    check_grad(a, 1.0f, "da");
    printf("OK\n");
}

static void test_pow_neg(void) {
    printf("  test_pow_neg... ");
    /* c = a^(-1) = 1/a */
    tensor *a = scalar(4.0f, 1);
    tensor *c = tensor_pow(a, -1.0f);  /* 0.25 */
    dnn_backward(c);
    check_grad(a, -0.0625f, "da");    /* d(a^-1)/da = -1*a^-2 = -1/16 = -0.0625 */
    printf("OK\n");
}

static void test_pow_broadcast(void) {
    printf("  test_pow_broadcast... ");
    /* a shape [3], c = a^2 */
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;

    tensor *c = tensor_pow(a, 2.0f);
    dnn_backward(c);

    /* da = 2*a = [2, 4, 6] */
    float exp_a[] = {2.0f, 4.0f, 6.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

/* ── Neg tests ── */

static void test_neg_simple(void) {
    printf("  test_neg_simple... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_neg(a);  /* -5 */
    dnn_backward(c);
    check_grad(a, -1.0f, "da");  /* d(-a)/da = -1 */
    printf("OK\n");
}

static void test_neg_broadcast(void) {
    printf("  test_neg_broadcast... ");
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = -2.0f; ap[2] = 3.0f;

    tensor *c = tensor_neg(a);
    dnn_backward(c);

    float exp_a[] = {-1.0f, -1.0f, -1.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

/* ── Sub tests ── */

static void test_sub_simple(void) {
    printf("  test_sub_simple... ");
    tensor *a = scalar(7.0f, 1);
    tensor *b = scalar(3.0f, 1);
    tensor *c = tensor_sub(a, b);  /* 4 */
    dnn_backward(c);
    check_grad(a, 1.0f, "da");   /* d(a-b)/da = 1 */
    check_grad(b, -1.0f, "db");  /* d(a-b)/db = -1 */
    printf("OK\n");
}

static void test_sub_self(void) {
    printf("  test_sub_self... ");
    /* a - a = 0, gradient is 0 */
    tensor *a = scalar(10.0f, 1);
    tensor *c = tensor_sub(a, a);
    dnn_backward(c);
    check_grad(a, 0.0f, "da");
    printf("OK\n");
}

static void test_sub_broadcast(void) {
    printf("  test_sub_broadcast... ");
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    tensor *b = tensor_zeros(1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 10.0f; ap[1] = 20.0f; ap[2] = 30.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 5.0f;

    tensor *c = tensor_sub(a, b);
    dnn_backward(c);

    /* da = gout = [1,1,1] */
    float exp_a[] = {1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 3, "da");
    /* db = sum(-gout) = -3 */
    check_grad(b, -3.0f, "db");
    printf("OK\n");
}

/* ── Div tests ── */

static void test_div_simple(void) {
    printf("  test_div_simple... ");
    tensor *a = scalar(10.0f, 1);
    tensor *b = scalar(2.0f, 1);
    tensor *c = tensor_div(a, b);  /* 5 */
    dnn_backward(c);
    check_grad(a, 0.5f, "da");     /* d(a/b)/da = 1/b = 0.5 */
    check_grad(b, -2.5f, "db");   /* d(a/b)/db = -a/b^2 = -10/4 = -2.5 */
    printf("OK\n");
}

static void test_div_self(void) {
    printf("  test_div_self... ");
    /* a/a = 1, gradient is 0 */
    tensor *a = scalar(7.0f, 1);
    tensor *c = tensor_div(a, a);
    dnn_backward(c);
    check_grad(a, 0.0f, "da");
    printf("OK\n");
}

static void test_div_broadcast(void) {
    printf("  test_div_broadcast... ");
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    tensor *b = tensor_zeros(1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 6.0f; ap[1] = 9.0f; ap[2] = 12.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 3.0f;

    tensor *c = tensor_div(a, b);
    dnn_backward(c);

    /* da = 1/b = [1/3, 1/3, 1/3] */
    float exp_a[] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
    check_grad_ary(a, exp_a, 3, "da");
    /* db = sum(-a/b^2) = -(6+9+12)/9 = -27/9 = -3 */
    check_grad(b, -3.0f, "db");
    printf("OK\n");
}

static void test_div_neg_b(void) {
    printf("  test_div_neg_b... ");
    /* a / (-b) to check division by negative */
    tensor *a = scalar(8.0f, 1);
    tensor *b = scalar(-2.0f, 1);
    tensor *c = tensor_div(a, b);  /* -4 */
    dnn_backward(c);
    check_grad(a, -0.5f, "da");    /* 1/b = 1/(-2) = -0.5 */
    check_grad(b, -2.0f, "db");    /* -a/b^2 = -8/4 = -2 */
    printf("OK\n");
}

/* ── Sum tests ── */

static void test_sum_simple(void) {
    printf("  test_sum_simple... ");
    /* a shape [3], sum over dim=0 → scalar (shape [1]) */
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;

    tensor *c = tensor_sum(a, 0);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 6.0f) < EPS && "sum forward wrong");

    dnn_backward(c);

    /* da = gout broadcast = [1, 1, 1] */
    float exp_a[] = {1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

static void test_sum_2d_dim0(void) {
    printf("  test_sum_2d_dim0... ");
    /* a shape [2,3], sum over dim=0 → shape [1,3] */
    tensor *a = tensor_zeros(2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 6; i++) ap[i] = (float)(i + 1);  /* [[1,2,3],[4,5,6]] */

    tensor *c = tensor_sum(a, 0);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 5.0f) < EPS);  /* 1+4 */
    assert(fabsf(cp[1] - 7.0f) < EPS);  /* 2+5 */
    assert(fabsf(cp[2] - 9.0f) < EPS);  /* 3+6 */

    dnn_backward(c);

    /* dnn_backward sets grad of c to all-1s → gout = [1,1,1] broadcast to [2,3] */
    float exp_a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

static void test_sum_2d_dim1(void) {
    printf("  test_sum_2d_dim1... ");
    /* a shape [2,3], sum over dim=1 → shape [2,1] */
    tensor *a = tensor_zeros(2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 6; i++) ap[i] = (float)(i + 1);

    tensor *c = tensor_sum(a, 1);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 6.0f) < EPS);   /* 1+2+3 */
    assert(fabsf(cp[1] - 15.0f) < EPS);  /* 4+5+6 */

    dnn_backward(c);

    /* gout = [1,1] (shape [2,1]), broadcast along dim 1 → all 1s */
    float exp_a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

static void test_sum_chain(void) {
    printf("  test_sum_chain... ");
    /* sum then mul — gradient flows through both */
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 2.0f; ap[1] = 3.0f; ap[2] = 4.0f;
    tensor *b = scalar(10.0f, 1);

    tensor *s = tensor_sum(a, 0);   /* 9, shape [1] */
    tensor *c = tensor_mul(s, b);   /* 90 */
    dnn_backward(c);

    /* da = b * ds/da = 10 * [1,1,1] = [10,10,10] */
    float exp_a[] = {10.0f, 10.0f, 10.0f};
    check_grad_ary(a, exp_a, 3, "da");
    check_grad(b, 9.0f, "db");
    printf("OK\n");
}

/* ── Mean tests ── */

static void test_mean_simple(void) {
    printf("  test_mean_simple... ");
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 2.0f; ap[1] = 4.0f; ap[2] = 6.0f;

    tensor *c = tensor_mean(a, 0);   /* shape [1], value 4 */
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 4.0f) < EPS && "mean forward wrong");

    dnn_backward(c);

    /* da = gout * 1/n = 1/3 for each element */
    float exp_a[] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

static void test_mean_2d_dim1(void) {
    printf("  test_mean_2d_dim1... ");
    /* a shape [2,3], mean over dim=1 → shape [2,1] */
    tensor *a = tensor_zeros(2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=6;
    ap[3]=2; ap[4]=4; ap[5]=6;
    /* row 0 mean = 3, row 1 mean = 4 */

    tensor *c = tensor_mean(a, 1);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 3.0f) < EPS);
    assert(fabsf(cp[1] - 4.0f) < EPS);

    dnn_backward(c);

    /* da = gout * 1/3 broadcast along dim 1 */
    float exp_a[] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

/* ── ReLU tests ── */

static void test_relu_positive(void) {
    printf("  test_relu_positive... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_relu(a);  /* 5 */
    dnn_backward(c);
    check_grad(a, 1.0f, "da");  /* drelu(5)/da = 1 */
    printf("OK\n");
}

static void test_relu_negative(void) {
    printf("  test_relu_negative... ");
    tensor *a = scalar(-3.0f, 1);
    tensor *c = tensor_relu(a);  /* 0 */
    dnn_backward(c);
    check_grad(a, 0.0f, "da");  /* drelu(-3)/da = 0 */
    printf("OK\n");
}

static void test_relu_mixed(void) {
    printf("  test_relu_mixed... ");
    tensor *a = tensor_zeros(1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = -1.0f; ap[1] = 0.0f; ap[2] = 2.0f; ap[3] = -5.0f;

    tensor *c = tensor_relu(a);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 0.0f) < EPS);
    assert(fabsf(cp[1] - 0.0f) < EPS);
    assert(fabsf(cp[2] - 2.0f) < EPS);
    assert(fabsf(cp[3] - 0.0f) < EPS);

    dnn_backward(c);

    /* grad = 1 for positive inputs, 0 otherwise */
    float exp_a[] = {0.0f, 0.0f, 1.0f, 0.0f};
    check_grad_ary(a, exp_a, 4, "da");
    printf("OK\n");
}

static void test_relu_chain(void) {
    printf("  test_relu_chain... ");
    /* c = relu(a) + b, grad flows through both paths */
    tensor *a = scalar(-2.0f, 1);
    tensor *b = scalar(5.0f, 1);
    tensor *ra = tensor_relu(a);  /* 0 */
    tensor *c = tensor_add(ra, b);  /* 5 */
    dnn_backward(c);
    check_grad(a, 0.0f, "da");  /* relu(-2)=0, derivative 0 */
    check_grad(b, 1.0f, "db");
    printf("OK\n");
}

/* ── Matmul tests ── */

static void test_matmul_simple(void) {
    printf("  test_matmul_simple... ");
    /* A (2,3) @ B (3,2) = C (2,2) */
    tensor *a = tensor_zeros(2, (int[]){2, 3}, 1);
    tensor *b = tensor_zeros(2, (int[]){3, 2}, 1);
    float *ap = tensor_data_ptr(a);
    float *bp = tensor_data_ptr(b);
    ap[0]=1; ap[1]=2; ap[2]=3; ap[3]=4; ap[4]=5; ap[5]=6;
    bp[0]=7; bp[1]=8; bp[2]=9; bp[3]=10; bp[4]=11; bp[5]=12;

    tensor *c = tensor_matmul(a, b);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 58.0f) < EPS);
    assert(fabsf(cp[1] - 64.0f) < EPS);
    assert(fabsf(cp[3] - 154.0f) < EPS);  /* cp[3] = (1,1) = 154 */

    dnn_backward(c);

    /* da = gd @ B^T = [[1,1],[1,1]] @ B^T */
    float exp_a[] = {15.0f, 19.0f, 23.0f, 15.0f, 19.0f, 23.0f};
    check_grad_ary(a, exp_a, 6, "da");
    /* db = A^T @ gd */
    float exp_b[] = {5.0f, 5.0f, 7.0f, 7.0f, 9.0f, 9.0f};
    check_grad_ary(b, exp_b, 6, "db");
    printf("OK\n");
}

static void test_matmul_linear(void) {
    printf("  test_matmul_linear... ");
    /* y = x @ W + b — one step of linear layer */
    tensor *x = tensor_zeros(2, (int[]){2, 3}, 1);
    tensor *W = tensor_zeros(2, (int[]){3, 2}, 1);
    tensor *b = scalar(1.0f, 1);
    float *xp = tensor_data_ptr(x);
    float *Wp = tensor_data_ptr(W);
    xp[0]=1; xp[1]=2; xp[2]=3; xp[3]=4; xp[4]=5; xp[5]=6;
    Wp[0]=1; Wp[1]=0; Wp[2]=0; Wp[3]=1; Wp[4]=1; Wp[5]=0;

    tensor *mm = tensor_matmul(x, W);  /* (2,2) */
    tensor *y = tensor_add(mm, b);     /* broadcast b to (2,2) */
    dnn_backward(y);

    /* y = x@W + b, dy/dx = W^T, dloss/dy = all-1s */
    float exp_x[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check_grad_ary(x, exp_x, 6, "dx");
    /* dy/dW = X^T, dloss/dy = all-1s */
    /* X^T @ ones = [[1+4,1+4],[2+5,2+5],[3+6,3+6]] = [[5,5],[7,7],[9,9]] */
    float exp_W[] = {5.0f, 5.0f, 7.0f, 7.0f, 9.0f, 9.0f};
    check_grad_ary(W, exp_W, 6, "dW");
    /* db = sum(grad_output) along batch dims */
    check_grad(b, 4.0f, "db");  /* 4 elements in y, each receives grad 1 */
    printf("OK\n");
}

static void test_matmul_square_self(void) {
    printf("  test_matmul_square_self... ");
    /* A (2,2) @ A (2,2), A == B (same pointer) */
    tensor *a = tensor_zeros(2, (int[]){2, 2}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=3; ap[3]=4;

    tensor *c = tensor_matmul(a, a);
    float *cp = tensor_data_ptr(c);
    /* A@A = [[7,10],[15,22]] */
    assert(fabsf(cp[0] - 7.0f) < EPS);
    assert(fabsf(cp[1] - 10.0f) < EPS);
    assert(fabsf(cp[2] - 15.0f) < EPS);
    assert(fabsf(cp[3] - 22.0f) < EPS);

    dnn_backward(c);

    /* da = dC@A^T + A^T@dC = [[3,7],[3,7]] + [[4,4],[6,6]] = [[7,11],[9,13]] */
    float exp_a[] = {7.0f, 11.0f, 9.0f, 13.0f};
    check_grad_ary(a, exp_a, 4, "da");
    printf("OK\n");
}

/* ── Fundamental correctness tests ── */

static void test_mixed_diamond(void) {
    printf("  test_mixed_diamond... ");
    /* a mul(b) + a add(b) — a receives grad from two different op types */
    tensor *a = scalar(2.0f, 1);
    tensor *b = scalar(3.0f, 1);
    tensor *c = tensor_mul(a, b);   /* 6 */
    tensor *d = tensor_add(a, b);   /* 5, but unused in grad check — just for diamond */
    tensor *e = tensor_add(c, d);   /* 11 */
    dnn_backward(e);
    check_grad(a, 4.0f, "da");     /* da = d(c)/da + d(d)/da = b + 1 = 4 */
    check_grad(b, 3.0f, "db");     /* db = d(c)/db + d(d)/db = a + 1 = 3 */
    printf("OK\n");
}

static void test_transpose_backward(void) {
    printf("  test_transpose_backward... ");
    /* a (2,3) → transpose(a,0,1) → sum(b,0) — b view has swapped strides */
    tensor *a = tensor_zeros(2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 6; i++) ap[i] = (float)(i + 1);

    tensor *b = tensor_transpose(a, 0, 1);  /* shape (3,2), strides (1,3) */
    tensor *c = tensor_sum(b, 0);           /* shape (1,2) */
    dnn_backward(c);

    /* grad_output = [1,1], broadcast via sum_backward:
       each input element used once → grad all 1s */
    float exp_a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

static void test_slice_backward(void) {
    printf("  test_slice_backward... ");
    /* a (4) → slice at offset 1, len 2 → sum(b, 0) → grad only flows to sliced region */
    tensor *a = tensor_zeros(1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=10; ap[1]=20; ap[2]=30; ap[3]=40;

    tensor *b = tensor_slice(a, 0, 1, 2);  /* shape (2), offset 1, values [20,30] */
    tensor *c = tensor_sum(b, 0);           /* shape (1), value 50 */
    dnn_backward(c);

    /* only sliced positions get gradient = 1 */
    float exp_a[] = {0.0f, 1.0f, 1.0f, 0.0f};
    check_grad_ary(a, exp_a, 4, "da");
    printf("OK\n");
}

static void test_mul_diamond_self(void) {
    printf("  test_mul_diamond_self... ");
    /* a used in mul(a,a) AND add(a,...) — tests multi-path with a==b case in mul */
    tensor *a = scalar(3.0f, 1);
    tensor *c = tensor_mul(a, a);       /* 9, grad contribution: d(a*a)/da = 6 */
    tensor *d = tensor_add(a, c);       /* 12, grad contribution: d(a)/da = 1 */
    dnn_backward(d);
    check_grad(a, 7.0f, "da");         /* da = 1 + 2*3 = 7 */
    printf("OK\n");
}

/* ── Softmax tests ── */

static void softmax_ref_1d(const float *x, float *y, int n) {
    /* compute numerically stable softmax */
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { y[i] = expf(x[i] - maxv); sum += y[i]; }
    for (int i = 0; i < n; i++) y[i] /= sum;
}

static void test_softmax_1d(void) {
    printf("  test_softmax_1d... ");
    /* a shape [4], softmax over dim=0 */
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_ref[4];
    softmax_ref_1d(x, y_ref, 4);

    tensor *a = tensor_zeros(1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 4; i++) ap[i] = x[i];

    tensor *c = tensor_softmax(a, 0);
    float *cp = tensor_data_ptr(c);

    for (int i = 0; i < 4; i++)
        assert(fabsf(cp[i] - y_ref[i]) < EPS && "softmax 1d forward");

    /* sum of output should be 1 */
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += cp[i];
    assert(fabsf(sum - 1.0f) < EPS && "softmax 1d sum=1");

    dnn_backward(c);

    /* analytical gradient check:
       dx_i = sm_i * (dg_i - sum_j(sm_j * dg_j))
       dg = all-1s → dx_i = sm_i * (1 - sum_j(sm_j)) = sm_i * (1 - 1) = 0
       So all gradients should be near 0 */
    float *ag = tensor_grad(a);
    for (int i = 0; i < 4; i++)
        assert(fabsf(ag[i]) < EPS && "softmax 1d grad with unit gout");

    printf("OK\n");
}

static void test_softmax_2d_dim0(void) {
    printf("  test_softmax_2d_dim0... ");
    /* a shape [2,3], softmax over dim=0 */
    tensor *a = tensor_zeros(2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=3;
    ap[3]=4; ap[4]=5; ap[5]=6;

    tensor *c = tensor_softmax(a, 0);
    float *cp = tensor_data_ptr(c);

    /* each column (dim 0) should sum to 1 */
    float sum0 = cp[0] + cp[3];
    float sum1 = cp[1] + cp[4];
    float sum2 = cp[2] + cp[5];
    assert(fabsf(sum0 - 1.0f) < EPS && "softmax 2d dim0 col0");
    assert(fabsf(sum1 - 1.0f) < EPS && "softmax 2d dim0 col1");
    assert(fabsf(sum2 - 1.0f) < EPS && "softmax 2d dim0 col2");

    /* rows not constrained to sum to 1 (just a smoke check) */
    (void)(cp[0]+cp[1]+cp[2]);

    dnn_backward(c);
    float *ag = tensor_grad(a);
    /* with unit gout, dx = sm_i * (1 - sum_j(sm_j)) = 0 for each column */
    for (int i = 0; i < 6; i++)
        assert(fabsf(ag[i]) < EPS && "softmax 2d dim0 grad");

    printf("OK\n");
}

static void test_softmax_2d_dim1(void) {
    printf("  test_softmax_2d_dim1... ");
    /* a shape [2,3], softmax over dim=1 */
    tensor *a = tensor_zeros(2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=3;
    ap[3]=4; ap[4]=5; ap[5]=6;

    tensor *c = tensor_softmax(a, 1);
    float *cp = tensor_data_ptr(c);

    /* each row (dim 1) should sum to 1 */
    float row0 = cp[0] + cp[1] + cp[2];
    float row1 = cp[3] + cp[4] + cp[5];
    assert(fabsf(row0 - 1.0f) < EPS && "softmax 2d dim1 row0");
    assert(fabsf(row1 - 1.0f) < EPS && "softmax 2d dim1 row1");

    dnn_backward(c);
    float *ag = tensor_grad(a);
    /* with unit gout, dx = 0 */
    for (int i = 0; i < 6; i++)
        assert(fabsf(ag[i]) < EPS && "softmax 2d dim1 grad");

    printf("OK\n");
}

static void test_softmax_chain(void) {
    printf("  test_softmax_chain... ");
    /* softmax → mul — check non-unit gout gradient
       a (3) → softmax(dim=0) → s, b (1) → mul(s, b) → c
       dc/ds = b = 3
       dx_i = sm_i * (b - sum_j(sm_j * b)) = sm_i * (b - b * sum_j(sm_j)) = sm_i * (b - b) = 0
       Actually with broadcasting b, all sm_i get same grad b.
       So dx_i = sm_i * (b - b * 1) = 0 */
    tensor *a = tensor_zeros(1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;
    tensor *b = scalar(3.0f, 1);

    tensor *s = tensor_softmax(a, 0);
    tensor *c = tensor_mul(s, b);
    dnn_backward(c);

    float *ag = tensor_grad(a);
    for (int i = 0; i < 3; i++)
        assert(fabsf(ag[i]) < EPS && "softmax chain grad");
    check_grad(b, 1.0f, "db");  /* dc/db = sum(s) = 1 */
    printf("OK\n");
}

static void test_softmax_stability(void) {
    printf("  test_softmax_stability... ");
    /* large values should not overflow */
    tensor *a = tensor_zeros(1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 100.0f; ap[1] = 101.0f; ap[2] = 102.0f; ap[3] = 103.0f;

    tensor *c = tensor_softmax(a, 0);
    float *cp = tensor_data_ptr(c);

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        assert(isfinite(cp[i]) && "softmax stability: non-finite");
        sum += cp[i];
    }
    assert(fabsf(sum - 1.0f) < EPS && "softmax stability sum=1");

    /* also very negative values */
    ap[0] = -1000.0f; ap[1] = -1001.0f; ap[2] = -1002.0f; ap[3] = -1003.0f;
    tensor *d = tensor_softmax(a, 0);
    float *dp = tensor_data_ptr(d);
    sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        assert(isfinite(dp[i]) && "softmax stability neg: non-finite");
        sum += dp[i];
    }
    assert(fabsf(sum - 1.0f) < 10*EPS && "softmax stability neg sum=1");

    printf("OK\n");
}

static void test_softmax_sum_to_one(void) {
    printf("  test_softmax_sum_to_one... ");
    /* 3D tensor, verify sum along dim=1 equals 1 */
    tensor *a = tensor_zeros(3, (int[]){2, 3, 4}, 0);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 24; i++) ap[i] = (float)(i - 10);

    tensor *c = tensor_softmax(a, 1);
    float *cp = tensor_data_ptr(c);

    /* sum over dim=1 for each (d0,d2) pair */
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 4; k++) {
            float sum = 0.0f;
            for (int j = 0; j < 3; j++) {
                int idx = i * 3 * 4 + j * 4 + k;
                sum += cp[idx];
            }
            assert(fabsf(sum - 1.0f) < EPS && "softmax 3d sum=1");
        }
    }
    printf("OK\n");
}

/* ── Cross-entropy tests ── */

static float ce_ref_2d(const float *logits, int target, int n_classes) {
    float maxv = logits[0];
    for (int i = 1; i < n_classes; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n_classes; i++) sum += expf(logits[i] - maxv);
    return (maxv + logf(sum)) - logits[target];
}

static void test_cross_entropy_simple(void) {
    printf("  test_cross_entropy_simple... ");
    /* 2 classes, logits [1.0, 2.0], target=1 */
    tensor *logits = tensor_zeros(1, (int[]){2}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0] = 1.0f; lp[1] = 2.0f;

    tensor *target = tensor_zeros(1, (int[]){1}, 0);
    ((int*)tensor_data_ptr(target))[0] = 1;

    tensor *loss = tensor_cross_entropy(logits, target, 0);
    float *loss_p = tensor_data_ptr(loss);

    float expected = ce_ref_2d(lp, 1, 2);
    assert(fabsf(loss_p[0] - expected) < EPS && "ce simple forward");

    dnn_backward(loss);
    float *lg = tensor_grad(logits);
    assert(lg && "ce simple grad not null");

    /* dlogits = (softmax - one_hot) */
    float sm0 = expf(1.0f) / (expf(1.0f) + expf(2.0f));
    float sm1 = expf(2.0f) / (expf(1.0f) + expf(2.0f));
    assert(fabsf(lg[0] - sm0) < EPS && "ce simple dlogits[0]");
    assert(fabsf(lg[1] - (sm1 - 1.0f)) < EPS && "ce simple dlogits[1]");
    printf("OK\n");
}

static void test_cross_entropy_batch(void) {
    printf("  test_cross_entropy_batch... ");
    /* batch of 3, 4 classes, targets [0, 2, 1] */
    tensor *logits = tensor_zeros(2, (int[]){3, 4}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0]=2.0f;  lp[1]=1.0f;  lp[2]=0.5f; lp[3]=0.1f;   /* sample 0 */
    lp[4]=0.5f;  lp[5]=2.0f;  lp[6]=3.0f; lp[7]=1.0f;   /* sample 1 */
    lp[8]=1.0f;  lp[9]=3.0f;  lp[10]=2.0f;lp[11]=0.5f;  /* sample 2 */

    tensor *target = tensor_zeros(1, (int[]){3}, 0);
    int *td = (int*)tensor_data_ptr(target);
    td[0] = 0; td[1] = 2; td[2] = 1;

    tensor *loss = tensor_cross_entropy(logits, target, 1);
    float *loss_p = tensor_data_ptr(loss);

    /* compute expected: mean of per-sample losses */
    float l0 = ce_ref_2d(lp + 0, 0, 4);
    float l1 = ce_ref_2d(lp + 4, 2, 4);
    float l2 = ce_ref_2d(lp + 8, 1, 4);
    float expected = (l0 + l1 + l2) / 3.0f;
    assert(fabsf(loss_p[0] - expected) < EPS && "ce batch forward");

    dnn_backward(loss);
    float *lg = tensor_grad(logits);
    assert(lg && "ce batch grad not null");

    /* per-sample dlogits = (softmax - one_hot) / 3 */
    float inv3 = 1.0f / 3.0f;
    /* sample 0, target=0 */
    float s00 = expf(2.0f) / (expf(2.0f)+expf(1.0f)+expf(0.5f)+expf(0.1f));
    assert(fabsf(lg[0] - (s00 - 1.0f)*inv3) < EPS && "ce batch d[0]");
    printf("OK\n");
}

static void test_cross_entropy_grad(void) {
    printf("  test_cross_entropy_grad... ");
    /* 3 classes, batch of 2, verify all gradient components */
    tensor *logits = tensor_zeros(2, (int[]){2, 3}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0]=1.0f; lp[1]=2.0f; lp[2]=3.0f;
    lp[3]=4.0f; lp[4]=5.0f; lp[5]=6.0f;

    tensor *target = tensor_zeros(1, (int[]){2}, 0);
    int *td = (int*)tensor_data_ptr(target);
    td[0] = 1; td[1] = 0;

    tensor *loss = tensor_cross_entropy(logits, target, 1);
    dnn_backward(loss);
    float *lg = tensor_grad(logits);

    /* sample 0: target=1 */
    float max0 = lp[0] > lp[1] ? (lp[0] > lp[2] ? lp[0] : lp[2]) : (lp[1] > lp[2] ? lp[1] : lp[2]);
    float sum0 = expf(lp[0]-max0) + expf(lp[1]-max0) + expf(lp[2]-max0);
    float sm00 = expf(lp[0]-max0)/sum0;
    float sm01 = expf(lp[1]-max0)/sum0;
    float sm02 = expf(lp[2]-max0)/sum0;

    /* sample 1: target=0 */
    float max1 = lp[3] > lp[4] ? (lp[3] > lp[5] ? lp[3] : lp[5]) : (lp[4] > lp[5] ? lp[4] : lp[5]);
    float sum1 = expf(lp[3]-max1) + expf(lp[4]-max1) + expf(lp[5]-max1);
    float sm10 = expf(lp[3]-max1)/sum1;
    float sm11 = expf(lp[4]-max1)/sum1;
    float sm12 = expf(lp[5]-max1)/sum1;

    float inv2 = 0.5f;
    assert(fabsf(lg[0] - (sm00 - 0.0f)*inv2) < EPS && "ce grad[0][0]");
    assert(fabsf(lg[1] - (sm01 - 1.0f)*inv2) < EPS && "ce grad[0][1]");
    assert(fabsf(lg[2] - (sm02 - 0.0f)*inv2) < EPS && "ce grad[0][2]");
    assert(fabsf(lg[3] - (sm10 - 1.0f)*inv2) < EPS && "ce grad[1][0]");
    assert(fabsf(lg[4] - (sm11 - 0.0f)*inv2) < EPS && "ce grad[1][1]");
    assert(fabsf(lg[5] - (sm12 - 0.0f)*inv2) < EPS && "ce grad[1][2]");
    printf("OK\n");
}

static void test_cross_entropy_stability(void) {
    printf("  test_cross_entropy_stability... ");
    /* large logits should not overflow */
    tensor *logits = tensor_zeros(1, (int[]){3}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0] = 1000.0f; lp[1] = 1001.0f; lp[2] = 1002.0f;

    tensor *target = tensor_zeros(1, (int[]){1}, 0);
    ((int*)tensor_data_ptr(target))[0] = 2;

    tensor *loss = tensor_cross_entropy(logits, target, 0);
    float *loss_p = tensor_data_ptr(loss);
    assert(isfinite(loss_p[0]) && "ce stability");

    /* also very negative */
    lp[0] = -1000.0f; lp[1] = -1001.0f; lp[2] = -1002.0f;
    tensor *loss2 = tensor_cross_entropy(logits, target, 0);
    float *loss_p2 = tensor_data_ptr(loss2);
    assert(isfinite(loss_p2[0]) && "ce stability neg");
    printf("OK\n");
}

static void test_cross_entropy_chain(void) {
    printf("  test_cross_entropy_chain... ");
    /* cross_entropy * scalar → backward */
    tensor *logits = tensor_zeros(1, (int[]){4}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0]=1; lp[1]=2; lp[2]=3; lp[3]=4;

    tensor *target = tensor_zeros(1, (int[]){1}, 0);
    ((int*)tensor_data_ptr(target))[0] = 0;

    tensor *loss = tensor_cross_entropy(logits, target, 0);
    tensor *scale = scalar(2.0f, 0);
    tensor *scaled = tensor_mul(loss, scale);
    dnn_backward(scaled);

    float *lg = tensor_grad(logits);
    assert(lg && "ce chain grad");

    /* gradient should be 2x the non-scaled grad */
    float maxv = 4.0f;  /* max of [1,2,3,4] */
    float sum = expf(1-4)+expf(2-4)+expf(3-4)+expf(4-4);
    float sm[4];
    for (int i = 0; i < 4; i++) sm[i] = expf(lp[i]-maxv)/sum;
    float invN = 1.0f;
    for (int i = 0; i < 4; i++) {
        float expected = (sm[i] - (i == 0 ? 1.0f : 0.0f)) * 2.0f * invN;
        assert(fabsf(lg[i] - expected) < 2*EPS && "ce chain dlogits");
    }
    printf("OK\n");
}

static void test_3d_multi_op(void) {
    printf("  test_3d_multi_op... ");
    /* 3D tensor (2,3,4) → sum(dim=1) → pow(2) → relu → sum(dim=0) → loss */
    tensor *t = tensor_zeros(3, (int[]){2, 3, 4}, 1);
    float *tp = tensor_data_ptr(t);
    for (int i = 0; i < 24; i++) tp[i] = (float)(i + 1);

    tensor *s = tensor_sum(t, 1);        /* (2,1,4) */
    tensor *p = tensor_pow(s, 2.0f);     /* (2,1,4) */
    tensor *r = tensor_relu(p);          /* (2,1,4) */
    tensor *loss = tensor_sum(r, 0);     /* (1,1,4) */
    dnn_backward(loss);

    /* expected grad (see manual trace in conversation) */
    float exp_t[] = {
        30.0f, 36.0f, 42.0f, 48.0f,
        30.0f, 36.0f, 42.0f, 48.0f,
        30.0f, 36.0f, 42.0f, 48.0f,
        102.0f, 108.0f, 114.0f, 120.0f,
        102.0f, 108.0f, 114.0f, 120.0f,
        102.0f, 108.0f, 114.0f, 120.0f
    };
    check_grad_ary(t, exp_t, 24, "dt");
    printf("OK\n");
}

/* ── LayerNorm tests ── */

static void test_ln_forward_stats(void) {
    printf("  test_ln_forward_stats... ");
    /* x shape (3, 4), γ=1, β=0, ensure output has mean~0, std~1 along last dim */
    tensor *x = tensor_zeros(2, (int[]){3, 4}, 1);
    float *xp = tensor_data_ptr(x);
    for (int i = 0; i < 12; i++) xp[i] = (float)(i + 1);

    tensor *weight = tensor_zeros(1, (int[]){4}, 1);
    float *wp = tensor_data_ptr(weight);
    for (int j = 0; j < 4; j++) wp[j] = 1.0f;

    tensor *out = tensor_layer_norm(x, weight, NULL, 1e-5f);
    float *od = tensor_data_ptr(out);

    /* each row should have mean≈0, std≈1 */
    for (int s = 0; s < 3; s++) {
        double sum = 0.0, sq = 0.0;
        for (int j = 0; j < 4; j++) {
            sum += od[s * 4 + j];
            sq  += (double)od[s * 4 + j] * od[s * 4 + j];
        }
        float m = (float)(sum / 4.0);
        float v = (float)(sq / 4.0 - (double)(m * m));
        assert(fabsf(m) < EPS && "ln mean ~ 0");
        assert(fabsf(v - 1.0f) < 1e-2f && "ln var ~ 1");
    }
    printf("OK\n");
}

static void test_ln_backward_grads(void) {
    printf("  test_ln_backward_grads... ");
    /* x shape (2, 3), weight and bias learnable, loss = sum(out) */
    tensor *x = tensor_zeros(2, (int[]){2, 3}, 1);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2; xp[2]=3;
    xp[3]=4; xp[4]=5; xp[5]=6;

    tensor *w = tensor_zeros(1, (int[]){3}, 1);
    float *wp = tensor_data_ptr(w);
    wp[0]=1; wp[1]=1; wp[2]=1;

    tensor *b = tensor_zeros(1, (int[]){3}, 1);

    tensor *out = tensor_layer_norm(x, w, b, 1e-5f);
    tensor *loss = tensor_sum(out, 0);  /* scalar */
    dnn_backward(loss);

    /* grads on all params should be non-NULL */
    assert(tensor_grad(x) != NULL);
    assert(tensor_grad(w) != NULL);
    assert(tensor_grad(b) != NULL);

    /* dβ should be 2 (sum of d_out over batch dim = 2 rows) */
    float *bg = tensor_grad(b);
    float exp_b[] = {2.0f, 2.0f, 2.0f};
    check_grad_ary(b, exp_b, 3, "db");
    printf("OK\n");
}

static void test_ln_no_bias(void) {
    printf("  test_ln_no_bias... ");
    tensor *x = tensor_zeros(1, (int[]){4}, 1);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2; xp[2]=3; xp[3]=4;

    tensor *w = tensor_zeros(1, (int[]){4}, 1);
    float *wp = tensor_data_ptr(w);
    wp[0]=1; wp[1]=1; wp[2]=1; wp[3]=1;

    tensor *out = tensor_layer_norm(x, w, NULL, 1e-5f);
    tensor *loss = tensor_sum(out, 0);
    dnn_backward(loss);

    assert(tensor_grad(x) != NULL);
    assert(tensor_grad(w) != NULL);
    printf("OK\n");
}

/* ── Conv2D tests ── */

static void test_conv_tiny(void) {
    printf("  test_conv_tiny... ");
    /* x=(1,1,2,2), w=(1,1,2,2), b=(1,), stride=1, pad=0
       out=[[[[5]]]], dx=[[[[1,0],[0,1]]]], dw=[[[[1,2],[3,4]]]], db=[1] */
    tensor *x = tensor_zeros(4, (int[]){1,1,2,2}, 1);
    float *xp = tensor_data_ptr(x); xp[0]=1; xp[1]=2; xp[2]=3; xp[3]=4;
    tensor *w = tensor_zeros(4, (int[]){1,1,2,2}, 1);
    float *wp = tensor_data_ptr(w); wp[0]=1; wp[1]=0; wp[2]=0; wp[3]=1;
    tensor *b = tensor_zeros(1, (int[]){1}, 1);

    tensor *out = tensor_conv2d(x, w, b, 1, 0);
    tensor *loss = tensor_sum(out, 0);
    dnn_backward(loss);

    float *od = tensor_data_ptr(out);
    if (fabsf(od[0] - 5.0f) > EPS) {
        printf("    FAIL: out: got %.4f, expected %.4f\n", od[0], 5.0f);
        assert(0);
    }

    float exp_dx[] = {1,0,0,1};
    check_grad_ary(x, exp_dx, 4, "dx");

    float exp_dw[] = {1,2,3,4};
    check_grad_ary(w, exp_dw, 4, "dw");

    float exp_db[] = {1.0f};
    check_grad_ary(b, exp_db, 1, "db");
    printf("OK\n");
}

static void test_conv_backward_grads(void) {
    printf("  test_conv_backward_grads... ");
    /* 1x1 kernel, (N=2, C=3, H=4, W=4, out_C=2), stride=1, pad=0 */
    srand(0);
    tensor *x = tensor_randn(4, (int[]){2,3,4,4}, 1);
    tensor *w = tensor_randn(4, (int[]){2,3,1,1}, 1);
    tensor *b = tensor_randn(1, (int[]){2}, 1);

    tensor *out = tensor_conv2d(x, w, b, 1, 0);
    tensor *loss = tensor_sum(out, 0);
    dnn_backward(loss);

    assert(tensor_grad(x) != NULL && "dx non-null");
    assert(tensor_grad(w) != NULL && "dw non-null");
    assert(tensor_grad(b) != NULL && "db non-null");
    printf("OK\n");
}

static void test_conv_no_bias(void) {
    printf("  test_conv_no_bias... ");
    srand(1);
    tensor *x = tensor_randn(4, (int[]){1,2,3,3}, 1);
    tensor *w = tensor_randn(4, (int[]){2,2,1,1}, 1);

    tensor *out = tensor_conv2d(x, w, NULL, 1, 0);
    tensor *loss = tensor_sum(out, 0);
    dnn_backward(loss);

    assert(tensor_grad(x) != NULL && "dx non-null");
    assert(tensor_grad(w) != NULL && "dw non-null");
    printf("OK\n");
}

static void test_conv_stride_pad(void) {
    printf("  test_conv_stride_pad... ");
    /* x=(1,1,4,4), w=(1,1,2,2), stride=2, pad=0 (from ref test 2)
       out = [[[[7,11],[23,27]]]] */
    tensor *x = tensor_zeros(4, (int[]){1,1,4,4}, 1);
    float *xp = tensor_data_ptr(x);
    for (int i = 0; i < 16; i++) xp[i] = (float)(i + 1);
    tensor *w = tensor_zeros(4, (int[]){1,1,2,2}, 1);
    float *wp = tensor_data_ptr(w); wp[0]=1; wp[1]=0; wp[2]=0; wp[3]=1;
    tensor *b = tensor_zeros(1, (int[]){1}, 1);

    tensor *out = tensor_conv2d(x, w, b, 2, 0);
    tensor *loss = tensor_sum(out, 0);
    dnn_backward(loss);

    float *od = tensor_data_ptr(out);
    float exp_out[] = {7.0f, 11.0f, 23.0f, 27.0f};
    for (int i = 0; i < 4; i++)
        if (fabsf(od[i] - exp_out[i]) > EPS) {
            printf("    FAIL: out[%d]: got %.4f, expected %.4f\n", i, od[i], exp_out[i]);
            assert(0);
        }

    float exp_dx[] = {1,0,1,0, 0,1,0,1, 1,0,1,0, 0,1,0,1};
    check_grad_ary(x, exp_dx, 16, "dx");

    float exp_dw[] = {24,28,40,44};
    check_grad_ary(w, exp_dw, 4, "dw");

    float exp_db[] = {4.0f};
    check_grad_ary(b, exp_db, 1, "db");
    printf("OK\n");
}

static void test_ln_exact_pytorch(void) {
    printf("  test_ln_exact_pytorch... ");
    /* exact setup from ref_layer_norm.py backward test */
    tensor *x = tensor_zeros(2, (int[]){2, 3}, 1);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2; xp[2]=3;
    xp[3]=4; xp[4]=5; xp[5]=6;

    tensor *w = tensor_zeros(1, (int[]){3}, 1);
    float *wp = tensor_data_ptr(w);
    wp[0]=1; wp[1]=1; wp[2]=1;

    tensor *b = tensor_zeros(1, (int[]){3}, 1);

    tensor *out = tensor_layer_norm(x, w, b, 1e-5f);
    tensor *loss = tensor_sum(out, 0);
    dnn_backward(loss);

    /* Isolated precision test (test_ln_precision.c) shows 0.00e+00 error.
       Expected values computed from the analytical formula: */
    /* dγ = sum(gd * y) over batch.  gd=1 for all elements (sum backward).
       y = [-1.224739, 0, 1.224739] for each row.
       sum over 2 rows: dγ = [-2.449478, 0, 2.449478] */
    float exp_w[] = {-2.449478f, 0.0f, 2.449478f};
    check_grad_ary(w, exp_w, 3, "dW");

    float exp_b[] = {2.0f, 2.0f, 2.0f};
    check_grad_ary(b, exp_b, 3, "db");
    printf("OK\n");
}

int main(void) {
    printf("test_autograd:\n");

    mem_pool params  = mem_pool_create(512 * 1024);
    mem_pool scratch = mem_pool_create(512 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    test_add_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_add_chain();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_add_multi_use();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_add_diamond();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_add_broadcast();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_no_grad();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_add_self_via_grad_fn();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mul_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mul_chain();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mul_self();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mul_broadcast();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mul_with_add();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_pow_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_pow_cube();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_pow_exp1();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_pow_neg();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_pow_broadcast();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_neg_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_neg_broadcast();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_sub_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_sub_self();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_sub_broadcast();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_div_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_div_self();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_div_broadcast();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_div_neg_b();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_sum_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_sum_2d_dim0();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_sum_2d_dim1();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_sum_chain();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mean_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mean_2d_dim1();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_relu_positive();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_relu_negative();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_relu_mixed();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_relu_chain();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_matmul_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_matmul_linear();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_matmul_square_self();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mixed_diamond();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_transpose_backward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_slice_backward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_mul_diamond_self();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_3d_multi_op();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_softmax_1d();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_softmax_2d_dim0();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_softmax_2d_dim1();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_softmax_chain();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_softmax_stability();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_softmax_sum_to_one();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_cross_entropy_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_cross_entropy_batch();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_cross_entropy_grad();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_cross_entropy_stability();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_cross_entropy_chain();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_ln_forward_stats();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_ln_backward_grads();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_ln_no_bias();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_ln_exact_pytorch();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_conv_tiny();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_conv_backward_grads();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_conv_no_bias();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_conv_stride_pad();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);

    printf("  ALL PASS\n");
    return 0;
}
