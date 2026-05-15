#include "dnn.h"
#include "context.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

static dnn_ctx ctx;

#define EPS 1e-5f

/* helper: create scalar leaf with given value */
static tensor *scalar(float v, int rg) {
    tensor *t = tensor_zeros(ctx.params, 1, (int[]){1}, rg);
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
    tensor *c = tensor_add(ctx.scratch, a, b);
    dnn_backward(ctx.scratch, c);
    check_grad(a, 1.0f, "da");
    check_grad(b, 1.0f, "db");
    printf("OK\n");
}

static void test_add_chain(void) {
    printf("  test_add_chain... ");
    tensor *a = scalar(1.0f, 1);
    tensor *b = scalar(2.0f, 1);
    tensor *c = scalar(3.0f, 1);
    tensor *d = tensor_add(ctx.scratch, a, b);
    tensor *e = tensor_add(ctx.scratch, d, c);
    dnn_backward(ctx.scratch, e);
    check_grad(a, 1.0f, "da");
    check_grad(b, 1.0f, "db");
    check_grad(c, 1.0f, "dc");
    printf("OK\n");
}

static void test_add_multi_use(void) {
    printf("  test_add_multi_use... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_add(ctx.scratch, a, a);     /* a + a = 2a */
    dnn_backward(ctx.scratch, c);
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
    tensor *d = tensor_add(ctx.scratch, a, b);
    tensor *e = tensor_add(ctx.scratch, a, c);
    tensor *loss = tensor_add(ctx.scratch, d, e);
    dnn_backward(ctx.scratch, loss);
    check_grad(a, 2.0f, "da");
    check_grad(b, 1.0f, "db");
    check_grad(c, 1.0f, "dc");
    printf("OK\n");
}

static void test_add_broadcast(void) {
    printf("  test_add_broadcast... ");
    /* a shape [3], b shape [1] → broadcast to [3] */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 10.0f;

    tensor *c = tensor_add(ctx.scratch, a, b);
    dnn_backward(ctx.scratch, c);

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

    dnn_grad_ctx gc = dnn_no_grad_enter();
    tensor *c = tensor_add(ctx.scratch, a, b);
    dnn_no_grad_exit(gc);

    /* inside no_grad: c should NOT have grad_fn */
    assert(c->grad_fn == NULL && "c has grad_fn despite no_grad");
    printf("OK\n");
}

static void test_add_self_via_grad_fn(void) {
    printf("  test_add_self_via_grad_fn... ");
    /* Use c = tensor_add(ctx.scratch, a, a) but interpose a non-
     * trivial op so a == b check in add_backward fires */
    tensor *a = scalar(7.0f, 1);
    tensor *c = tensor_add(ctx.scratch, a, a);
    dnn_backward(ctx.scratch, c);
    check_grad(a, 2.0f, "da");
    printf("OK\n");
}

/* ── Multiplication tests ── */

static void test_mul_simple(void) {
    printf("  test_mul_simple... ");
    /* c = a * b, d = a + c (to get non-unit gout) */
    tensor *a = scalar(3.0f, 1);
    tensor *b = scalar(4.0f, 1);
    tensor *c = tensor_mul(ctx.scratch, a, b);  /* 12 */
    dnn_backward(ctx.scratch, c);
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
    tensor *d = tensor_mul(ctx.scratch, tensor_mul(ctx.scratch, a, b), c);  /* 24 */
    dnn_backward(ctx.scratch, d);
    check_grad(a, 12.0f, "da");   /* d(a*b*c)/da = b*c */
    check_grad(b, 8.0f,  "db");   /* d(a*b*c)/db = a*c */
    check_grad(c, 6.0f,  "dc");   /* d(a*b*c)/dc = a*b */
    printf("OK\n");
}

static void test_mul_self(void) {
    printf("  test_mul_self... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_mul(ctx.scratch, a, a);  /* a*a = 25 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 10.0f, "da");   /* d(a*a)/da = 2*a = 10 */
    printf("OK\n");
}

static void test_mul_broadcast(void) {
    printf("  test_mul_broadcast... ");
    /* a shape [3], b shape [1] → broadcast to [3] */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 2.0f; ap[1] = 3.0f; ap[2] = 4.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 10.0f;

    tensor *c = tensor_mul(ctx.scratch, a, b);
    dnn_backward(ctx.scratch, c);

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
    tensor *ab = tensor_mul(ctx.scratch, a, b);   /* 6 */
    tensor *out = tensor_add(ctx.scratch, ab, c); /* 10 */
    dnn_backward(ctx.scratch, out);
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
    tensor *c = tensor_pow(ctx.scratch, a, 2.0f);  /* 9 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 6.0f, "da");       /* d(a^2)/da = 2*a = 6 */
    printf("OK\n");
}

static void test_pow_cube(void) {
    printf("  test_pow_cube... ");
    /* c = a^3 */
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_pow(ctx.scratch, a, 3.0f);  /* 125 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 75.0f, "da");      /* d(a^3)/da = 3*a^2 = 75 */
    printf("OK\n");
}

static void test_pow_exp1(void) {
    printf("  test_pow_exp1... ");
    /* c = a^1 — derivative is 1 */
    tensor *a = scalar(42.0f, 1);
    tensor *c = tensor_pow(ctx.scratch, a, 1.0f);
    dnn_backward(ctx.scratch, c);
    check_grad(a, 1.0f, "da");
    printf("OK\n");
}

static void test_pow_neg(void) {
    printf("  test_pow_neg... ");
    /* c = a^(-1) = 1/a */
    tensor *a = scalar(4.0f, 1);
    tensor *c = tensor_pow(ctx.scratch, a, -1.0f);  /* 0.25 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, -0.0625f, "da");    /* d(a^-1)/da = -1*a^-2 = -1/16 = -0.0625 */
    printf("OK\n");
}

static void test_pow_broadcast(void) {
    printf("  test_pow_broadcast... ");
    /* a shape [3], c = a^2 */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;

    tensor *c = tensor_pow(ctx.scratch, a, 2.0f);
    dnn_backward(ctx.scratch, c);

    /* da = 2*a = [2, 4, 6] */
    float exp_a[] = {2.0f, 4.0f, 6.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

/* ── Neg tests ── */

static void test_neg_simple(void) {
    printf("  test_neg_simple... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_neg(ctx.scratch, a);  /* -5 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, -1.0f, "da");  /* d(-a)/da = -1 */
    printf("OK\n");
}

static void test_neg_broadcast(void) {
    printf("  test_neg_broadcast... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = -2.0f; ap[2] = 3.0f;

    tensor *c = tensor_neg(ctx.scratch, a);
    dnn_backward(ctx.scratch, c);

    float exp_a[] = {-1.0f, -1.0f, -1.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

/* ── Sub tests ── */

static void test_sub_simple(void) {
    printf("  test_sub_simple... ");
    tensor *a = scalar(7.0f, 1);
    tensor *b = scalar(3.0f, 1);
    tensor *c = tensor_sub(ctx.scratch, a, b);  /* 4 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 1.0f, "da");   /* d(a-b)/da = 1 */
    check_grad(b, -1.0f, "db");  /* d(a-b)/db = -1 */
    printf("OK\n");
}

static void test_sub_self(void) {
    printf("  test_sub_self... ");
    /* a - a = 0, gradient is 0 */
    tensor *a = scalar(10.0f, 1);
    tensor *c = tensor_sub(ctx.scratch, a, a);
    dnn_backward(ctx.scratch, c);
    check_grad(a, 0.0f, "da");
    printf("OK\n");
}

static void test_sub_broadcast(void) {
    printf("  test_sub_broadcast... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 10.0f; ap[1] = 20.0f; ap[2] = 30.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 5.0f;

    tensor *c = tensor_sub(ctx.scratch, a, b);
    dnn_backward(ctx.scratch, c);

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
    tensor *c = tensor_div(ctx.scratch, a, b);  /* 5 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 0.5f, "da");     /* d(a/b)/da = 1/b = 0.5 */
    check_grad(b, -2.5f, "db");   /* d(a/b)/db = -a/b^2 = -10/4 = -2.5 */
    printf("OK\n");
}

static void test_div_self(void) {
    printf("  test_div_self... ");
    /* a/a = 1, gradient is 0 */
    tensor *a = scalar(7.0f, 1);
    tensor *c = tensor_div(ctx.scratch, a, a);
    dnn_backward(ctx.scratch, c);
    check_grad(a, 0.0f, "da");
    printf("OK\n");
}

static void test_div_broadcast(void) {
    printf("  test_div_broadcast... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){1}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 6.0f; ap[1] = 9.0f; ap[2] = 12.0f;
    float *bp = tensor_data_ptr(b);
    bp[0] = 3.0f;

    tensor *c = tensor_div(ctx.scratch, a, b);
    dnn_backward(ctx.scratch, c);

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
    tensor *c = tensor_div(ctx.scratch, a, b);  /* -4 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, -0.5f, "da");    /* 1/b = 1/(-2) = -0.5 */
    check_grad(b, -2.0f, "db");    /* -a/b^2 = -8/4 = -2 */
    printf("OK\n");
}

/* ── Sum tests ── */

static void test_sum_simple(void) {
    printf("  test_sum_simple... ");
    /* a shape [3], sum over dim=0 → scalar (shape [1]) */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;

    tensor *c = tensor_sum(ctx.scratch, a, 0);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 6.0f) < EPS && "sum forward wrong");

    dnn_backward(ctx.scratch, c);

    /* da = gout broadcast = [1, 1, 1] */
    float exp_a[] = {1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

static void test_sum_2d_dim0(void) {
    printf("  test_sum_2d_dim0... ");
    /* a shape [2,3], sum over dim=0 → shape [1,3] */
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 6; i++) ap[i] = (float)(i + 1);  /* [[1,2,3],[4,5,6]] */

    tensor *c = tensor_sum(ctx.scratch, a, 0);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 5.0f) < EPS);  /* 1+4 */
    assert(fabsf(cp[1] - 7.0f) < EPS);  /* 2+5 */
    assert(fabsf(cp[2] - 9.0f) < EPS);  /* 3+6 */

    dnn_backward(ctx.scratch, c);

    /* dnn_backward sets grad of c to all-1s → gout = [1,1,1] broadcast to [2,3] */
    float exp_a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

static void test_sum_2d_dim1(void) {
    printf("  test_sum_2d_dim1... ");
    /* a shape [2,3], sum over dim=1 → shape [2,1] */
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 6; i++) ap[i] = (float)(i + 1);

    tensor *c = tensor_sum(ctx.scratch, a, 1);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 6.0f) < EPS);   /* 1+2+3 */
    assert(fabsf(cp[1] - 15.0f) < EPS);  /* 4+5+6 */

    dnn_backward(ctx.scratch, c);

    /* gout = [1,1] (shape [2,1]), broadcast along dim 1 → all 1s */
    float exp_a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

static void test_sum_chain(void) {
    printf("  test_sum_chain... ");
    /* sum then mul — gradient flows through both */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 2.0f; ap[1] = 3.0f; ap[2] = 4.0f;
    tensor *b = scalar(10.0f, 1);

    tensor *s = tensor_sum(ctx.scratch, a, 0);   /* 9, shape [1] */
    tensor *c = tensor_mul(ctx.scratch, s, b);   /* 90 */
    dnn_backward(ctx.scratch, c);

    /* da = b * ds/da = 10 * [1,1,1] = [10,10,10] */
    float exp_a[] = {10.0f, 10.0f, 10.0f};
    check_grad_ary(a, exp_a, 3, "da");
    check_grad(b, 9.0f, "db");
    printf("OK\n");
}

/* ── Mean tests ── */

static void test_mean_simple(void) {
    printf("  test_mean_simple... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 2.0f; ap[1] = 4.0f; ap[2] = 6.0f;

    tensor *c = tensor_mean(ctx.scratch, a, 0);   /* shape [1], value 4 */
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 4.0f) < EPS && "mean forward wrong");

    dnn_backward(ctx.scratch, c);

    /* da = gout * 1/n = 1/3 for each element */
    float exp_a[] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
    check_grad_ary(a, exp_a, 3, "da");
    printf("OK\n");
}

static void test_mean_2d_dim1(void) {
    printf("  test_mean_2d_dim1... ");
    /* a shape [2,3], mean over dim=1 → shape [2,1] */
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=6;
    ap[3]=2; ap[4]=4; ap[5]=6;
    /* row 0 mean = 3, row 1 mean = 4 */

    tensor *c = tensor_mean(ctx.scratch, a, 1);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 3.0f) < EPS);
    assert(fabsf(cp[1] - 4.0f) < EPS);

    dnn_backward(ctx.scratch, c);

    /* da = gout * 1/3 broadcast along dim 1 */
    float exp_a[] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

/* ── ReLU tests ── */

static void test_relu_positive(void) {
    printf("  test_relu_positive... ");
    tensor *a = scalar(5.0f, 1);
    tensor *c = tensor_relu(ctx.scratch, a);  /* 5 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 1.0f, "da");  /* drelu(5)/da = 1 */
    printf("OK\n");
}

static void test_relu_negative(void) {
    printf("  test_relu_negative... ");
    tensor *a = scalar(-3.0f, 1);
    tensor *c = tensor_relu(ctx.scratch, a);  /* 0 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 0.0f, "da");  /* drelu(-3)/da = 0 */
    printf("OK\n");
}

static void test_relu_mixed(void) {
    printf("  test_relu_mixed... ");
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = -1.0f; ap[1] = 0.0f; ap[2] = 2.0f; ap[3] = -5.0f;

    tensor *c = tensor_relu(ctx.scratch, a);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 0.0f) < EPS);
    assert(fabsf(cp[1] - 0.0f) < EPS);
    assert(fabsf(cp[2] - 2.0f) < EPS);
    assert(fabsf(cp[3] - 0.0f) < EPS);

    dnn_backward(ctx.scratch, c);

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
    tensor *ra = tensor_relu(ctx.scratch, a);  /* 0 */
    tensor *c = tensor_add(ctx.scratch, ra, b);  /* 5 */
    dnn_backward(ctx.scratch, c);
    check_grad(a, 0.0f, "da");  /* relu(-2)=0, derivative 0 */
    check_grad(b, 1.0f, "db");
    printf("OK\n");
}

/* ── Matmul tests ── */

/* ── Sigmoid tests ── */

static void test_sigmoid_simple(void) {
    printf("  test_sigmoid_simple... ");
    tensor *a = scalar(0.0f, 1);
    tensor *c = tensor_sigmoid(ctx.scratch, a);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 0.5f) < EPS && "sigmoid(0) = 0.5");
    dnn_backward(ctx.scratch, c);
    /* dσ/dx at 0 = 0.5 * 0.5 = 0.25 */
    check_grad(a, 0.25f, "da");
    printf("OK\n");
}

static void test_sigmoid_positive(void) {
    printf("  test_sigmoid_positive... ");
    tensor *a = scalar(2.0f, 1);
    tensor *c = tensor_sigmoid(ctx.scratch, a);
    float *cp = tensor_data_ptr(c);
    float sig_ref = 1.0f / (1.0f + expf(-2.0f));
    assert(fabsf(cp[0] - sig_ref) < EPS && "sigmoid(2)");
    dnn_backward(ctx.scratch, c);
    float expected_grad = sig_ref * (1.0f - sig_ref);
    check_grad(a, expected_grad, "da");
    printf("OK\n");
}

static void test_sigmoid_negative(void) {
    printf("  test_sigmoid_negative... ");
    tensor *a = scalar(-2.0f, 1);
    tensor *c = tensor_sigmoid(ctx.scratch, a);
    float *cp = tensor_data_ptr(c);
    float sig_ref = 1.0f / (1.0f + expf(2.0f));
    assert(fabsf(cp[0] - sig_ref) < EPS && "sigmoid(-2)");
    dnn_backward(ctx.scratch, c);
    float expected_grad = sig_ref * (1.0f - sig_ref);
    check_grad(a, expected_grad, "da");
    printf("OK\n");
}

static void test_sigmoid_array(void) {
    printf("  test_sigmoid_array... ");
    float vals[] = {-3.0f, -1.0f, 0.0f, 2.0f, 4.0f};
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){5}, 1);
    float *ad = tensor_data_ptr(a);
    for (int i = 0; i < 5; i++) ad[i] = vals[i];

    tensor *c = tensor_sigmoid(ctx.scratch, a);
    float *cd = tensor_data_ptr(c);

    /* check forward */
    for (int i = 0; i < 5; i++) {
        float expected = 1.0f / (1.0f + expf(-vals[i]));
        assert(fabsf(cd[i] - expected) < EPS && "sigmoid array forward");
    }

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    assert(ag && "sigmoid array grad not NULL");

    /* check backward: dσ/dx = σ*(1-σ) for unit gout */
    for (int i = 0; i < 5; i++) {
        float sig = 1.0f / (1.0f + expf(-vals[i]));
        float expected_grad = sig * (1.0f - sig);
        assert(fabsf(ag[i] - expected_grad) < EPS && "sigmoid array backward");
    }
    printf("OK\n");
}

static void test_sigmoid_chain(void) {
    printf("  test_sigmoid_chain... ");
    /* c = sigmoid(a) + a
     * dc/da = σ(a)*(1-σ(a)) + 1 */
    tensor *a = scalar(1.0f, 1);
    tensor *s = tensor_sigmoid(ctx.scratch, a);
    tensor *c = tensor_add(ctx.scratch, s, a);
    dnn_backward(ctx.scratch, c);
    float sig = 1.0f / (1.0f + expf(-1.0f));
    float expected_grad = sig * (1.0f - sig) + 1.0f;
    check_grad(a, expected_grad, "da");
    printf("OK\n");
}

static void test_sigmoid_no_grad(void) {
    printf("  test_sigmoid_no_grad... ");
    tensor *a = scalar(3.0f, 0);
    tensor *c = tensor_sigmoid(ctx.scratch, a);
    float *cp = tensor_data_ptr(c);
    float sig_ref = 1.0f / (1.0f + expf(-3.0f));
    assert(fabsf(cp[0] - sig_ref) < EPS && "sigmoid no-grad forward");
    assert(tensor_grad(a) == NULL && "sigmoid no-grad: no grad allocated");
    assert(tensor_grad(c) == NULL && "sigmoid no-grad: output has no grad");
    printf("OK\n");
}

static void test_silu_scalar(void) {
    printf("  test_silu_scalar... ");
    /* 9 scalar values in one 1D tensor — forward + backward, check all */
    float xs[]  = { 0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 0.5f, -0.5f };
    float ef[]  = { 0.00000000f, 0.73105860f, -0.26894143f, 1.76159406f, -0.23840584f,
                    2.85772252f, -0.14227761f, 0.31122968f, -0.18877034f };
    float eg[]  = { 0.50000000f, 0.92767054f, 0.07232948f, 1.09078431f, -0.09078425f,
                    1.08810413f, -0.08810411f, 0.73996121f, 0.26003882f };
    int n = 9;

    tensor *a = tensor_zeros(ctx.params, 1, (int[]){n}, 1);
    float *ad = tensor_data_ptr(a);
    for (int i = 0; i < n; i++) ad[i] = xs[i];

    tensor *c = tensor_silu(ctx.scratch, a);
    float *cd = tensor_data_ptr(c);
    for (int i = 0; i < n; i++) {
        if (fabsf(cd[i] - ef[i]) > EPS) {
            printf("\n    FAIL: silu(%g) forward[%d]: got %.8f, expected %.8f\n",
                   xs[i], i, cd[i], ef[i]); assert(0);
        }
    }

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    assert(ag && "silu scalar grad not NULL");
    for (int i = 0; i < n; i++) {
        if (fabsf(ag[i] - eg[i]) > EPS) {
            printf("\n    FAIL: silu(%g) grad[%d]: got %.8f, expected %.8f\n",
                   xs[i], i, ag[i], eg[i]); assert(0);
        }
    }
    printf("OK\n");
}

static void test_silu_array(void) {
    printf("  test_silu_array... ");
    float vals[]  = {-3.0f, -1.0f, 0.0f, 2.0f, 4.0f};
    float exp_fwd[] = {-0.14227761f, -0.26894143f, 0.00000000f, 1.76159406f, 3.92805505f};
    float exp_grad[] = {-0.08810411f, 0.07232948f, 0.50000000f, 1.09078431f, 1.05266464f};
    int n = 5;

    tensor *a = tensor_zeros(ctx.params, 1, (int[]){n}, 1);
    float *ad = tensor_data_ptr(a);
    for (int i = 0; i < n; i++) ad[i] = vals[i];

    tensor *c = tensor_silu(ctx.scratch, a);
    float *cd = tensor_data_ptr(c);

    for (int i = 0; i < n; i++) {
        if (fabsf(cd[i] - exp_fwd[i]) > EPS) {
            printf("\n    FAIL: silu array forward[%d]: got %.8f, expected %.8f\n",
                   i, cd[i], exp_fwd[i]); assert(0);
        }
    }

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    assert(ag && "silu array grad not NULL");

    for (int i = 0; i < n; i++) {
        if (fabsf(ag[i] - exp_grad[i]) > EPS) {
            printf("\n    FAIL: silu array grad[%d]: got %.8f, expected %.8f\n",
                   i, ag[i], exp_grad[i]); assert(0);
        }
    }
    printf("OK\n");
}

static void test_silu_chain(void) {
    printf("  test_silu_chain... ");
    /* c = silu(a) + a
     * dc/da = silu'(a) + 1 */
    tensor *a = scalar(1.0f, 1);
    tensor *s = tensor_silu(ctx.scratch, a);
    tensor *c = tensor_add(ctx.scratch, s, a);
    dnn_backward(ctx.scratch, c);
    float silu_grad = 0.92767054f;  /* silu'(1) from ref */
    check_grad(a, silu_grad + 1.0f, "da");
    printf("OK\n");
}

static void test_silu_no_grad(void) {
    printf("  test_silu_no_grad... ");
    tensor *a = scalar(3.0f, 0);
    tensor *c = tensor_silu(ctx.scratch, a);
    float *cp = tensor_data_ptr(c);
    float ref = 3.0f / (1.0f + expf(-3.0f));  /* silu(3) */
    assert(fabsf(cp[0] - ref) < EPS && "silu no-grad forward");
    assert(tensor_grad(a) == NULL && "silu no-grad: no grad allocated");
    assert(tensor_grad(c) == NULL && "silu no-grad: output has no grad");
    printf("OK\n");
}

/* ── SwiGLU tests ── */

static void test_swiglu_simple(void) {
    printf("  test_swiglu_simple... ");
    /* out = SiLU(gate) * up, both scalars */
    tensor *gate = scalar(2.0f, 1);
    tensor *up   = scalar(3.0f, 1);
    tensor *out  = tensor_swiglu(ctx.scratch, gate, up);
    /* SiLU(2) = 2/(1+exp(-2)) ≈ 2/1.135335 = 1.761594, * 3 ≈ 5.284782 */
    float ref_silu = 2.0f / (1.0f + expf(-2.0f));
    float ref_out = ref_silu * 3.0f;
    float *op = tensor_data_ptr(out);
    assert(fabsf(op[0] - ref_out) < EPS && "swiglu scalar forward");

    dnn_backward(ctx.scratch, out);
    /* SiLU'(2) = sig*(1 + x - x*sig) where sig=sigmoid(2) */
    float sig = 1.0f / (1.0f + expf(-2.0f));
    float silu_deriv = sig * (1.0f + 2.0f - 2.0f * sig);
    float exp_dgate = 1.0f * silu_deriv * 3.0f;  /* d_gate = d_out * SiLU'(gate) * up */
    float exp_dup   = 1.0f * ref_silu;            /* d_up = d_out * SiLU(gate) */
    check_grad(gate, exp_dgate, "d_gate");
    check_grad(up, exp_dup, "d_up");
    printf("OK\n");
}

static void test_swiglu_array(void) {
    printf("  test_swiglu_array... ");
    /* both [3], no broadcast */
    tensor *gate = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *up   = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *gp = tensor_data_ptr(gate);
    float *up_p = tensor_data_ptr(up);
    gp[0] = 0.0f; gp[1] = 1.0f; gp[2] = -2.0f;
    up_p[0] = 2.0f; up_p[1] = 3.0f; up_p[2] = 4.0f;

    tensor *out = tensor_swiglu(ctx.scratch, gate, up);
    float *od = tensor_data_ptr(out);

    /* reference values computed via python */
    float expected[] = {
        0.0f,
        3.0f / (1.0f + expf(-1.0f)),       /* SiLU(1)*3 */
        (-2.0f / (1.0f + expf(2.0f))) * 4.0f  /* SiLU(-2)*4 */
    };
    expected[1] = (1.0f / (1.0f + expf(-1.0f))) * 3.0f;
    expected[2] = (-2.0f / (1.0f + expf(2.0f))) * 4.0f;
    for (int i = 0; i < 3; i++)
        assert(fabsf(od[i] - expected[i]) < EPS && "swiglu array forward");

    dnn_backward(ctx.scratch, out);

    float *ag = tensor_grad(gate);
    float *bg = tensor_grad(up);
    assert(ag && bg && "swiglu array grads not NULL");

    for (int i = 0; i < 3; i++) {
        float g = gp[i];
        float s = 1.0f / (1.0f + expf(-g));
        float silu = g * s;
        float silu_deriv = s * (1.0f + g - g * s);
        float exp_dg = 1.0f * silu_deriv * up_p[i];
        float exp_du = 1.0f * silu;
        assert(fabsf(ag[i] - exp_dg) < EPS && "swiglu array d_gate");
        assert(fabsf(bg[i] - exp_du) < EPS && "swiglu array d_up");
    }
    printf("OK\n");
}

static void test_swiglu_broadcast_gate(void) {
    printf("  test_swiglu_broadcast_gate... ");
    /* gate [1], up [3] → broadcast to [3] */
    tensor *gate = tensor_zeros(ctx.params, 1, (int[]){1}, 1);
    tensor *up   = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *gp = tensor_data_ptr(gate);
    float *up_p = tensor_data_ptr(up);
    gp[0] = 2.0f;
    up_p[0] = 1.0f; up_p[1] = 2.0f; up_p[2] = 3.0f;

    tensor *out = tensor_swiglu(ctx.scratch, gate, up);
    dnn_backward(ctx.scratch, out);

    float sig = 1.0f / (1.0f + expf(-2.0f));
    float silu = 2.0f * sig;
    float silu_deriv = sig * (1.0f + 2.0f - 2.0f * sig);
    /* d_gate = sum over up dim: go_i * silu'(gate) * up_i */
    float exp_dgate = silu_deriv * (1.0f + 2.0f + 3.0f);
    check_grad(gate, exp_dgate, "d_gate broadcast");
    /* d_up = go * silu(gate) = [silu, silu, silu] */
    float exp_dup[] = {silu, silu, silu};
    check_grad_ary(up, exp_dup, 3, "d_up broadcast");
    printf("OK\n");
}

static void test_swiglu_broadcast_up(void) {
    printf("  test_swiglu_broadcast_up... ");
    /* gate [3], up [1] → broadcast to [3] */
    tensor *gate = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    tensor *up   = tensor_zeros(ctx.params, 1, (int[]){1}, 1);
    float *gp = tensor_data_ptr(gate);
    float *up_p = tensor_data_ptr(up);
    gp[0] = 0.0f; gp[1] = 1.0f; gp[2] = -1.0f;
    up_p[0] = 5.0f;

    tensor *out = tensor_swiglu(ctx.scratch, gate, up);
    dnn_backward(ctx.scratch, out);

    float *ag = tensor_grad(gate);
    float *bg = tensor_grad(up);
    assert(ag && bg && "swiglu broadcast grads");

    for (int i = 0; i < 3; i++) {
        float g = gp[i];
        float s = 1.0f / (1.0f + expf(-g));
        float silu_deriv = s * (1.0f + g - g * s);
        float exp_dg = 1.0f * silu_deriv * up_p[0];
        assert(fabsf(ag[i] - exp_dg) < EPS && "swiglu broadcast d_gate");
    }
    /* d_up = sum_i go_i * SiLU(gate_i) */
    float sum_silu = 0.0f;
    for (int i = 0; i < 3; i++) {
        float g = gp[i];
        float s = 1.0f / (1.0f + expf(-g));
        sum_silu += g * s;
    }
    check_grad(up, sum_silu, "d_up broadcast");
    printf("OK\n");
}

static void test_swiglu_no_grad(void) {
    printf("  test_swiglu_no_grad... ");
    tensor *gate = scalar(2.0f, 0);
    tensor *up   = scalar(3.0f, 0);
    dnn_grad_ctx gc = dnn_no_grad_enter();
    tensor *out = tensor_swiglu(ctx.scratch, gate, up);
    dnn_no_grad_exit(gc);
    float ref = (2.0f / (1.0f + expf(-2.0f))) * 3.0f;
    float *op = tensor_data_ptr(out);
    assert(fabsf(op[0] - ref) < EPS && "swiglu no-grad forward");
    assert(out->grad_fn == NULL && "swiglu no-grad: no grad_fn");
    printf("OK\n");
}

static void test_swiglu_chain(void) {
    printf("  test_swiglu_chain... ");
    /* out = swiglu(gate, up) + gate, verify gradient flows */
    tensor *gate = scalar(1.5f, 1);
    tensor *up   = scalar(2.0f, 1);
    tensor *s = tensor_swiglu(ctx.scratch, gate, up);
    tensor *out = tensor_add(ctx.scratch, s, gate);
    dnn_backward(ctx.scratch, out);
    /* d_out/d_gate = SiLU'(gate)*up + 1 */
    float sig = 1.0f / (1.0f + expf(-1.5f));
    float silu_deriv = sig * (1.0f + 1.5f - 1.5f * sig);
    float exp_dgate = 1.0f * silu_deriv * 2.0f + 1.0f;
    /* d_out/d_up = SiLU(gate) */
    float silu = 1.5f * sig;
    float exp_dup = 1.0f * silu;
    check_grad(gate, exp_dgate, "d_gate chain");
    check_grad(up, exp_dup, "d_up chain");
    printf("OK\n");
}

static void test_matmul_simple(void) {
    printf("  test_matmul_simple... ");
    /* A (2,3) @ B (3,2) = C (2,2) */
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    tensor *b = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 1);
    float *ap = tensor_data_ptr(a);
    float *bp = tensor_data_ptr(b);
    ap[0]=1; ap[1]=2; ap[2]=3; ap[3]=4; ap[4]=5; ap[5]=6;
    bp[0]=7; bp[1]=8; bp[2]=9; bp[3]=10; bp[4]=11; bp[5]=12;

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    float *cp = tensor_data_ptr(c);
    assert(fabsf(cp[0] - 58.0f) < EPS);
    assert(fabsf(cp[1] - 64.0f) < EPS);
    assert(fabsf(cp[3] - 154.0f) < EPS);  /* cp[3] = (1,1) = 154 */

    dnn_backward(ctx.scratch, c);

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
    tensor *x = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    tensor *W = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 1);
    tensor *b = scalar(1.0f, 1);
    float *xp = tensor_data_ptr(x);
    float *Wp = tensor_data_ptr(W);
    xp[0]=1; xp[1]=2; xp[2]=3; xp[3]=4; xp[4]=5; xp[5]=6;
    Wp[0]=1; Wp[1]=0; Wp[2]=0; Wp[3]=1; Wp[4]=1; Wp[5]=0;

    tensor *mm = tensor_matmul(ctx.scratch, x, W);  /* (2,2) */
    tensor *y = tensor_add(ctx.scratch, mm, b);     /* broadcast b to (2,2) */
    dnn_backward(ctx.scratch, y);

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
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 2}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=3; ap[3]=4;

    tensor *c = tensor_matmul(ctx.scratch, a, a);
    float *cp = tensor_data_ptr(c);
    /* A@A = [[7,10],[15,22]] */
    assert(fabsf(cp[0] - 7.0f) < EPS);
    assert(fabsf(cp[1] - 10.0f) < EPS);
    assert(fabsf(cp[2] - 15.0f) < EPS);
    assert(fabsf(cp[3] - 22.0f) < EPS);

    dnn_backward(ctx.scratch, c);

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
    tensor *c = tensor_mul(ctx.scratch, a, b);   /* 6 */
    tensor *d = tensor_add(ctx.scratch, a, b);   /* 5, but unused in grad check — just for diamond */
    tensor *e = tensor_add(ctx.scratch, c, d);   /* 11 */
    dnn_backward(ctx.scratch, e);
    check_grad(a, 4.0f, "da");     /* da = d(c)/da + d(d)/da = b + 1 = 4 */
    check_grad(b, 3.0f, "db");     /* db = d(c)/db + d(d)/db = a + 1 = 3 */
    printf("OK\n");
}

static void test_transpose_backward(void) {
    printf("  test_transpose_backward... ");
    /* a (2,3) → transpose(a,0,1) → sum(b,0) — b view has swapped strides */
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 6; i++) ap[i] = (float)(i + 1);

    tensor *b = tensor_transpose(ctx.scratch, a, 0, 1);  /* shape (3,2), strides (1,3) */
    tensor *c = tensor_sum(ctx.scratch, b, 0);           /* shape (1,2) */
    dnn_backward(ctx.scratch, c);

    /* grad_output = [1,1], broadcast via sum_backward:
       each input element used once → grad all 1s */
    float exp_a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check_grad_ary(a, exp_a, 6, "da");
    printf("OK\n");
}

static void test_slice_backward(void) {
    printf("  test_slice_backward... ");
    /* a (4) → slice at offset 1, len 2 → sum(b, 0) → grad only flows to sliced region */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=10; ap[1]=20; ap[2]=30; ap[3]=40;

    tensor *b = tensor_slice(ctx.scratch, a, 0, 1, 2);  /* shape (2), offset 1, values [20,30] */
    tensor *c = tensor_sum(ctx.scratch, b, 0);           /* shape (1), value 50 */
    dnn_backward(ctx.scratch, c);

    /* only sliced positions get gradient = 1 */
    float exp_a[] = {0.0f, 1.0f, 1.0f, 0.0f};
    check_grad_ary(a, exp_a, 4, "da");
    printf("OK\n");
}

static void test_mul_diamond_self(void) {
    printf("  test_mul_diamond_self... ");
    /* a used in mul(a,a) AND add(a,...) — tests multi-path with a==b case in mul */
    tensor *a = scalar(3.0f, 1);
    tensor *c = tensor_mul(ctx.scratch, a, a);       /* 9, grad contribution: d(a*a)/da = 6 */
    tensor *d = tensor_add(ctx.scratch, a, c);       /* 12, grad contribution: d(a)/da = 1 */
    dnn_backward(ctx.scratch, d);
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

    tensor *a = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 4; i++) ap[i] = x[i];

    tensor *c = tensor_softmax(ctx.scratch, a, 0);
    float *cp = tensor_data_ptr(c);

    for (int i = 0; i < 4; i++)
        assert(fabsf(cp[i] - y_ref[i]) < EPS && "softmax 1d forward");

    /* sum of output should be 1 */
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += cp[i];
    assert(fabsf(sum - 1.0f) < EPS && "softmax 1d sum=1");

    dnn_backward(ctx.scratch, c);

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
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=3;
    ap[3]=4; ap[4]=5; ap[5]=6;

    tensor *c = tensor_softmax(ctx.scratch, a, 0);
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

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    /* with unit gout, dx = sm_i * (1 - sum_j(sm_j)) = 0 for each column */
    for (int i = 0; i < 6; i++)
        assert(fabsf(ag[i]) < EPS && "softmax 2d dim0 grad");

    printf("OK\n");
}

static void test_softmax_2d_dim1(void) {
    printf("  test_softmax_2d_dim1... ");
    /* a shape [2,3], softmax over dim=1 */
    tensor *a = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0]=1; ap[1]=2; ap[2]=3;
    ap[3]=4; ap[4]=5; ap[5]=6;

    tensor *c = tensor_softmax(ctx.scratch, a, 1);
    float *cp = tensor_data_ptr(c);

    /* each row (dim 1) should sum to 1 */
    float row0 = cp[0] + cp[1] + cp[2];
    float row1 = cp[3] + cp[4] + cp[5];
    assert(fabsf(row0 - 1.0f) < EPS && "softmax 2d dim1 row0");
    assert(fabsf(row1 - 1.0f) < EPS && "softmax 2d dim1 row1");

    dnn_backward(ctx.scratch, c);
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
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;
    tensor *b = scalar(3.0f, 1);

    tensor *s = tensor_softmax(ctx.scratch, a, 0);
    tensor *c = tensor_mul(ctx.scratch, s, b);
    dnn_backward(ctx.scratch, c);

    float *ag = tensor_grad(a);
    for (int i = 0; i < 3; i++)
        assert(fabsf(ag[i]) < EPS && "softmax chain grad");
    check_grad(b, 1.0f, "db");  /* dc/db = sum(s) = 1 */
    printf("OK\n");
}

static void test_softmax_stability(void) {
    printf("  test_softmax_stability... ");
    /* large values should not overflow */
    tensor *a = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *ap = tensor_data_ptr(a);
    ap[0] = 100.0f; ap[1] = 101.0f; ap[2] = 102.0f; ap[3] = 103.0f;

    tensor *c = tensor_softmax(ctx.scratch, a, 0);
    float *cp = tensor_data_ptr(c);

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        assert(isfinite(cp[i]) && "softmax stability: non-finite");
        sum += cp[i];
    }
    assert(fabsf(sum - 1.0f) < EPS && "softmax stability sum=1");

    /* also very negative values */
    ap[0] = -1000.0f; ap[1] = -1001.0f; ap[2] = -1002.0f; ap[3] = -1003.0f;
    tensor *d = tensor_softmax(ctx.scratch, a, 0);
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
    tensor *a = tensor_zeros(ctx.params, 3, (int[]){2, 3, 4}, 0);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 24; i++) ap[i] = (float)(i - 10);

    tensor *c = tensor_softmax(ctx.scratch, a, 1);
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

/* ── Causal softmax tests ── */

/* Reference: 2D causal softmax (N, N). For each row i, softmax over cols 0..i */
static void causal_softmax_ref_2d(const float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        int vis = i + 1;  /* visible columns: 0..i */
        /* max over visible */
        float mx = x[i * N];
        for (int j = 1; j < vis; j++) {
            float v = x[i * N + j];
            if (v > mx) mx = v;
        }
        /* sum of exp over visible */
        float se = 0.0f;
        for (int j = 0; j < vis; j++)
            se += expf(x[i * N + j] - mx);
        /* write output */
        float inv_se = 1.0f / se;
        for (int j = 0; j < vis; j++)
            y[i * N + j] = expf(x[i * N + j] - mx) * inv_se;
        for (int j = vis; j < N; j++)
            y[i * N + j] = 0.0f;
    }
}

/* Reference: grad of causal softmax w.r.t scores for 2D with given upstream grad */
static void causal_softmax_grad_ref_2d(const float *x, const float *upstream,
                                         float *grad_out, int N) {
    /* compute forward first */
    float *sm = malloc(N * N * sizeof(float));
    assert(sm);
    causal_softmax_ref_2d(x, sm, N);

    for (int i = 0; i < N; i++) {
        float dot = 0.0f;
        for (int j = 0; j <= i; j++)
            dot += sm[i * N + j] * upstream[i * N + j];
        for (int j = 0; j <= i; j++)
            grad_out[i * N + j] = sm[i * N + j] * (upstream[i * N + j] - dot);
        for (int j = i + 1; j < N; j++)
            grad_out[i * N + j] = 0.0f;
    }
    free(sm);
}

static void test_causal_softmax_2d_forward(void) {
    printf("  test_causal_softmax_2d_forward... ");
    int N = 4;
    float scores[] = {-0.1440903296f, -0.1729036003f, -0.1113158616f, 0.7019837251f,
                      -0.1275882838f, -1.4973534143f,  0.3323183441f, -0.2673374785f,
                      -0.2169586841f,  0.1158847867f,  0.2322977369f,  1.1635586866f,
                       0.6566365068f,  0.1105071774f, -0.7383216023f, -1.0146623675f};
    float expected[] = {1.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
                        0.7973422042f, 0.2026577958f, 0.0000000000f, 0.0000000000f,
                        0.2523929763f, 0.3520702655f, 0.3955367581f, 0.0000000000f,
                        0.4962696763f, 0.2874331041f, 0.1229971731f, 0.0933000466f};

    tensor *t = tensor_zeros(ctx.params, 2, (int[]){N, N}, 0);
    float *tp = tensor_data_ptr(t);
    for (int i = 0; i < N * N; i++) tp[i] = scores[i];

    tensor *out = tensor_causal_softmax(ctx.scratch, t);
    float *op = tensor_data_ptr(out);

    for (int i = 0; i < N * N; i++) {
        if (fabsf(op[i] - expected[i]) > 1e-5f) {
            printf("FAIL: out[%d] = %.6f, expected %.6f\n", i, op[i], expected[i]);
            assert(0);
        }
    }

    /* also verify row sums */
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j <= i; j++) sum += op[i * N + j];
        assert(fabsf(sum - 1.0f) < 1e-5f && "causal softmax row sum != 1");
    }
    printf("OK\n");
}

static void test_causal_softmax_4d_forward(void) {
    printf("  test_causal_softmax_4d_forward... ");
    /* Test (B=2, H=3, N=4) with the same data repeated for simplicity */
    int B = 2, H = 3, N = 4;
    float scores[] = {-0.1440903296f, -0.1729036003f, -0.1113158616f, 0.7019837251f,
                      -0.1275882838f, -1.4973534143f,  0.3323183441f, -0.2673374785f,
                      -0.2169586841f,  0.1158847867f,  0.2322977369f,  1.1635586866f,
                       0.6566365068f,  0.1105071774f, -0.7383216023f, -1.0146623675f};
    float expected[] = {1.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
                        0.7973422042f, 0.2026577958f, 0.0000000000f, 0.0000000000f,
                        0.2523929763f, 0.3520702655f, 0.3955367581f, 0.0000000000f,
                        0.4962696763f, 0.2874331041f, 0.1229971731f, 0.0933000466f};

    tensor *t = tensor_zeros(ctx.params, 4, (int[]){B, H, N, N}, 0);
    float *tp = tensor_data_ptr(t);
    for (int i = 0; i < B * H * N * N; i++)
        tp[i] = scores[i % (N * N)];

    tensor *out = tensor_causal_softmax(ctx.scratch, t);
    float *op = tensor_data_ptr(out);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N * N; i++) {
                int idx = (b * H + h) * N * N + i;
                if (fabsf(op[idx] - expected[i]) > 1e-5f) {
                    printf("FAIL: [%d,%d,%d] = %.6f, expected %.6f\n",
                           b, h, i, op[idx], expected[i]);
                    assert(0);
                }
            }
        }
    }

    printf("OK\n");
}

static void test_causal_softmax_2d_backward(void) {
    printf("  test_causal_softmax_2d_backward... ");
    int N = 4;
    float scores[] = {-0.1440903296f, -0.1729036003f, -0.1113158616f, 0.7019837251f,
                      -0.1275882838f, -1.4973534143f,  0.3323183441f, -0.2673374785f,
                      -0.2169586841f,  0.1158847867f,  0.2322977369f,  1.1635586866f,
                       0.6566365068f,  0.1105071774f, -0.7383216023f, -1.0146623675f};
    float grad_in[] = {1.0f, 2.0f, 3.0f, 4.0f,
                       1.0f, 2.0f, 3.0f, 4.0f,
                       1.0f, 2.0f, 3.0f, 4.0f,
                       1.0f, 2.0f, 3.0f, 4.0f};
    float expected_grad[16];
    causal_softmax_grad_ref_2d(scores, grad_in, expected_grad, N);

    tensor *t = tensor_zeros(ctx.params, 2, (int[]){N, N}, 1);
    float *tp = tensor_data_ptr(t);
    for (int i = 0; i < N * N; i++) tp[i] = scores[i];

    /* Forward + backward */
    tensor *out = tensor_causal_softmax(ctx.scratch, t);

    /* Attach a gradient by summing with weights: loss = sum(out[i][j] * grad_in[i][j]) */
    /* We can do this by making a weighted sum using tensor_mul + tensor_sum */
    /* Simpler: just call dnn_backward on out, but we need non-unit grad */
    /* Create a weight tensor of grad_in values, multiply out by weights, sum */
    tensor *weights = tensor_zeros(ctx.params, 2, (int[]){N, N}, 0);
    float *wp = tensor_data_ptr(weights);
    for (int i = 0; i < N * N; i++) wp[i] = grad_in[i];

    dnn_grad_ctx gc = dnn_no_grad_enter();
    tensor *loss = tensor_sum(ctx.scratch, tensor_mul(ctx.scratch, out, weights), -1);  /* scalar */
    dnn_no_grad_exit(gc);

    /* Use mul+sum to inject non-unit upstream gradient into causal_softmax backward.
     * loss = sum(out * weights) where out = causal_softmax(t).
     * d(loss)/d(out) = weights, so causal_softmax_backward receives weights as grad_output.
     *
     * out has requires_grad=1 (from causal_softmax autograd tape).
     * tensor_mul creates grad_fn because out requires_grad.
     * tensor_sum creates grad_fn because mul output requires_grad.
     * dnn_backward(ctx.scratch, loss_scalar) propagates: loss_scalar → sum1 → sum0 → mul → causal_softmax → t
     */
    tensor *weighted = tensor_mul(ctx.scratch, out, weights);
    tensor *sum0 = tensor_sum(ctx.scratch, weighted, 1);   /* (N,N) → (N,) */
    tensor *loss_s = tensor_sum(ctx.scratch, sum0, 0);     /* (N,) → (1,) */

    dnn_backward(ctx.scratch, loss_s);

    float *ag = tensor_grad(t);
    assert(ag && "causal softmax grad not null");
    for (int i = 0; i < N * N; i++) {
        if (fabsf(ag[i] - expected_grad[i]) > 1e-5f) {
            printf("FAIL: grad[%d] = %.6f, expected %.6f\n", i, ag[i], expected_grad[i]);
            assert(0);
        }
    }
    printf("OK\n");
}

static void test_cross_entropy_simple(void) {
    /* 2 classes, logits [1.0, 2.0], target=1 */
    tensor *logits = tensor_zeros(ctx.params, 1, (int[]){2}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0] = 1.0f; lp[1] = 2.0f;

    tensor *target = tensor_zeros(ctx.params, 1, (int[]){1}, 0);
    ((int*)tensor_data_ptr(target))[0] = 1;

    tensor *loss = tensor_cross_entropy(ctx.scratch, logits, target, 0);
    float *loss_p = tensor_data_ptr(loss);

    float expected = ce_ref_2d(lp, 1, 2);
    assert(fabsf(loss_p[0] - expected) < EPS && "ce simple forward");

    dnn_backward(ctx.scratch, loss);
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
    tensor *logits = tensor_zeros(ctx.params, 2, (int[]){3, 4}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0]=2.0f;  lp[1]=1.0f;  lp[2]=0.5f; lp[3]=0.1f;   /* sample 0 */
    lp[4]=0.5f;  lp[5]=2.0f;  lp[6]=3.0f; lp[7]=1.0f;   /* sample 1 */
    lp[8]=1.0f;  lp[9]=3.0f;  lp[10]=2.0f;lp[11]=0.5f;  /* sample 2 */

    tensor *target = tensor_zeros(ctx.params, 1, (int[]){3}, 0);
    int *td = (int*)tensor_data_ptr(target);
    td[0] = 0; td[1] = 2; td[2] = 1;

    tensor *loss = tensor_cross_entropy(ctx.scratch, logits, target, 1);
    float *loss_p = tensor_data_ptr(loss);

    /* compute expected: mean of per-sample losses */
    float l0 = ce_ref_2d(lp + 0, 0, 4);
    float l1 = ce_ref_2d(lp + 4, 2, 4);
    float l2 = ce_ref_2d(lp + 8, 1, 4);
    float expected = (l0 + l1 + l2) / 3.0f;
    assert(fabsf(loss_p[0] - expected) < EPS && "ce batch forward");

    dnn_backward(ctx.scratch, loss);
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
    tensor *logits = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0]=1.0f; lp[1]=2.0f; lp[2]=3.0f;
    lp[3]=4.0f; lp[4]=5.0f; lp[5]=6.0f;

    tensor *target = tensor_zeros(ctx.params, 1, (int[]){2}, 0);
    int *td = (int*)tensor_data_ptr(target);
    td[0] = 1; td[1] = 0;

    tensor *loss = tensor_cross_entropy(ctx.scratch, logits, target, 1);
    dnn_backward(ctx.scratch, loss);
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
    tensor *logits = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0] = 1000.0f; lp[1] = 1001.0f; lp[2] = 1002.0f;

    tensor *target = tensor_zeros(ctx.params, 1, (int[]){1}, 0);
    ((int*)tensor_data_ptr(target))[0] = 2;

    tensor *loss = tensor_cross_entropy(ctx.scratch, logits, target, 0);
    float *loss_p = tensor_data_ptr(loss);
    assert(isfinite(loss_p[0]) && "ce stability");

    /* also very negative */
    lp[0] = -1000.0f; lp[1] = -1001.0f; lp[2] = -1002.0f;
    tensor *loss2 = tensor_cross_entropy(ctx.scratch, logits, target, 0);
    float *loss_p2 = tensor_data_ptr(loss2);
    assert(isfinite(loss_p2[0]) && "ce stability neg");
    printf("OK\n");
}

static void test_cross_entropy_chain(void) {
    printf("  test_cross_entropy_chain... ");
    /* cross_entropy * scalar → backward */
    tensor *logits = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *lp = tensor_data_ptr(logits);
    lp[0]=1; lp[1]=2; lp[2]=3; lp[3]=4;

    tensor *target = tensor_zeros(ctx.params, 1, (int[]){1}, 0);
    ((int*)tensor_data_ptr(target))[0] = 0;

    tensor *loss = tensor_cross_entropy(ctx.scratch, logits, target, 0);
    tensor *scale = scalar(2.0f, 0);
    tensor *scaled = tensor_mul(ctx.scratch, loss, scale);
    dnn_backward(ctx.scratch, scaled);

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
    tensor *t = tensor_zeros(ctx.params, 3, (int[]){2, 3, 4}, 1);
    float *tp = tensor_data_ptr(t);
    for (int i = 0; i < 24; i++) tp[i] = (float)(i + 1);

    tensor *s = tensor_sum(ctx.scratch, t, 1);        /* (2,1,4) */
    tensor *p = tensor_pow(ctx.scratch, s, 2.0f);     /* (2,1,4) */
    tensor *r = tensor_relu(ctx.scratch, p);          /* (2,1,4) */
    tensor *loss = tensor_sum(ctx.scratch, r, 0);     /* (1,1,4) */
    dnn_backward(ctx.scratch, loss);

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
    tensor *x = tensor_zeros(ctx.params, 2, (int[]){3, 4}, 1);
    float *xp = tensor_data_ptr(x);
    for (int i = 0; i < 12; i++) xp[i] = (float)(i + 1);

    tensor *weight = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *wp = tensor_data_ptr(weight);
    for (int j = 0; j < 4; j++) wp[j] = 1.0f;

    tensor *out = tensor_layer_norm(ctx.scratch, x, weight, NULL, 1e-5f);
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
    tensor *x = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2; xp[2]=3;
    xp[3]=4; xp[4]=5; xp[5]=6;

    tensor *w = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *wp = tensor_data_ptr(w);
    wp[0]=1; wp[1]=1; wp[2]=1;

    tensor *b = tensor_zeros(ctx.params, 1, (int[]){3}, 1);

    tensor *out = tensor_layer_norm(ctx.scratch, x, w, b, 1e-5f);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);  /* scalar */
    dnn_backward(ctx.scratch, loss);

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
    tensor *x = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2; xp[2]=3; xp[3]=4;

    tensor *w = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    float *wp = tensor_data_ptr(w);
    wp[0]=1; wp[1]=1; wp[2]=1; wp[3]=1;

    tensor *out = tensor_layer_norm(ctx.scratch, x, w, NULL, 1e-5f);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

    assert(tensor_grad(x) != NULL);
    assert(tensor_grad(w) != NULL);
    printf("OK\n");
}

/* ── Conv2D tests ── */

static void test_conv_tiny(void) {
    printf("  test_conv_tiny... ");
    /* x=(1,1,2,2), w=(1,1,2,2), b=(1,), stride=1, pad=0
       out=[[[[5]]]], dx=[[[[1,0],[0,1]]]], dw=[[[[1,2],[3,4]]]], db=[1] */
    tensor *x = tensor_zeros(ctx.params, 4, (int[]){1,1,2,2}, 1);
    float *xp = tensor_data_ptr(x); xp[0]=1; xp[1]=2; xp[2]=3; xp[3]=4;
    tensor *w = tensor_zeros(ctx.params, 4, (int[]){1,1,2,2}, 1);
    float *wp = tensor_data_ptr(w); wp[0]=1; wp[1]=0; wp[2]=0; wp[3]=1;
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){1}, 1);

    tensor *out = tensor_conv2d(ctx.scratch, x, w, b, 1, 0);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

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
    tensor *x = tensor_randn(ctx.params, 4, (int[]){2,3,4,4}, 1);
    tensor *w = tensor_randn(ctx.params, 4, (int[]){2,3,1,1}, 1);
    tensor *b = tensor_randn(ctx.params, 1, (int[]){2}, 1);

    tensor *out = tensor_conv2d(ctx.scratch, x, w, b, 1, 0);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

    assert(tensor_grad(x) != NULL && "dx non-null");
    assert(tensor_grad(w) != NULL && "dw non-null");
    assert(tensor_grad(b) != NULL && "db non-null");
    printf("OK\n");
}

static void test_conv_no_bias(void) {
    printf("  test_conv_no_bias... ");
    srand(1);
    tensor *x = tensor_randn(ctx.params, 4, (int[]){1,2,3,3}, 1);
    tensor *w = tensor_randn(ctx.params, 4, (int[]){2,2,1,1}, 1);

    tensor *out = tensor_conv2d(ctx.scratch, x, w, NULL, 1, 0);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

    assert(tensor_grad(x) != NULL && "dx non-null");
    assert(tensor_grad(w) != NULL && "dw non-null");
    printf("OK\n");
}

static void test_conv_stride_pad(void) {
    printf("  test_conv_stride_pad... ");
    /* x=(1,1,4,4), w=(1,1,2,2), stride=2, pad=0 (from ref test 2)
       out = [[[[7,11],[23,27]]]] */
    tensor *x = tensor_zeros(ctx.params, 4, (int[]){1,1,4,4}, 1);
    float *xp = tensor_data_ptr(x);
    for (int i = 0; i < 16; i++) xp[i] = (float)(i + 1);
    tensor *w = tensor_zeros(ctx.params, 4, (int[]){1,1,2,2}, 1);
    float *wp = tensor_data_ptr(w); wp[0]=1; wp[1]=0; wp[2]=0; wp[3]=1;
    tensor *b = tensor_zeros(ctx.params, 1, (int[]){1}, 1);

    tensor *out = tensor_conv2d(ctx.scratch, x, w, b, 2, 0);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

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

/* ── Winograd-specific tests for 3×3 stride=1 pad=1 ── */

static void test_winograd_simple(void) {
    printf("  test_winograd_simple... ");
    tensor *x = tensor_zeros(ctx.params, 4, (int[]){1,1,4,4}, 1);
    float *xp = tensor_data_ptr(x);
    for (int i = 0; i < 16; i++) xp[i] = (float)i;
    tensor *w = tensor_zeros(ctx.params, 4, (int[]){1,1,3,3}, 1);
    float *wp = tensor_data_ptr(w);
    wp[0]=1; wp[1]=0; wp[2]=0;
    wp[3]=0; wp[4]=0; wp[5]=0;
    wp[6]=0; wp[7]=0; wp[8]=1;

    tensor *out = tensor_conv2d(ctx.scratch, x, w, NULL, 1, 1);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

    float *od = tensor_data_ptr(out);
    float exp_out[] = {5.0f, 6.0f, 7.0f, 0.0f, 9.0f, 10.0f, 12.0f, 2.0f,
                       13.0f, 18.0f, 20.0f, 6.0f, 0.0f, 8.0f, 9.0f, 10.0f};
    for (int i = 0; i < 16; i++)
        if (fabsf(od[i] - exp_out[i]) > 1e-4f) {
            printf("    FAIL: out[%d]: got %.4f, expected %.4f\n", i, od[i], exp_out[i]);
            assert(0);
        }

    float exp_dx[] = {1,1,1,0, 1,2,2,1, 1,2,2,1, 0,1,1,1};
    check_grad_ary(x, exp_dx, 16, "dx");

    float exp_dw[] = {45,66,54, 84,120,96, 81,114,90};
    check_grad_ary(w, exp_dw, 9, "dw");
    printf("OK\n");
}

static void test_winograd_multi_channel(void) {
    printf("  test_winograd_multi_channel... ");
    srand(0);
    tensor *x = tensor_randn(ctx.params, 4, (int[]){1,2,4,4}, 1);
    tensor *w = tensor_randn(ctx.params, 4, (int[]){3,2,3,3}, 1);
    tensor *b = tensor_randn(ctx.params, 1, (int[]){3}, 1);

    tensor *out = tensor_conv2d(ctx.scratch, x, w, b, 1, 1);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

    assert(tensor_grad(x) != NULL && "dx non-null");
    assert(tensor_grad(w) != NULL && "dw non-null");
    assert(tensor_grad(b) != NULL && "db non-null");
    assert(out->shape[0] == 1 && out->shape[1] == 3);
    assert(out->shape[2] == 4 && out->shape[3] == 4);
    printf("OK\n");
}

static void test_winograd_no_bias(void) {
    printf("  test_winograd_no_bias... ");
    srand(1);
    tensor *x = tensor_randn(ctx.params, 4, (int[]){2,1,6,6}, 1);
    tensor *w = tensor_randn(ctx.params, 4, (int[]){2,1,3,3}, 1);

    tensor *out = tensor_conv2d(ctx.scratch, x, w, NULL, 1, 1);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

    assert(tensor_grad(x) != NULL && "dx non-null");
    assert(tensor_grad(w) != NULL && "dw non-null");
    printf("OK\n");
}

static void test_ln_exact_pytorch(void) {
    printf("  test_ln_exact_pytorch... ");
    /* exact setup from ref_layer_norm.py backward test */
    tensor *x = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2; xp[2]=3;
    xp[3]=4; xp[4]=5; xp[5]=6;

    tensor *w = tensor_zeros(ctx.params, 1, (int[]){3}, 1);
    float *wp = tensor_data_ptr(w);
    wp[0]=1; wp[1]=1; wp[2]=1;

    tensor *b = tensor_zeros(ctx.params, 1, (int[]){3}, 1);

    tensor *out = tensor_layer_norm(ctx.scratch, x, w, b, 1e-5f);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);
    dnn_backward(ctx.scratch, loss);

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

    dnn_ctx_init(&ctx, 512 * 1024, 512 * 1024, 8*1024*1024);

    test_add_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_add_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_add_multi_use();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_add_diamond();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_add_broadcast();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_no_grad();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_add_self_via_grad_fn();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mul_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mul_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mul_self();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mul_broadcast();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mul_with_add();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_pow_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_pow_cube();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_pow_exp1();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_pow_neg();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_pow_broadcast();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_neg_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_neg_broadcast();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sub_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sub_self();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sub_broadcast();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_div_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_div_self();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_div_broadcast();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_div_neg_b();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sum_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sum_2d_dim0();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sum_2d_dim1();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sum_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mean_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mean_2d_dim1();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_relu_positive();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_relu_negative();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_relu_mixed();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_relu_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sigmoid_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sigmoid_positive();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sigmoid_negative();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sigmoid_array();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sigmoid_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_sigmoid_no_grad();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_silu_scalar();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_silu_array();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_silu_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_silu_no_grad();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_swiglu_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_swiglu_array();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_swiglu_broadcast_gate();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_swiglu_broadcast_up();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_swiglu_no_grad();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_swiglu_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_matmul_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_matmul_linear();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_matmul_square_self();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mixed_diamond();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_transpose_backward();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_slice_backward();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_mul_diamond_self();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_3d_multi_op();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_softmax_1d();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_softmax_2d_dim0();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_softmax_2d_dim1();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_softmax_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_softmax_stability();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_softmax_sum_to_one();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_causal_softmax_2d_forward();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_causal_softmax_4d_forward();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_causal_softmax_2d_backward();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_cross_entropy_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_cross_entropy_batch();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_cross_entropy_grad();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_cross_entropy_stability();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_cross_entropy_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_ln_forward_stats();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_ln_backward_grads();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_ln_no_bias();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_ln_exact_pytorch();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_conv_tiny();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_conv_backward_grads();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_conv_no_bias();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_conv_stride_pad();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_winograd_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_winograd_multi_channel();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_winograd_no_bias();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    printf("  ALL PASS\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
