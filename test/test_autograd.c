#include "dnn.h"
#include <stdio.h>
#include <assert.h>
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

int main(void) {
    printf("test_autograd:\n");

    mem_pool params  = mem_pool_create(64 * 1024);
    mem_pool scratch = mem_pool_create(64 * 1024);
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

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);

    printf("  ALL PASS\n");
    return 0;
}
