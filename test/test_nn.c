#include "dnn.h"
#include "nn.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define EPS 1e-5f

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

/* helper: check tensor data elements */
static void check_data_ary(tensor *t, const float *exp, int n, const char *label) {
    float *d = tensor_data_ptr(t);
    for (int i = 0; i < n; i++) {
        if (fabsf(d[i] - exp[i]) > EPS) {
            printf("    FAIL: %s[%d]: got %.4f, expected %.4f\n", label, i, d[i], exp[i]);
            assert(0);
        }
    }
}

/* ── Creation tests ── */

static void test_linear_create_shapes(void) {
    printf("  test_linear_create_shapes... ");
    linear *l = linear_create(4, 3);
    assert(l->in_features  == 4);
    assert(l->out_features == 3);
    assert(tensor_ndim(l->weight) == 2);
    assert(tensor_shape(l->weight, 0) == 4);
    assert(tensor_shape(l->weight, 1) == 3);
    assert(tensor_ndim(l->bias) == 1);
    assert(tensor_shape(l->bias, 0) == 3);
    assert(tensor_requires_grad(l->weight));
    assert(tensor_requires_grad(l->bias));
    printf("OK\n");
}

/* ── Forward tests ── */

static void test_linear_forward_single(void) {
    printf("  test_linear_forward_single... ");
    /* single sample: input (1, 3), weight (3, 2), bias (2) */
    linear *l = linear_create(3, 2);
    /* set weight & bias to known values */
    float *wp = tensor_data_ptr(l->weight);
    wp[0]=1; wp[1]=2; wp[2]=3; wp[3]=4; wp[4]=5; wp[5]=6;
    float *bp = tensor_data_ptr(l->bias);
    bp[0]=10; bp[1]=20;

    tensor *x = tensor_zeros(2, (int[]){1, 3}, 0);
    float *xp = tensor_data_ptr(x);
    xp[0]=2; xp[1]=3; xp[2]=4;

    tensor *y = linear_forward(l, x);
    /* y = x@W + b = [2*1+3*3+4*5, 2*2+3*4+4*6] + [10,20]
             = [2+9+20, 4+12+24] + [10,20] = [31, 40] + [10,20] = [41, 60] */
    float exp[] = {41.0f, 60.0f};
    check_data_ary(y, exp, 2, "y");
    printf("OK\n");
}

static void test_linear_forward_batch(void) {
    printf("  test_linear_forward_batch... ");
    /* batch (2, 3), weight (3, 2), bias (2) */
    linear *l = linear_create(3, 2);
    float *wp = tensor_data_ptr(l->weight);
    wp[0]=1; wp[1]=2; wp[2]=3; wp[3]=4; wp[4]=5; wp[5]=6;
    float *bp = tensor_data_ptr(l->bias);
    bp[0]=10; bp[1]=20;

    tensor *x = tensor_zeros(2, (int[]){2, 3}, 0);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2; xp[2]=3;  /* row 0 */
    xp[3]=4; xp[4]=5; xp[5]=6;  /* row 1 */

    tensor *y = linear_forward(l, x);
    /* row 0: [1*1+2*3+3*5, 1*2+2*4+3*6] + [10,20] = [1+6+15, 2+8+18]+[10,20] = [22,28]+[10,20] = [32,48] */
    /* row 1: [4*1+5*3+6*5, 4*2+5*4+6*6] + [10,20] = [4+15+30, 8+20+36]+[10,20] = [49,64]+[10,20] = [59,84] */
    float exp[] = {32.0f, 48.0f, 59.0f, 84.0f};
    check_data_ary(y, exp, 4, "y");
    printf("OK\n");
}

/* ── Backward / grad tests ── */

static void test_linear_backward_simple(void) {
    printf("  test_linear_backward_simple... ");
    /* y = x@W + b, with known values, verify grads */
    linear *l = linear_create(2, 2);
    float *wp = tensor_data_ptr(l->weight);
    wp[0]=1; wp[1]=2; wp[2]=3; wp[3]=4;
    float *bp = tensor_data_ptr(l->bias);
    bp[0]=0; bp[1]=0;

    tensor *x = tensor_zeros(2, (int[]){1, 2}, 0);
    float *xp = tensor_data_ptr(x);
    xp[0]=5; xp[1]=6;
    /* requires_grad for params already set; input has requires_grad=0 */

    tensor *y = linear_forward(l, x);
    dnn_backward(y);

    /* dy/dW = X^T @ dloss/dy = [5,6]^T @ [1,1] = [[5,5],[6,6]] */
    float exp_W[] = {5.0f, 5.0f, 6.0f, 6.0f};
    check_grad_ary(l->weight, exp_W, 4, "dW");

    /* db = sum(dloss/dy) along batch = [1, 1] */
    float exp_b[] = {1.0f, 1.0f};
    check_grad_ary(l->bias, exp_b, 2, "db");
    printf("OK\n");
}

static void test_linear_backward_batch(void) {
    printf("  test_linear_backward_batch... ");
    linear *l = linear_create(2, 3);
    float *wp = tensor_data_ptr(l->weight);
    wp[0]=1; wp[1]=2; wp[2]=3; wp[3]=4; wp[4]=5; wp[5]=6;
    float *bp = tensor_data_ptr(l->bias);
    bp[0]=0; bp[1]=0; bp[2]=0;

    /* batch (3, 2) */
    tensor *x = tensor_zeros(2, (int[]){3, 2}, 0);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2;
    xp[2]=3; xp[3]=4;
    xp[4]=5; xp[5]=6;

    tensor *y = linear_forward(l, x);
    dnn_backward(y);

    /* dW = X^T @ dloss/dy = X^T @ ones(3,3) */
    /* X^T = [[1,3,5],[2,4,6]], sum rows: dW = [[1+3+5, 1+3+5, 1+3+5],[2+4+6,2+4+6,2+4+6]] = [[9,9,9],[12,12,12]] */
    float exp_W[] = {9.0f, 9.0f, 9.0f, 12.0f, 12.0f, 12.0f};
    check_grad_ary(l->weight, exp_W, 6, "dW");

    /* db = sum(dloss/dy) over batch = [3, 3, 3] */
    float exp_b[] = {3.0f, 3.0f, 3.0f};
    check_grad_ary(l->bias, exp_b, 3, "db");
    printf("OK\n");
}

static void test_linear_backward_input_grad(void) {
    printf("  test_linear_backward_input_grad... ");
    /* verify gradient flows back to input when requires_grad=1 */
    linear *l = linear_create(2, 2);
    float *wp = tensor_data_ptr(l->weight);
    wp[0]=1; wp[1]=0; wp[2]=0; wp[3]=1;

    tensor *x = tensor_zeros(2, (int[]){1, 2}, 1);
    float *xp = tensor_data_ptr(x);
    xp[0]=3; xp[1]=4;

    tensor *y = linear_forward(l, x);
    dnn_backward(y);

    /* dy/dx = W^T = [[1,0],[0,1]] = identity, dloss/dy = [1,1] */
    /* dx = dloss/dy @ W^T = [1,1] @ I = [1,1] */
    float exp_x[] = {1.0f, 1.0f};
    check_grad_ary(x, exp_x, 2, "dx");
    printf("OK\n");
}

static void test_linear_chain(void) {
    printf("  test_linear_chain... ");
    /* two linear layers: y = (x @ W1 + b1) @ W2 + b2 */
    linear *l1 = linear_create(2, 3);
    linear *l2 = linear_create(3, 1);

    /* set known weights */
    float *w1p = tensor_data_ptr(l1->weight);
    w1p[0]=1; w1p[1]=2; w1p[2]=3; w1p[3]=4; w1p[4]=5; w1p[5]=6;
    float *b1p = tensor_data_ptr(l1->bias);
    b1p[0]=0; b1p[1]=0; b1p[2]=0;

    float *w2p = tensor_data_ptr(l2->weight);
    w2p[0]=1; w2p[1]=2; w2p[2]=3;  /* (3, 1) */
    float *b2p = tensor_data_ptr(l2->bias);
    b2p[0]=0;

    tensor *x = tensor_zeros(2, (int[]){1, 2}, 0);
    float *xp = tensor_data_ptr(x);
    xp[0]=1; xp[1]=2;

    tensor *h = linear_forward(l1, x);   /* (1, 3) */
    tensor *y = linear_forward(l2, h);   /* (1, 1) */
    dnn_backward(y);

    /* verify forward intermediate:
     * W1 shape [2,3] row-major: [[1,2,3],[4,5,6]]
     * h = x @ W1 =  [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
     * y = h @ W2 = 9*1 + 12*2 + 15*3 = 9+24+45 = 78 */
    float exp_h[] = {9.0f, 12.0f, 15.0f};
    check_data_ary(h, exp_h, 3, "h");

    float exp_y[] = {78.0f};
    check_data_ary(y, exp_y, 1, "y");

    /* gradients flow through both layers — non-zero */
    float *gw1 = tensor_grad(l1->weight);
    float *gb1 = tensor_grad(l1->bias);
    float *gw2 = tensor_grad(l2->weight);
    float *gb2 = tensor_grad(l2->bias);
    assert(gw1 && gb1 && gw2 && gb2 && "all params should have grads");

    /* dW1 exists */
    (void)gw1; (void)gb1; (void)gw2; (void)gb2;

    float exp_dL2[] = {9.0f, 12.0f, 15.0f};  /* h^T @ 1 */
    check_grad_ary(l2->weight, exp_dL2, 3, "dW2");
    check_grad_ary(l2->bias, (float[]){1.0f}, 1, "db2");
    printf("OK\n");
}

static void test_linear_no_bias(void) {
    printf("  test_linear_no_bias... ");
    /* verify linear layer still works — though our API always creates bias,
     * we can check that gradient computation still functions correctly */
    linear *l = linear_create(3, 2);
    float *wp = tensor_data_ptr(l->weight);
    wp[0]=1; wp[1]=2; wp[2]=3; wp[3]=4; wp[4]=5; wp[5]=6;
    /* zero out bias so it has no effect */
    float *bp = tensor_data_ptr(l->bias);
    bp[0]=0; bp[1]=0;

    tensor *x = tensor_zeros(2, (int[]){1, 3}, 0);
    float *xp = tensor_data_ptr(x);
    xp[0]=2; xp[1]=3; xp[2]=4;

    tensor *y = linear_forward(l, x);
    /* y = x@W + 0 = [31, 40] */
    float exp[] = {31.0f, 40.0f};
    check_data_ary(y, exp, 2, "y");
    printf("OK\n");
}

/* ── Integration: linear + relu ── */

static void test_linear_relu(void) {
    printf("  test_linear_relu... ");
    linear *l = linear_create(2, 3);
    float *wp = tensor_data_ptr(l->weight);
    wp[0]= 1; wp[1]=-1; wp[2]= 1;
    wp[3]=-1; wp[4]= 1; wp[5]=-1;
    float *bp = tensor_data_ptr(l->bias);
    bp[0]=0; bp[1]=0; bp[2]=0;

    tensor *x = tensor_zeros(2, (int[]){1, 2}, 0);
    float *xp = tensor_data_ptr(x);
    xp[0]= 5; xp[1]=-3;

    tensor *h = linear_forward(l, x);      /* (1, 3) */
    tensor *r = tensor_relu(h);
    tensor *s = tensor_sum(r, 0);          /* scalar (1,1) */
    dnn_backward(s);

    /* h = [5*1+(-3)*(-1), 5*(-1)+(-3)*1, 5*1+(-3)*(-1)] = [8, -8, 8] */
    /* relu: [8, 0, 8] */
    /* sum: scalar 16 */
    /* grad: dloss/dh = [1, 0, 1] (relu gate) */
    /* dW = x^T @ dloss/dh = [5,-3]^T @ [1,0,1] = [[5,0,5],[-3,0,-3]] */
    float exp_W[] = {5.0f, 0.0f, 5.0f, -3.0f, 0.0f, -3.0f};
    check_grad_ary(l->weight, exp_W, 6, "dW");

    /* db = dloss/dh = [1, 0, 1] */
    float exp_b[] = {1.0f, 0.0f, 1.0f};
    check_grad_ary(l->bias, exp_b, 3, "db");
    printf("OK\n");
}

int main(void) {
    printf("test_nn:\n");

    mem_pool params  = mem_pool_create(128 * 1024);
    mem_pool scratch = mem_pool_create(128 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    test_linear_create_shapes();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_forward_single();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_forward_batch();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_backward_simple();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_backward_batch();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_backward_input_grad();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_chain();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_no_bias();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_linear_relu();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);

    printf("  ALL PASS\n");
    return 0;
}
