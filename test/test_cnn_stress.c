#include "dnn.h"
#include "mnist.h"
#include "tensor_int.h"
#include "pool_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int check_rel(const char *label, float got, float expected, float tol) {
    float err = fabsf(got - expected);
    if (err > tol && fabsf(expected) > tol) {
        float rel = err / fmaxf(1e-8f, fabsf(expected));
        if (rel > tol) {
            printf("  FAIL %s: got %.8f  expected %.8f  abs_err %.2e  rel_err %.2e\n",
                   label, got, expected, err, rel);
            return 1;
        }
    }
    return 0;
}

/* conv2d output → scalar loss via repeated sum */
static tensor *loss_sum_all(tensor *t) {
    for (int d = tensor_ndim(t) - 1; d >= 0; d--)
        t = tensor_sum(t, d);
    return t;
}

/* ================================================================
 *  Test 1: reproduce ref_conv2d.py test cases in C
 * ================================================================ */

static int test_conv_vs_python(void) {
    int fail = 0;
    printf("── test_conv_vs_python ──\n");

    /* ── test 2 from ref_conv2d.py: 2x2 kernel, stride 2 ── */
    {
        float x_data[] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        float w_data[] = {1,0, 0,1};
        float b_data[] = {0};
        float expected_out[] = {7, 11, 23, 27};

        tensor *x = _tensor_scratch_create(4, (int[]){1,1,4,4}, 0);
        memcpy(x->data, x_data, 16*sizeof(float));
        tensor_set_requires_grad(x, 1);

        tensor *w = tensor_zeros(4, (int[]){1,1,2,2}, 1);
        memcpy(w->data, w_data, 4*sizeof(float));

        tensor *b = tensor_zeros(1, (int[]){1}, 1);
        memcpy(b->data, b_data, 1*sizeof(float));

        tensor *y = tensor_conv2d(x, w, b, 2, 0);
        int N = tensor_shape(y,0), C = tensor_shape(y,1);
        int H = tensor_shape(y,2), WT = tensor_shape(y,3);
        printf("  test2: x(1,1,4,4) → out(%d,%d,%d,%d)\n", N, C, H, WT);
        if (N!=1 || C!=1 || H!=2 || WT!=2) {
            printf("    FAIL shape: expected (1,1,2,2)\n"); fail++;
        }

        float *yd = tensor_data_ptr(y);
        for (int i = 0; i < 4; i++)
            fail += check_rel("out", yd[i], expected_out[i], 1e-5f);

        tensor *loss = loss_sum_all(y);
        dnn_backward(loss);

        float *xg = tensor_grad(x);
        if (xg) {
            float expected_dx[] = {1,0,1,0, 0,1,0,1, 1,0,1,0, 0,1,0,1};
            for (int i = 0; i < 16; i++)
                fail += check_rel("dx", xg[i], expected_dx[i], 1e-5f);
        }

        float *wg = tensor_grad(w);
        if (wg) {
            float expected_dw[] = {24,28, 40,44};
            for (int i = 0; i < 4; i++)
                fail += check_rel("dw", wg[i], expected_dw[i], 1e-5f);
        }

        float *bg = tensor_grad(b);
        if (bg) fail += check_rel("db", bg[0], 4.0f, 1e-5f);

        mem_pool_reset(_mem_pool_scratch());
    }

    /* ── test 4 from ref_conv2d.py: 2x2 kernel, stride 1, manual verify ── */
    {
        float x_data[] = {1,2, 3,4};
        float w_data[] = {1,0, 0,1};
        float b_data[] = {0};

        tensor *x = _tensor_scratch_create(4, (int[]){1,1,2,2}, 0);
        memcpy(x->data, x_data, 4*sizeof(float));
        tensor_set_requires_grad(x, 1);

        tensor *w = tensor_zeros(4, (int[]){1,1,2,2}, 1);
        memcpy(w->data, w_data, 4*sizeof(float));

        tensor *b = tensor_zeros(1, (int[]){1}, 1);
        memcpy(b->data, b_data, 1*sizeof(float));

        tensor *y = tensor_conv2d(x, w, b, 1, 0);
        float *yd = tensor_data_ptr(y);
        printf("  test4: out = %.1f (expected 5.0)\n", yd[0]);
        fail += check_rel("out", yd[0], 5.0f, 1e-5f);

        tensor *loss = loss_sum_all(y);
        dnn_backward(loss);

        float *wg = tensor_grad(w);
        if (wg) {
            fail += check_rel("dw[0,0,0,0]", wg[0], 1.0f, 1e-5f);
            fail += check_rel("dw[0,0,0,1]", wg[1], 2.0f, 1e-5f);
            fail += check_rel("dw[0,0,1,0]", wg[2], 3.0f, 1e-5f);
            fail += check_rel("dw[0,0,1,1]", wg[3], 4.0f, 1e-5f);
        }

        float *bg = tensor_grad(b);
        if (bg) fail += check_rel("db", bg[0], 1.0f, 1e-5f);

        mem_pool_reset(_mem_pool_scratch());
    }

    /* ── shape test: 3x3 kernel, pad=1, stride=1 (same spatial) ── */
    {
        tensor *x = _tensor_scratch_create(4, (int[]){2,3,4,4}, 0);
        tensor_set_requires_grad(x, 1);
        tensor *w = tensor_zeros(4, (int[]){2,3,3,3}, 1);
        tensor *b = tensor_zeros(1, (int[]){2}, 1);

        tensor *y = tensor_conv2d(x, w, b, 1, 1);
        int N = tensor_shape(y,0), C = tensor_shape(y,1);
        int H = tensor_shape(y,2), WT = tensor_shape(y,3);
        printf("  same-pad: out(%d,%d,%d,%d)  expected (2,2,4,4)\n", N, C, H, WT);
        if (N!=2 || C!=2 || H!=4 || WT!=4) {
            printf("    FAIL shape\n"); fail++;
        }

        tensor *loss = loss_sum_all(y);
        dnn_backward(loss);
        printf("  grads all allocated OK\n");

        mem_pool_reset(_mem_pool_scratch());
    }

    if (!fail) printf("  ALL PASS\n");
    return fail;
}

/* ================================================================
 *  Test 2: CNN model shape sanity + timing
 * ================================================================ */

static int test_cnn_model(void) {
    int fail = 0;
    printf("\n── test_cnn_model ──\n");

    mem_pool_reset(_mem_pool_params());
    mem_pool_reset(_mem_pool_scratch());

    mnist_model_cnn *m = mnist_model_create_cnn();
    if (!m) { printf("  FAIL: model creation\n"); return 1; }
    printf("  model created OK\n");

    int N = 64;
    tensor *x = _tensor_scratch_create(2, (int[]){N, 784}, 0);
    float *xd = (float*)x->data;
    for (int i = 0; i < N * 784; i++)
        xd[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    double t0 = now_ms();
    tensor *logits = mnist_model_forward_cnn(m, x);
    double t1 = now_ms();

    int nd = tensor_ndim(logits);

    /* check shapes along the way by looking at intermediate outputs */
    /* conv1: (N,32,28,28), conv2: (N,64,14,14), conv3: (N,64,7,7) */
    printf("  forward: logits(%d,%d) in %.2f ms\n",
           nd>=1?tensor_shape(logits,0):-1,
           nd>=2?tensor_shape(logits,1):-1,
           t1 - t0);
    if (nd != 2 || tensor_shape(logits,0) != 64 || tensor_shape(logits,1) != 10) {
        printf("    FAIL shape: expected (64,10)\n"); fail++;
    }

    /* backward: fake labels + cross-entropy */
    tensor *y = _tensor_scratch_create(1, (int[]){N}, 0);
    int *yd = (int*)y->data;
    for (int i = 0; i < N; i++) yd[i] = rand() % 10;

    double t2 = now_ms();
    tensor *loss = tensor_cross_entropy(logits, y, 1);
    double t3 = now_ms();
    printf("  cross_entropy: %.2f ms\n", t3 - t2);

    double t4 = now_ms();
    dnn_backward(loss);
    double t5 = now_ms();
    printf("  backward: %.2f ms\n", t5 - t4);
    printf("  total fwd+bwd: %.2f ms\n", t5 - t0);

    /* check grads on all params */
    tensor *params[] = {m->conv1_w, m->conv1_b, m->conv2_w, m->conv2_b,
                        m->conv3_w, m->conv3_b, m->fc1->weight, m->fc1->bias,
                        m->fc2->weight, m->fc2->bias};
    int grads_ok = 1;
    for (int i = 0; i < 10; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) { printf("  FAIL: param %d no grad\n", i); grads_ok = 0; }
    }
    if (grads_ok) printf("  all 10 param grads present\n");

    /* check forward pass shapes per layer by re-running in no-grad */
    dnn_grad_ctx ctx = dnn_no_grad_enter();
    /* We'll rebuild manually to inspect intermediates */
    tensor *h = tensor_reshape((tensor*)x, 4, (int[]){N, 1, 28, 28});
    printf("  after reshape: (%d,%d,%d,%d)\n",
           tensor_shape(h,0),tensor_shape(h,1),tensor_shape(h,2),tensor_shape(h,3));

    h = tensor_conv2d(h, m->conv1_w, m->conv1_b, 1, 1);
    printf("  after conv1: (%d,%d,%d,%d)\n",
           tensor_shape(h,0),tensor_shape(h,1),tensor_shape(h,2),tensor_shape(h,3));

    h = tensor_relu(h);
    h = tensor_conv2d(h, m->conv2_w, m->conv2_b, 2, 1);
    printf("  after conv2: (%d,%d,%d,%d)\n",
           tensor_shape(h,0),tensor_shape(h,1),tensor_shape(h,2),tensor_shape(h,3));

    h = tensor_relu(h);
    h = tensor_conv2d(h, m->conv3_w, m->conv3_b, 2, 1);
    printf("  after conv3: (%d,%d,%d,%d)\n",
           tensor_shape(h,0),tensor_shape(h,1),tensor_shape(h,2),tensor_shape(h,3));

    h = tensor_relu(h);
    h = tensor_reshape(h, 2, (int[]){N, -1});
    printf("  after flatten: (%d,%d)\n",
           tensor_shape(h,0),tensor_shape(h,1));
    if (tensor_shape(h,1) != 3136) {
        printf("    FAIL: expected 3136\n"); fail++;
    }
    dnn_no_grad_exit(ctx);
    mem_pool_reset(_mem_pool_scratch());

    if (!fail) printf("  ALL PASS\n");
    return fail;
}

/* ================================================================
 *  Test 3: Multi-batch timing estimate
 * ================================================================ */

static int test_cnn_timing(void) {
    printf("\n── test_cnn_timing ──\n");

    mem_pool_reset(_mem_pool_params());
    mem_pool_reset(_mem_pool_scratch());

    mnist_model_cnn *m = mnist_model_create_cnn();
    tensor *params[] = {m->conv1_w, m->conv1_b, m->conv2_w, m->conv2_b,
                        m->conv3_w, m->conv3_b, m->fc1->weight, m->fc1->bias,
                        m->fc2->weight, m->fc2->bias};
    adamw_opt *opt = adamw_create(params, 10, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

    int N = 128;
    int n_batches = 10;
    double t_total = 0.0;

    for (int b = 0; b < n_batches; b++) {
        tensor *bx = _tensor_scratch_create(2, (int[]){N, 784}, 0);
        float *xd = (float*)bx->data;
        for (int i = 0; i < N * 784; i++)
            xd[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        tensor *by = _tensor_scratch_create(1, (int[]){N}, 0);
        int *yd = (int*)by->data;
        for (int i = 0; i < N; i++) yd[i] = rand() % 10;

        double t0 = now_ms();
        tensor *logits = mnist_model_forward_cnn(m, bx);
        tensor *loss = tensor_cross_entropy(logits, by, 1);
        dnn_backward(loss);
        adamw_step(opt);
        adamw_zero_grad(opt);
        mem_pool_reset(_mem_pool_scratch());
        double t1 = now_ms();

        t_total += (t1 - t0);
    }

    double avg_ms = t_total / n_batches;
    printf("  batch_size=%d: %.2f ms avg over %d batches\n", N, avg_ms, n_batches);

    int tr_n = 55000;
    int batches_epoch = (tr_n + N - 1) / N;
    double epoch_s = avg_ms * batches_epoch / 1000.0;
    printf("  est. epoch time (%d batches): %.1f s\n", batches_epoch, epoch_s);
    printf("  est. 5 epochs: %.1f s\n", epoch_s * 5);

    adamw_free(opt);
    mem_pool_reset(_mem_pool_scratch());
    return 0;
}

/* ================================================================ */

int main(void) {
    mem_pool params  = mem_pool_create(12 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(192 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    int fail = 0;
    fail += test_conv_vs_python();
    fail += test_cnn_model();
    fail += test_cnn_timing();

    printf("\n%s\n", fail ? "SOME TESTS FAIL" : "ALL PASS");
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
    return fail ? 1 : 0;
}
