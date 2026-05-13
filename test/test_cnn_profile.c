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

int main(void) {
    mem_pool params  = mem_pool_create(12 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(128 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    mnist_model_cnn *m = mnist_model_create_cnn();
    int N = 64;
    double t0, t1;

    /* dummy labels */
    tensor *y = _tensor_scratch_create(1, (int[]){N}, 0);
    for (int i = 0; i < N; i++) ((int*)y->data)[i] = i % 10;

    /* warmup */
    for (int w = 0; w < 3; w++) {
        tensor *x = _tensor_scratch_create(2, (int[]){N, 784}, 0);
        for (int i = 0; i < N*784; i++) ((float*)x->data)[i] = ((float)rand()/RAND_MAX)*2-1;
        tensor *h = mnist_model_forward_cnn(m, x);
        tensor *l = tensor_cross_entropy(h, y, 1);
        dnn_backward(l);
        mem_pool_reset(&scratch);
    }
    mem_pool_reset(&params);

    /* recreate model for clean state */
    m = mnist_model_create_cnn();

    printf("── per-layer forward breakdown (batch=%d) ──\n", N);
    mem_pool_reset(&scratch);

    tensor *x = _tensor_scratch_create(2, (int[]){N, 784}, 0);
    for (int i = 0; i < N*784; i++) ((float*)x->data)[i] = ((float)rand()/RAND_MAX)*2-1;

    tensor *h = tensor_reshape((tensor*)x, 4, (int[]){N, 1, 28, 28});

    t0 = now_ms();
    h = tensor_conv2d(h, m->conv1_w, m->conv1_b, 1, 1);
    t1 = now_ms(); printf("  conv1:   %6.2f ms\n", t1 - t0); t0 = t1;

    h = tensor_relu(h);
    t1 = now_ms(); printf("  relu1:   %6.2f ms\n", t1 - t0); t0 = t1;

    h = tensor_conv2d(h, m->conv2_w, m->conv2_b, 2, 1);
    t1 = now_ms(); printf("  conv2:   %6.2f ms\n", t1 - t0); t0 = t1;

    h = tensor_relu(h);
    t1 = now_ms(); printf("  relu2:   %6.2f ms\n", t1 - t0); t0 = t1;

    h = tensor_conv2d(h, m->conv3_w, m->conv3_b, 2, 1);
    t1 = now_ms(); printf("  conv3:   %6.2f ms\n", t1 - t0); t0 = t1;

    h = tensor_relu(h);
    t1 = now_ms(); printf("  relu3:   %6.2f ms\n", t1 - t0); t0 = t1;

    h = tensor_reshape(h, 2, (int[]){N, -1});
    h = linear_forward(m->fc1, h);
    t1 = now_ms(); printf("  fc1:     %6.2f ms\n", t1 - t0); t0 = t1;

    h = tensor_relu(h);
    h = tensor_dropout(h, 0.5f);
    h = linear_forward(m->fc2, h);
    t1 = now_ms(); printf("  fc2+etc: %6.2f ms\n", t1 - t0); t0 = t1;

    printf("\n── backward breakdown ──\n");
    mem_pool_reset(&scratch);

    /* fresh forward + full backward with per-op timing */
    x = _tensor_scratch_create(2, (int[]){N, 784}, 0);
    for (int i = 0; i < N*784; i++) ((float*)x->data)[i] = ((float)rand()/RAND_MAX)*2-1;

    /* forward without timing */
    h = tensor_reshape((tensor*)x, 4, (int[]){N, 1, 28, 28});
    h = tensor_conv2d(h, m->conv1_w, m->conv1_b, 1, 1);
    h = tensor_relu(h);
    tensor *c1 = h;
    h = tensor_conv2d(h, m->conv2_w, m->conv2_b, 2, 1);
    h = tensor_relu(h);
    tensor *c2 = h;
    h = tensor_conv2d(h, m->conv3_w, m->conv3_b, 2, 1);
    h = tensor_relu(h);
    tensor *c3 = h;
    h = tensor_reshape(h, 2, (int[]){N, -1});
    h = linear_forward(m->fc1, h);
    h = tensor_relu(h);
    h = tensor_dropout(h, 0.5f);
    h = linear_forward(m->fc2, h);

    tensor *loss = tensor_cross_entropy(h, y, 1);

    /* manually time the backward by marking moments */
    /* Since dnn_backward does everything at once, we can't easily time per-op.
       Instead we'll use the microbenchmark data to estimate. */
    t0 = now_ms();
    dnn_backward(loss);
    t1 = now_ms();
    printf("  total backward: %6.2f ms\n", t1 - t0);

    /* microbenchmark the conv2d backward contribution */
    printf("\n── estimated from sgemm microbench ──\n");
    printf("  conv2 fwd sgemm (12544,64,288):    0.36 ms\n");
    printf("  conv2 bwd d_weight (64,288,12544): 0.88 ms (slow: M=%d)\n", 64);
    printf("  conv2 bwd d_input (12544,288,64):  0.27 ms\n");
    printf("  ── conv2 total sgemm:              ~1.5 ms\n");
    printf("  im2col+col2im overhead is the rest\n");

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
    return 0;
}
