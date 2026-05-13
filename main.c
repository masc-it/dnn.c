#include "dnn.h"
#include "nn.h"
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── XOR helpers ── */

static void train_xor(const char *label, mem_pool *scratch,
                      void (*step_fn)(void*),
                      void (*zero_grad_fn)(void*),
                      void *opt, linear *l1, linear *l2) {
    printf("── XOR MLP (%s, seed=42) ──\n", label);

    srand(42);

    tensor *X = tensor_zeros(2, (int[]){4, 2}, 0);
    float *xp = tensor_data_ptr(X);
    xp[0]=0; xp[1]=0; xp[2]=0; xp[3]=1; xp[4]=1; xp[5]=0; xp[6]=1; xp[7]=1;

    tensor *y = tensor_zeros(1, (int[]){4}, 0);
    int *yp = (int*)tensor_data_ptr(y);
    yp[0]=0; yp[1]=1; yp[2]=1; yp[3]=0;

    int epochs = 200;
    for (int epoch = 0; epoch < epochs; epoch++) {
        mem_pool_reset(scratch);

        tensor *h  = linear_forward(l1, X);
        tensor *r  = tensor_relu(h);
        tensor *logits = linear_forward(l2, r);
        tensor *loss   = tensor_cross_entropy(logits, y, 1);

        dnn_backward(loss);

        step_fn(opt);
        zero_grad_fn(opt);

        if (epoch % 40 == 0 || epoch == epochs - 1) {
            printf("  epoch %3d, loss %.6f\n", epoch, ((float*)loss->data)[0]);
        }
    }

    printf("\n  predictions:\n");
    dnn_grad_ctx ctx = dnn_no_grad_enter();
    tensor *h = linear_forward(l1, X);
    tensor *r = tensor_relu(h);
    tensor *logits = linear_forward(l2, r);
    dnn_no_grad_exit(ctx);

    float *ld = tensor_data_ptr(logits);
    for (int i = 0; i < 4; i++) {
        int pred = ld[i*2 + 1] > ld[i*2] ? 1 : 0;
        printf("    [%d %d] → %d  (true %d)%s\n",
               (int)xp[i*2], (int)xp[i*2+1], pred, yp[i],
               pred == yp[i] ? "" : " ✗");
    }
    printf("  done.\n\n");
}

static void sgd_step_wrap(void *p)   { sgd_step((sgd_opt*)p); }
static void sgd_zg_wrap(void *p)     { sgd_zero_grad((sgd_opt*)p); }
static void adamw_step_wrap(void *p) { adamw_step((adamw_opt*)p); }
static void adamw_zg_wrap(void *p)   { adamw_zero_grad((adamw_opt*)p); }

/* ── main ── */

int main(void) {
    /* ── pools ── */
    mem_pool params  = mem_pool_create(10 * 1024 * 1024);     /* model (CNN: ~7MB) */
    mem_pool scratch = mem_pool_create(192 * 1024 * 1024);    /* scratch (CNN batch=128: ~160MB peak with P1) */
    mem_pool data    = mem_pool_create(210 * 1024 * 1024);    /* MNIST data (~210MB) */
    mem_pool_set_defaults(&params, &scratch, &data);

    /* ── XOR demo ── */

    srand(42);
    linear *l1 = linear_create(2, 16);
    linear *l2 = linear_create(16, 2);
    tensor *sgd_params[] = {l1->weight, l1->bias, l2->weight, l2->bias};
    sgd_opt *sgd = sgd_create(sgd_params, 4, 0.1f, 0.9f);
    train_xor("SGD, lr=0.1, momentum=0.9", &scratch,
              sgd_step_wrap, sgd_zg_wrap, sgd, l1, l2);

    /* reset params for next demo */
    mem_pool_reset(&params);

    srand(42);
    linear *l1b = linear_create(2, 16);
    linear *l2b = linear_create(16, 2);
    tensor *aw_params[] = {l1b->weight, l1b->bias, l2b->weight, l2b->bias};
    adamw_opt *adamw = adamw_create(aw_params, 4, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
    train_xor("AdamW, lr=0.01, wd=0.01", &scratch,
              adamw_step_wrap, adamw_zg_wrap, adamw, l1b, l2b);

    /* ── MNIST ── */
    printf("\n═══ MNIST ═══\n\n");

    if (mnist_download() != 0) {
        fprintf(stderr, "MNIST download failed.\n");
        goto cleanup;
    }

    printf("Loading training data...\n");
    tensor *train_images = mnist_load_images("train-images-idx3-ubyte");
    tensor *train_labels = mnist_load_labels("train-labels-idx1-ubyte");
    if (!train_images || !train_labels) {
        fprintf(stderr, "Failed to load training data.\n");
        goto cleanup;
    }
    printf("  %d images loaded.\n", MNIST_TRAIN_N);

    printf("Loading test data...\n");
    tensor *test_images = mnist_load_images("t10k-images-idx3-ubyte");
    tensor *test_labels = mnist_load_labels("t10k-labels-idx1-ubyte");
    if (!test_images || !test_labels) {
        fprintf(stderr, "Failed to load test data.\n");
        goto cleanup;
    }
    printf("  %d images loaded.\n", MNIST_TEST_N);

    mnist_model *m = mnist_model_create();
    printf("Model created.\n\n");

    printf("Training (AdamW, lr=0.001, batch=64, max_epochs=20, patience=3):\n");
    mnist_train(m, train_images, train_labels, 20, 64, 0.001f,
                5000, 3);

    printf("\nEvaluating:\n");
    float train_acc = mnist_eval(m, train_images, train_labels);
    float test_acc  = mnist_eval(m, test_images, test_labels);
    printf("  Train accuracy: %.4f\n", train_acc);
    printf("  Test accuracy:  %.4f\n", test_acc);

    /* ── CNN ── */
    printf("\n═══ CNN ═══\n\n");

    /* free MLP memory before building CNN */
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    mnist_model_cnn *m_cnn = mnist_model_create_cnn();
    printf("CNN model created.\n\n");

    printf("Training CNN (AdamW, lr=0.001, batch=128, max_epochs=20, patience=3):\n");
    mnist_train_cnn(m_cnn, train_images, train_labels, 20, 128, 0.001f, 5000, 3);

    printf("\nEvaluating CNN:\n");
    float cnn_train_acc = mnist_eval_cnn(m_cnn, train_images, train_labels);
    float cnn_test_acc  = mnist_eval_cnn(m_cnn, test_images, test_labels);
    printf("  Train accuracy: %.4f\n", cnn_train_acc);
    printf("  Test accuracy:  %.4f\n", cnn_test_acc);

    /* ── done ── */
    printf("\nAll done.\n");

cleanup:
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
    return 0;
}
