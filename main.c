#include "dnn.h"
#include "nn.h"
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* ── pools ── */
    mem_pool params  = mem_pool_create(10 * 1024 * 1024);     /* model (CNN: ~7MB) */
    mem_pool scratch = mem_pool_create(192 * 1024 * 1024);    /* scratch (CNN batch=128: ~160MB peak with P1) */
    mem_pool data    = mem_pool_create(210 * 1024 * 1024);    /* MNIST data (~210MB) */
    mem_pool_set_defaults(&params, &scratch, &data);

    /* ── MNIST CNN ── */

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

    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    mnist_model_cnn *m_cnn = mnist_model_create_cnn();
    printf("CNN model created.\n\n");

    printf("Training CNN (AdamW, lr=0.001, batch=128, max_epochs=3, patience=3):\n");
    mnist_train_cnn(m_cnn, train_images, train_labels, 3, 128, 0.001f, 5000, 3);

    printf("\nEvaluating CNN:\n");
    float cnn_train_acc = mnist_eval_cnn(m_cnn, train_images, train_labels);
    float cnn_test_acc  = mnist_eval_cnn(m_cnn, test_images, test_labels);
    printf("  Train accuracy: %.4f\n", cnn_train_acc);
    printf("  Test accuracy:  %.4f\n", cnn_test_acc);

    printf("\nAll done.\n");

cleanup:
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
    return 0;
}
