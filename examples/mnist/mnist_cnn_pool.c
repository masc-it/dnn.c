#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 10*1024*1024, 192*1024*1024, 256*1024*1024);

    if (mnist_download() != 0) {
        fprintf(stderr, "MNIST download failed.\n");
        goto cleanup;
    }

    printf("Loading training data...\n");
    tensor *train_images = mnist_load_images(ctx.data, MNIST_DATA_DIR "/train-images-idx3-ubyte");
    tensor *train_labels = mnist_load_labels(ctx.data, MNIST_DATA_DIR "/train-labels-idx1-ubyte");
    if (!train_images || !train_labels) {
        fprintf(stderr, "Failed to load training data.\n");
        goto cleanup;
    }

    printf("Loading test data...\n");
    tensor *test_images = mnist_load_images(ctx.data, MNIST_DATA_DIR "/t10k-images-idx3-ubyte");
    tensor *test_labels = mnist_load_labels(ctx.data, MNIST_DATA_DIR "/t10k-labels-idx1-ubyte");
    if (!test_images || !test_labels) {
        fprintf(stderr, "Failed to load test data.\n");
        goto cleanup;
    }

    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    mnist_model_cnn_pool *m = mnist_model_create_cnn_pool(ctx.params);

    {
        long long n = module_num_parameters(&m->base);
        printf("CNN-pool model created.  Parameters: %lld (%.2fM)\n", n, n / 1e6);
        module_summary(&m->base, 0, 0);
        printf("\n");
    }

    printf("Training CNN-pool (AdamW, lr=0.001, batch=128, max_epochs=10, patience=3):\n");
    mnist_train_cnn_pool(&ctx, m, train_images, train_labels, 10, 128, 0.001f, 5000, 3);

    printf("\nEvaluating CNN-pool:\n");
    float train_acc = mnist_eval(ctx.scratch, &m->base, mnist_model_forward_cnn_pool,
                                  train_images, train_labels);
    float test_acc  = mnist_eval(ctx.scratch, &m->base, mnist_model_forward_cnn_pool,
                                  test_images, test_labels);
    printf("  Train accuracy: %.4f\n", train_acc);
    printf("  Test accuracy:  %.4f\n", test_acc);

    printf("\nAll done.\n");

cleanup:
    dnn_ctx_destroy(&ctx);
    return 0;
}
