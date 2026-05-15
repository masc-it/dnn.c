#include "mnist.h"
#include <assert.h>

/* ================================================================
 *  CNN model (stride-2 downsampling)
 * ================================================================ */

mnist_model_cnn *mnist_model_create_cnn(struct mem_pool *params) {
    mnist_model_cnn *m = _mem_pool_alloc(params, sizeof(mnist_model_cnn), NULL);
    module_init(&m->base, params, "mnist_model_cnn");

    m->conv1 = conv2d_create(params, 1, 32, 3, 1, 1);
    module_add_child(&m->base, "conv1", &m->conv1->base);

    m->conv2 = conv2d_create(params, 32, 64, 3, 2, 1);
    module_add_child(&m->base, "conv2", &m->conv2->base);

    m->conv3 = conv2d_create(params, 64, 64, 3, 2, 1);
    module_add_child(&m->base, "conv3", &m->conv3->base);

    m->fc1 = linear_create(params, 3136, 128);
    module_add_child(&m->base, "fc1", &m->fc1->base);
    m->fc2 = linear_create(params, 128, 10);
    module_add_child(&m->base, "fc2", &m->fc2->base);

    return m;
}

tensor *mnist_model_forward_cnn(struct mem_pool *scratch, struct module *base, const tensor *x) {
    mnist_model_cnn *m = (mnist_model_cnn *)base;
    int N = tensor_shape(x, 0);
    tensor *h = tensor_reshape(scratch, (tensor*)x, 4, (int[]){N, 1, 28, 28});

    h = conv2d_forward(scratch, m->conv1, h);
    h = tensor_relu(scratch, h);

    h = conv2d_forward(scratch, m->conv2, h);
    h = tensor_relu(scratch, h);

    h = conv2d_forward(scratch, m->conv3, h);
    h = tensor_relu(scratch, h);

    h = tensor_reshape(scratch, h, 2, (int[]){N, -1});
    h = linear_forward(scratch, m->fc1, h);
    h = tensor_relu(scratch, h);
    h = tensor_dropout(scratch, h, 0.5f);
    h = linear_forward(scratch, m->fc2, h);
    return h;
}

/* ── CNN training wrapper ── */

void mnist_train_cnn(struct dnn_ctx *ctx, mnist_model_cnn *m,
                     tensor *train_images, tensor *train_labels,
                     int epochs, int batch_size, float lr,
                     int val_n, int patience) {
    mnist_train_impl(ctx, train_images, train_labels,
                     epochs, batch_size, lr, val_n, patience,
                     &m->base, mnist_model_forward_cnn);
}
