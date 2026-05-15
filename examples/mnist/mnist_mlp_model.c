#include "mnist.h"
#include <assert.h>

/* ================================================================
 *  MLP model
 * ================================================================ */

mnist_model *mnist_model_create(struct mem_pool *params) {
    mnist_model *m = _mem_pool_alloc(params, sizeof(mnist_model), NULL);
    module_init(&m->base, params, "mnist_model");
    m->fc1 = linear_create(params, MNIST_PIXELS, 256);
    module_add_child(&m->base, "fc1", &m->fc1->base);
    m->fc2 = linear_create(params, 256, MNIST_CLASSES);
    module_add_child(&m->base, "fc2", &m->fc2->base);
    return m;
}

tensor *mnist_model_forward(struct mem_pool *scratch, struct module *base, const tensor *x) {
    mnist_model *m = (mnist_model *)base;
    assert(m && x);
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == MNIST_PIXELS);

    tensor *h = linear_forward(scratch, m->fc1, x);
    h = tensor_relu(scratch, h);
    h = tensor_dropout(scratch, h, 0.2f);
    h = linear_forward(scratch, m->fc2, h);
    return h;
}

/* ── MLP training wrapper ── */

void mnist_train(struct dnn_ctx *ctx, mnist_model *m,
                 tensor *train_images, tensor *train_labels,
                 int epochs, int batch_size, float lr,
                 int val_n, int patience) {
    mnist_train_impl(ctx, train_images, train_labels,
                     epochs, batch_size, lr, val_n, patience,
                     &m->base, mnist_model_forward);
}
