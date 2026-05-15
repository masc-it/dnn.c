#!/usr/bin/env python3
"""Rewrite src/mnist.c with pool-aware signatures."""

import re

with open('src/mnist.c') as f:
    orig = f.read()

# 1. Includes
orig = orig.replace('#include "mnist.h"\n#include "tensor_int.h"\n#include "pool_int.h"',
                    '#include "mnist.h"\n#include "context.h"\n#include "tensor_int.h"\n#include "pool_int.h"')

# 2. Data loading functions — add data pool param
orig = orig.replace(
    'tensor *mnist_load_images(const char *path) {',
    'tensor *mnist_load_images(struct mem_pool *data, const char *path) {')
orig = orig.replace(
    'tensor *t = tensor_zeros_data(2, (int[]){N, img_size});',
    'tensor *t = tensor_zeros_data(data, 2, (int[]){N, img_size});')

orig = orig.replace(
    'tensor *mnist_load_labels(const char *path) {',
    'tensor *mnist_load_labels(struct mem_pool *data, const char *path) {')
orig = orig.replace(
    'tensor *t = tensor_zeros_data(1, (int[]){N});',
    'tensor *t = tensor_zeros_data(data, 1, (int[]){N});')

# 3. Model create — MLP
orig = orig.replace(
    'mnist_model *mnist_model_create(void) {\n    assert(MNIST_PIXELS > 0 && MNIST_CLASSES > 0);\n\n    mnist_model *m = mem_params_alloc(sizeof(mnist_model), NULL);\n    m->fc1 = linear_create(MNIST_PIXELS, 256);\n    m->fc2 = linear_create(256, MNIST_CLASSES);',
    'mnist_model *mnist_model_create(struct mem_pool *params_pool) {\n    assert(MNIST_PIXELS > 0 && MNIST_CLASSES > 0);\n\n    mnist_model *m = _mem_pool_alloc(params_pool, sizeof(mnist_model), NULL);\n    m->fc1 = linear_create(params_pool, MNIST_PIXELS, 256);\n    m->fc2 = linear_create(params_pool, 256, MNIST_CLASSES);')

# 4. Model forward — MLP
orig = orig.replace(
    'tensor *mnist_model_forward(mnist_model *m, const tensor *x) {\n    assert(m && x);\n    assert(x->ndim >= 2);\n    assert(x->shape[x->ndim - 1] == MNIST_PIXELS);\n\n    tensor *h = linear_forward(m->fc1, x);\n    h = tensor_relu(h);\n    h = tensor_dropout(h, 0.2f);\n    h = linear_forward(m->fc2, h);',
    'tensor *mnist_model_forward(struct mem_pool *scratch, mnist_model *m, const tensor *x) {\n    assert(m && x);\n    assert(x->ndim >= 2);\n    assert(x->shape[x->ndim - 1] == MNIST_PIXELS);\n\n    tensor *h = linear_forward(scratch, m->fc1, x);\n    h = tensor_relu(scratch, h);\n    h = tensor_dropout(scratch, h, 0.2f);\n    h = linear_forward(scratch, m->fc2, h);')

# 5. CNN model create
orig = orig.replace(
    'mnist_model_cnn *mnist_model_create_cnn(void) {\n    mnist_model_cnn *m = mem_params_alloc(sizeof(mnist_model_cnn), NULL);\n\n    float b1 = kaiming_bound(1, 3, 3);\n    m->conv1_w = tensor_uniform(4, (int[]){32, 1, 3, 3}, 1, b1);\n    m->conv1_b = tensor_zeros(1, (int[]){32}, 1);\n\n    float b2 = kaiming_bound(32, 3, 3);\n    m->conv2_w = tensor_uniform(4, (int[]){64, 32, 3, 3}, 1, b2);\n    m->conv2_b = tensor_zeros(1, (int[]){64}, 1);\n\n    float b3 = kaiming_bound(64, 3, 3);\n    m->conv3_w = tensor_uniform(4, (int[]){64, 64, 3, 3}, 1, b3);\n    m->conv3_b = tensor_zeros(1, (int[]){64}, 1);\n\n    m->fc1 = linear_create(3136, 128);\n    m->fc2 = linear_create(128, 10);',
    'mnist_model_cnn *mnist_model_create_cnn(struct mem_pool *params_pool) {\n    mnist_model_cnn *m = _mem_pool_alloc(params_pool, sizeof(mnist_model_cnn), NULL);\n\n    float b1 = kaiming_bound(1, 3, 3);\n    m->conv1_w = tensor_uniform(params_pool, 4, (int[]){32, 1, 3, 3}, 1, b1);\n    m->conv1_b = tensor_zeros(params_pool, 1, (int[]){32}, 1);\n\n    float b2 = kaiming_bound(32, 3, 3);\n    m->conv2_w = tensor_uniform(params_pool, 4, (int[]){64, 32, 3, 3}, 1, b2);\n    m->conv2_b = tensor_zeros(params_pool, 1, (int[]){64}, 1);\n\n    float b3 = kaiming_bound(64, 3, 3);\n    m->conv3_w = tensor_uniform(params_pool, 4, (int[]){64, 64, 3, 3}, 1, b3);\n    m->conv3_b = tensor_zeros(params_pool, 1, (int[]){64}, 1);\n\n    m->fc1 = linear_create(params_pool, 3136, 128);\n    m->fc2 = linear_create(params_pool, 128, 10);')

# 6. CNN model forward
orig = orig.replace(
    'tensor *mnist_model_forward_cnn(mnist_model_cnn *m, const tensor *x) {',
    'tensor *mnist_model_forward_cnn(struct mem_pool *scratch, mnist_model_cnn *m, const tensor *x) {')

# Fix internal calls in forward_cnn
orig = orig.replace(
    'tensor *h = tensor_reshape((tensor*)x, 4, (int[]){N, 1, 28, 28});\n\n    h = tensor_conv2d(h, m->conv1_w, m->conv1_b, 1, 1);\n    h = tensor_relu(h);\n\n    h = tensor_conv2d(h, m->conv2_w, m->conv2_b, 2, 1);\n    h = tensor_relu(h);\n\n    h = tensor_conv2d(h, m->conv3_w, m->conv3_b, 2, 1);\n    h = tensor_relu(h);\n\n    h = tensor_reshape(h, 2, (int[]){N, -1});\n    h = linear_forward(m->fc1, h);\n    h = tensor_relu(h);\n    h = tensor_dropout(h, 0.5f);\n    h = linear_forward(m->fc2, h);',
    'tensor *h = tensor_reshape(scratch, (tensor*)x, 4, (int[]){N, 1, 28, 28});\n\n    h = tensor_conv2d(scratch, h, m->conv1_w, m->conv1_b, 1, 1);\n    h = tensor_relu(scratch, h);\n\n    h = tensor_conv2d(scratch, h, m->conv2_w, m->conv2_b, 2, 1);\n    h = tensor_relu(scratch, h);\n\n    h = tensor_conv2d(scratch, h, m->conv3_w, m->conv3_b, 2, 1);\n    h = tensor_relu(scratch, h);\n\n    h = tensor_reshape(scratch, h, 2, (int[]){N, -1});\n    h = linear_forward(scratch, m->fc1, h);\n    h = tensor_relu(scratch, h);\n    h = tensor_dropout(scratch, h, 0.5f);\n    h = linear_forward(scratch, m->fc2, h);')

# 7. CNN pool variant create
orig = orig.replace(
    'mnist_model_cnn_pool *mnist_model_create_cnn_pool(void) {\n    mnist_model_cnn_pool *m = mem_params_alloc(sizeof(mnist_model_cnn_pool), NULL);',
    'mnist_model_cnn_pool *mnist_model_create_cnn_pool(struct mem_pool *params_pool) {\n    mnist_model_cnn_pool *m = _mem_pool_alloc(params_pool, sizeof(mnist_model_cnn_pool), NULL);')

# Fix tensor_uniform/tensor_zeros calls in create_cnn_pool (they still use old sigs)
# The function body has these calls that need params_pool
orig = orig.replace(
    'm->conv1_w = tensor_uniform(4, (int[]){32, 1, 3, 3}, 1, b1);\n    m->conv1_b = tensor_zeros(1, (int[]){32}, 1);',
    'm->conv1_w = tensor_uniform(params_pool, 4, (int[]){32, 1, 3, 3}, 1, b1);\n    m->conv1_b = tensor_zeros(params_pool, 1, (int[]){32}, 1);')

# Fix remaining tensor_uniform/tensor_zeros/linear_create in create_cnn_pool
# These need params_pool - will handle with a broader replacement below

# 8. CNN pool variant forward  
orig = orig.replace(
    'tensor *mnist_model_forward_cnn_pool(mnist_model_cnn_pool *m, const tensor *x) {',
    'tensor *mnist_model_forward_cnn_pool(struct mem_pool *scratch, mnist_model_cnn_pool *m, const tensor *x) {')

# Fix forward_cnn_pool internal calls
orig = orig.replace(
    'h = tensor_conv2d(h, m->conv1_w, m->conv1_b, 1, 1);  /* Winograd: 28×28 */\n    h = tensor_relu(h);',
    'h = tensor_conv2d(scratch, h, m->conv1_w, m->conv1_b, 1, 1);  /* Winograd: 28×28 */\n    h = tensor_relu(scratch, h);')
orig = orig.replace(
    'h = tensor_conv2d(h, m->conv2_w, m->conv2_b, 1, 1);  /* Winograd: 28×28 */\n    h = tensor_relu(h);',
    'h = tensor_conv2d(scratch, h, m->conv2_w, m->conv2_b, 1, 1);  /* Winograd: 28×28 */\n    h = tensor_relu(scratch, h);')
orig = orig.replace(
    'h = tensor_avg_pool2d(h, 2, 2);                       /* 14×14 */',
    'h = tensor_avg_pool2d(scratch, h, 2, 2);                       /* 14×14 */')
orig = orig.replace(
    'h = tensor_conv2d(h, m->conv3_w, m->conv3_b, 1, 1);  /* Winograd: 14×14 */\n    h = tensor_relu(h);',
    'h = tensor_conv2d(scratch, h, m->conv3_w, m->conv3_b, 1, 1);  /* Winograd: 14×14 */\n    h = tensor_relu(scratch, h);')
orig = orig.replace(
    'h = tensor_avg_pool2d(h, 2, 2);                       /* 7×7 */',
    'h = tensor_avg_pool2d(scratch, h, 2, 2);                       /* 7×7 */')

# Fix remaining reshape, linear_forward, dropout in forward_cnn_pool
orig = orig.replace(
    'tensor *h = tensor_reshape((tensor*)x, 4, (int[]){N, 1, 28, 28});',
    'tensor *h = tensor_reshape(scratch, (tensor*)x, 4, (int[]){N, 1, 28, 28});')
orig = orig.replace(
    'h = tensor_reshape(h, 2, (int[]){N, -1});  /* (N, 3136) */',
    'h = tensor_reshape(scratch, h, 2, (int[]){N, -1});  /* (N, 3136) */')
orig = orig.replace(
    'h = linear_forward(m->fc1, h);',
    'h = linear_forward(scratch, m->fc1, h);')
orig = orig.replace(
    'h = linear_forward(m->fc2, h);',
    'h = linear_forward(scratch, m->fc2, h);')
orig = orig.replace(
    'h = tensor_dropout(h, 0.5f);',
    'h = tensor_dropout(scratch, h, 0.5f);')

# 9. Replace remaining mem_params_alloc -> _mem_pool_alloc(params_pool,  
#    and mem_scratch_alloc -> _mem_pool_alloc(scratch,
#    and mem_data_alloc -> _mem_pool_alloc(data,
# These are in cnn_model_create_pool body and train_impl
orig = orig.replace('mem_params_alloc(', '_mem_pool_alloc(params_pool, ')
orig = orig.replace('mem_scratch_alloc(', '_mem_pool_alloc(scratch, ')
orig = orig.replace('mem_data_alloc(', '_mem_pool_alloc(data, ')

# But the above is wrong for functions where param is params not params_pool
# Let me fix the cnn_pool create function - it uses params_pool as param
# Actually, mem_params_alloc was replaced with _mem_pool_alloc(params_pool, 
# which works for create functions. For train_impl, we need different pool name.

# 10. Train impl and wrappers
# mnist_train_impl needs pool params
orig = orig.replace(
    'static void mnist_train_impl(\n    tensor *train_images, tensor *train_labels,\n    int epochs, int batch_size, float lr,\n    int val_n, int patience,\n    tensor **params, int n_params,\n    void *model, forward_fn_t forward_fn) {',
    'static void mnist_train_impl(\n    struct mem_pool *params_pool, struct mem_pool *scratch, struct mem_pool *data,\n    tensor *train_images, tensor *train_labels,\n    int epochs, int batch_size, float lr,\n    int val_n, int patience,\n    tensor **params, int n_params,\n    void *model, forward_fn_t forward_fn) {')

# adamw_create needs params_pool
orig = orig.replace(
    'adamw_opt *opt = adamw_create(params, n_params, lr,\n                                  0.9f, 0.999f, 1e-8f, 0.01f);',
    'adamw_opt *opt = adamw_create(params_pool, params, n_params, lr,\n                                  0.9f, 0.999f, 1e-8f, 0.01f);')

# forward_fn call needs scratch
orig = orig.replace(
    'tensor *logits = forward_fn(model, bx);',
    'tensor *logits = forward_fn(scratch, model, bx);')

# dnn_backward needs scratch
orig = orig.replace(
    'dnn_backward(loss);',
    'dnn_backward(scratch, loss);')

# tensor_scratch was already replaced by mem_scratch_alloc -> _mem_pool_alloc(scratch 
# But _tensor_scratch_create needs to be handled
orig = orig.replace('_tensor_scratch_create(', 'tensor_scratch(scratch, ')

# 11. Fix the misapplied _mem_pool_alloc(params_pool, -> in train_impl
# where "params" is the array of tensors, not the pool
# Inside train_impl, "params" is the tensor array. _mem_pool_alloc(params_pool, ...) 
# would actually try to use params_pool which is the variable name.
# This is correct because in train_impl, params_pool is the pool.

# But wait - in train_impl from step 10, params_pool IS the parameter name.
# So _mem_pool_alloc(params_pool, ...) should work.

# 12. Wrapper functions
orig = orig.replace(
    'void mnist_train(mnist_model *m,\n                 tensor *train_images, tensor *train_labels,\n                 int epochs, int batch_size, float lr,\n                 int val_n, int patience) {',
    'void mnist_train(struct dnn_ctx *ctx, mnist_model *m,\n                 tensor *train_images, tensor *train_labels,\n                 int epochs, int batch_size, float lr,\n                 int val_n, int patience) {')

orig = orig.replace(
    'void mnist_train_cnn(mnist_model_cnn *m,\n                     tensor *train_images, tensor *train_labels,\n                     int epochs, int batch_size, float lr,\n                     int val_n, int patience) {',
    'void mnist_train_cnn(struct dnn_ctx *ctx, mnist_model_cnn *m,\n                     tensor *train_images, tensor *train_labels,\n                     int epochs, int batch_size, float lr,\n                     int val_n, int patience) {')

orig = orig.replace(
    'void mnist_train_cnn_pool(mnist_model_cnn_pool *m,\n                          tensor *train_images, tensor *train_labels,\n                          int epochs, int batch_size, float lr,\n                          int val_n, int patience) {',
    'void mnist_train_cnn_pool(struct dnn_ctx *ctx, mnist_model_cnn_pool *m,\n                          tensor *train_images, tensor *train_labels,\n                          int epochs, int batch_size, float lr,\n                          int val_n, int patience) {')

# Fix wrapper calls to train_impl
orig = orig.replace(
    'mnist_train_impl(train_images, train_labels,\n                     epochs, batch_size, lr, val_n, patience,\n                     params, 4, m,\n                     (forward_fn_t)mnist_model_forward);',
    'mnist_train_impl(ctx->params, ctx->scratch, ctx->data, train_images, train_labels,\n                     epochs, batch_size, lr, val_n, patience,\n                     params, 4, m,\n                     (forward_fn_t)mnist_model_forward);')

orig = orig.replace(
    'mnist_train_impl(train_images, train_labels,\n                     epochs, batch_size, lr, val_n, patience,\n                     params, 10, m,\n                     (forward_fn_t)mnist_model_forward_cnn);',
    'mnist_train_impl(ctx->params, ctx->scratch, ctx->data, train_images, train_labels,\n                     epochs, batch_size, lr, val_n, patience,\n                     params, 10, m,\n                     (forward_fn_t)mnist_model_forward_cnn);')

orig = orig.replace(
    'mnist_train_impl(train_images, train_labels,\n                     epochs, batch_size, lr, val_n, patience,\n                     params, 10, m,\n                     (forward_fn_t)mnist_model_forward_cnn_pool);\n}',
    'mnist_train_impl(ctx->params, ctx->scratch, ctx->data, train_images, train_labels,\n                     epochs, batch_size, lr, val_n, patience,\n                     params, 10, m,\n                     (forward_fn_t)mnist_model_forward_cnn_pool);\n}')

# 13. mnist_eval_generic and inline evals
orig = orig.replace(
    'float mnist_eval_generic(void *model,\n                          tensor *(*forward_fn)(void *, const tensor *),\n                          tensor *images, tensor *labels) {',
    'float mnist_eval_generic(struct mem_pool *scratch, void *model,\n                          tensor *(*forward_fn)(struct mem_pool *, void *, const tensor *),\n                          tensor *images, tensor *labels) {')

orig = orig.replace(
    'tensor *bx = tensor_slice(images, 0, start, bs);',
    'tensor *bx = tensor_slice(scratch, images, 0, start, bs);')

orig = orig.replace(
    'tensor *logits = forward_fn(model, bx);',
    'tensor *logits = forward_fn(scratch, model, bx);')

# Fix the _mem_pool_alloc(params_pool, issue in tensor_zeros_data calls
# These should use data not params_pool
# Fix by checking what files reference data pool
# In train_impl, tensor_zeros_data calls were replaced with _mem_pool_alloc(data 
# but they should use tensor_zeros_data(data, ...)
# Actually tensor_zeros_data was not replaced by mem_data_alloc replacement since
# tensor_zeros_data is a function call not mem_data_alloc

# 14. Fix mem_pool_reset calls - replace _mem_pool_scratch/params/data
orig = orig.replace('_mem_pool_scratch()', 'scratch')
orig = orig.replace('_mem_pool_params()', 'params')
orig = orig.replace('_mem_pool_data()', 'data')

# Fix mem_pool_reset calls (they now reference scratch/params/data directly)
# These are inside functions that now have scratch/params/data as parameters

# 15. Fix forward_fn_t typedef
orig = orig.replace(
    'typedef tensor *(*forward_fn_t)(void *, const tensor *);',
    'typedef tensor *(*forward_fn_t)(struct mem_pool *, void *, const tensor *);')

with open('src/mnist.c', 'w') as f:
    f.write(orig)

print('mnist.c rewritten')
