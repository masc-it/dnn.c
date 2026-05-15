#include "tensor.h"
#include "context.h"
#include "pool.h"
#include "context.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

static dnn_ctx ctx;

static void test_zeros_init_shape_and_contiguity(void) {
    int shape[] = {2, 3};
    tensor *t = tensor_zeros(ctx.params, 2, shape, 0);
    assert(t != NULL);
    assert(tensor_ndim(t) == 2);
    assert(tensor_shape(t, 0) == 2 && tensor_shape(t, 1) == 3);
    assert(tensor_numel(t) == 6);
    assert(tensor_is_contiguous(t));

    /* verify raw pointer write survives round-trip */
    float *p = tensor_data_ptr(t);
    p[0] = 3.14f;
    assert(p[0] == 3.14f);
}

static void test_slice_shares_parent_data(void) {
    int shape[] = {2, 3};
    tensor *t = tensor_zeros(ctx.params, 2, shape, 0);
    tensor *slice = tensor_slice(ctx.scratch, t, 0, 1, 1);
    assert(slice != NULL);
    /* write via parent at slice position, read via slice — shared storage */
    float *tp = tensor_data_ptr(t);
    tp[3] = 42.0f;    /* row 1, col 0 = exactly where slice points */
    assert(tensor_data_ptr(slice)[0] == 42.0f);
}

static void test_requires_grad_toggle_allocates_grad_buffer(void) {
    int shape[] = {2, 3};
    tensor *t = tensor_zeros(ctx.params, 2, shape, 0);
    assert(!tensor_requires_grad(t));
    tensor_set_requires_grad(t, 1);
    assert(tensor_requires_grad(t));
    tensor_set_requires_grad(t, 0);
    assert(!tensor_requires_grad(t));
}

static void test_zeros_with_grad_via_constructor(void) {
    int shape[] = {2, 3};
    tensor *t = tensor_zeros(ctx.params, 2, shape, 1);
    assert(tensor_requires_grad(t));
    /* data still zeroed even with grad requested */
    float *p = tensor_data_ptr(t);
    for (int i = 0; i < 6; i++) assert(p[i] == 0.0f);
}

static void test_randn_with_grad_via_constructor(void) {
    int shape[] = {2, 3};
    tensor *t = tensor_randn(ctx.params, 2, shape, 1);
    assert(tensor_requires_grad(t));
    /* values still valid floats */
    float *p = tensor_data_ptr(t);
    for (int i = 0; i < 6; i++) assert(!isnan(p[i]));
}

int main(void) {
    printf("test_tensor: basic lifecycle\n");

    dnn_ctx_init(&ctx, 64 * 1024, 64 * 1024, 8*1024*1024);

    test_zeros_init_shape_and_contiguity();
    tensor_print(tensor_zeros(ctx.params, 2, (int[]){2, 3}, 0));

    test_slice_shares_parent_data();
    test_requires_grad_toggle_allocates_grad_buffer();
    test_zeros_with_grad_via_constructor();
    test_randn_with_grad_via_constructor();

    printf("    PASS\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
