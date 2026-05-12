#include "tensor.h"
#include "pool.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

static void test_randn_produces_non_nan_values(void) {
    int shape[] = {2, 3};
    tensor *r = tensor_randn(2, shape, 0);
    assert(r != NULL);
    assert(tensor_ndim(r) == 2);
    assert(tensor_shape(r, 0) == 2 && tensor_shape(r, 1) == 3);
    assert(tensor_numel(r) == 6);
    float *rp = tensor_data_ptr(r);
    /* standard-normal samping must produce valid floats */
    for (int i = 0; i < 6; i++) assert(!isnan(rp[i]));
}

static void test_reshape_contiguous_returns_view(void) {
    int shape2d[] = {2, 3};
    int shape1d[] = {6};
    tensor *a = tensor_zeros(2, shape2d, 0);
    float *ap = tensor_data_ptr(a);
    for (int i = 0; i < 6; i++) ap[i] = 1.0f;

    tensor *b = tensor_reshape(a, 1, shape1d);
    assert(b != NULL);
    assert(tensor_ndim(b) == 1);
    assert(tensor_shape(b, 0) == 6);
    /* contiguous input -> reshape is a zero-copy view — write via a visible in b */
    ap[3] = 99.0f;
    assert(tensor_data_ptr(b)[3] == 99.0f);
    assert(tensor_is_contiguous(b));
    float *bp = tensor_data_ptr(b);
    for (int i = 0; i < 6; i++) assert(bp[i] == (i == 3 ? 99.0f : 1.0f));
}

static void test_reshape_noncontiguous_copies_data(void) {
    int shape2d[] = {2, 3};
    int flat[] = {6};
    tensor *c = tensor_zeros(2, shape2d, 0);
    float *cp = tensor_data_ptr(c);
    for (int i = 0; i < 6; i++) cp[i] = 1.0f;

    /* transpose makes it non-contiguous */
    tensor *ct = tensor_transpose(c, 0, 1);
    assert(!tensor_is_contiguous(ct));

    /* reshape on non-contiguous must copy to a new contiguous buffer */
    tensor *d = tensor_reshape(ct, 1, flat);
    assert(d != NULL);
    assert(tensor_ndim(d) == 1);
    assert(tensor_shape(d, 0) == 6);
    assert(tensor_is_contiguous(d));
    float *dp = tensor_data_ptr(d);
    for (int i = 0; i < 6; i++) assert(dp[i] == 1.0f);
}

static void test_contiguous_on_already_contiguous_is_noop(void) {
    int shape2d[] = {2, 3};
    tensor *e = tensor_zeros(2, shape2d, 0);
    tensor *f = tensor_contiguous(e);
    /* already contiguous — result is contiguous with same data, no alloc */
    assert(tensor_is_contiguous(f));
    float *fp = tensor_data_ptr(f);
    for (int i = 0; i < 6; i++) assert(fp[i] == 0.0f);
}

static void test_slice_multi_element_contiguous(void) {
    int shape[] = {6};
    tensor *t = tensor_zeros(1, shape, 0);
    float *tp = tensor_data_ptr(t);
    for (int i = 0; i < 6; i++) tp[i] = (float)i;

    /* contiguous sub-range: [2,3,4] */
    tensor *s = tensor_slice(t, 0, 2, 3);
    assert(tensor_ndim(s) == 1);
    assert(tensor_shape(s, 0) == 3);
    assert(tensor_is_contiguous(s));
    float *sp = tensor_data_ptr(s);
    assert(sp[0] == 2.0f && sp[1] == 3.0f && sp[2] == 4.0f);
}

static void test_slice_of_noncontiguous_produces_correct_values(void) {
    int shape2d[] = {2, 3};
    tensor *g = tensor_zeros(2, shape2d, 0);
    float *gp = tensor_data_ptr(g);
    for (int i = 0; i < 6; i++) gp[i] = (float)i;  /* [[0,1,2],[3,4,5]] */

    tensor *gt = tensor_transpose(g, 0, 1);          /* [[0,3],[1,4],[2,5]] */
    assert(!tensor_is_contiguous(gt));

    /* slice keeps ndim, just shrinks the sliced dim */
    tensor *s = tensor_slice(gt, 0, 1, 1);
    assert(tensor_ndim(s) == 2);
    assert(tensor_shape(s, 0) == 1 && tensor_shape(s, 1) == 2);
    /* s is non-contiguous — contiguous() to read values */
    tensor *c = tensor_contiguous(s);
    assert(tensor_is_contiguous(c));
    assert(tensor_shape(c, 0) == 1 && tensor_shape(c, 1) == 2);
    assert(tensor_data_ptr(c)[0] == 1.0f);
    assert(tensor_data_ptr(c)[1] == 4.0f);
}

static void test_contiguous_on_noncontiguous_copies(void) {
    int shape2d[] = {2, 3};
    tensor *g = tensor_zeros(2, shape2d, 0);
    float *gp = tensor_data_ptr(g);
    for (int i = 0; i < 6; i++) gp[i] = 1.0f;

    tensor *gt = tensor_transpose(g, 0, 1);
    assert(!tensor_is_contiguous(gt));

    /* must produce a contiguous tensor with correct transposed shape */
    tensor *h = tensor_contiguous(gt);
    assert(tensor_is_contiguous(h));
    assert(tensor_shape(h, 0) == 3 && tensor_shape(h, 1) == 2);
    float *hp = tensor_data_ptr(h);
    for (int i = 0; i < 6; i++) assert(hp[i] == 1.0f);
}

int main(void) {
    printf("test_tensor_ext: randn, reshape, contiguous\n");

    mem_pool params  = mem_pool_create(64 * 1024);
    mem_pool scratch = mem_pool_create(64 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    test_randn_produces_non_nan_values();
    printf("  randn: OK\n");
    test_reshape_contiguous_returns_view();
    printf("  reshape (contiguous view): OK\n");
    test_reshape_noncontiguous_copies_data();
    printf("  reshape (non-contiguous copy): OK\n");
    test_contiguous_on_already_contiguous_is_noop();
    printf("  contiguous (already): OK\n");
    test_contiguous_on_noncontiguous_copies();
    printf("  contiguous (copy): OK\n");
    test_slice_multi_element_contiguous();
    printf("  slice multi-element (contiguous): OK\n");
    test_slice_of_noncontiguous_produces_correct_values();
    printf("  slice of non-contiguous: OK\n");

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);

    printf("    PASS\n");
    return 0;
}
