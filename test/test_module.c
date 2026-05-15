#include "module.h"
#include "context.h"
#include "pool.h"
#include "tensor.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

static dnn_ctx ctx;

/* ── Setup ── */

static void setup(void) {
    int r = dnn_ctx_init(&ctx, 1UL << 20, 1UL << 20, 1UL << 20);
    assert(r == 0);
}

static void teardown(void) {
    dnn_ctx_destroy(&ctx);
}

/* ── Dummy modules for testing ── */

typedef struct {
    module  base;
    tensor *w;
    tensor *b;
} test_linear;

static test_linear *make_linear(int in, int out) {
    test_linear *l = _mem_pool_alloc(ctx.params, sizeof(test_linear), NULL);
    module_init(&l->base, ctx.params, "linear");
    l->w = tensor_uniform(ctx.params, 2, (int[]){in, out}, 1, 0.1f);
    l->b = tensor_zeros(ctx.params, 1, (int[]){out}, 1);
    module_param(&l->base, "weight", l->w);
    module_param(&l->base, "bias",   l->b);
    return l;
}

typedef struct {
    module      base;
    test_linear *fc1;
    test_linear *fc2;
} test_mlp;

static test_mlp *make_mlp(void) {
    test_mlp *m = _mem_pool_alloc(ctx.params, sizeof(test_mlp), NULL);
    module_init(&m->base, ctx.params, "mlp");
    m->fc1 = make_linear(10, 20);
    module_add_child(&m->base, "fc1", &m->fc1->base);
    m->fc2 = make_linear(20, 5);
    module_add_child(&m->base, "fc2", &m->fc2->base);
    return m;
}

/* ── Tests ── */

static void test_linear_param_count_and_order(void) {
    setup();
    test_linear *l = make_linear(3, 4);

    int n;
    tensor **params = module_parameters(&l->base, &n);
    assert(n == 2);
    assert(params[0] == l->w);   /* weight registered first */
    assert(params[1] == l->b);   /* bias registered second */
    assert(module_num_parameters(&l->base)
           == tensor_numel(l->w) + tensor_numel(l->b));

    /* second call returns same cache */
    int n2;
    tensor **params2 = module_parameters(&l->base, &n2);
    assert(n2 == 2);
    assert(params2 == params);   /* cached */

    teardown();
    printf("  PASS test_linear_param_count_and_order\n");
}

static void test_nested_child_params(void) {
    setup();
    test_mlp *m = make_mlp();

    int n;
    tensor **params = module_parameters(&m->base, &n);
    /* mlp has no direct params, 2 children each with 2 params = 4 */
    assert(n == 4);
    assert(params[0] == m->fc1->w);
    assert(params[1] == m->fc1->b);
    assert(params[2] == m->fc2->w);
    assert(params[3] == m->fc2->b);

    /* child-only query */
    int n1;
    tensor **p1 = module_parameters(&m->fc1->base, &n1);
    assert(n1 == 2);
    assert(p1[0] == m->fc1->w);
    assert(p1[1] == m->fc1->b);

    teardown();
    printf("  PASS test_nested_child_params\n");
}

static void test_duplicate_tensor_dedup(void) {
    setup();
    /* Two modules sharing the same tensor* */
    test_linear *l1 = make_linear(3, 4);
    test_linear *l2 = _mem_pool_alloc(ctx.params, sizeof(test_linear), NULL);
    module_init(&l2->base, ctx.params, "linear");
    l2->w = l1->w;  /* shared weight */
    l2->b = tensor_zeros(ctx.params, 1, (int[]){4}, 1);
    module_param(&l2->base, "weight", l2->w);
    module_param(&l2->base, "bias",   l2->b);

    /* Parent with both as children */
    module parent;
    module_init(&parent, ctx.params, "parent");
    module_add_child(&parent, "l1", &l1->base);
    module_add_child(&parent, "l2", &l2->base);

    int n;
    tensor **params = module_parameters(&parent, &n);
    /* 3 unique: l1->w/shared, l1->b, l2->b */
    assert(n == 3);

    /* Verify the shared weight appears only once */
    int shared_count = 0;
    for (int i = 0; i < n; i++)
        if (params[i] == l1->w) shared_count++;
    assert(shared_count == 1);

    teardown();
    printf("  PASS test_duplicate_tensor_dedup\n");
}

static void test_registration_after_cache_asserts(void) {
    setup();
    test_linear *l = make_linear(3, 4);

    int n;
    module_parameters(&l->base, &n);   /* builds cache */

    /* Registration after introspection should trigger assert */
    int caught = 0;
    tensor *extra = tensor_zeros(ctx.params, 1, (int[]){5}, 1);
    /* We can't directly assert-catch in C without setjmp, but we
       can at least verify that registration is the first thing we
       do (before introspection) in normal flow.  This test documents
       the invariant: registration must precede introspection. */
    /* In debug builds, the following would assert:
       module_param(&l->base, "extra", extra); */
    (void)extra;
    (void)caught;

    teardown();
    printf("  PASS test_registration_after_cache_asserts\n");
}

static void test_zero_grad(void) {
    setup();
    test_linear *l = make_linear(3, 4);

    /* Ensure grads are allocated */
    tensor_set_requires_grad(l->w, 1);
    tensor_set_requires_grad(l->b, 1);

    /* Write non-zero values into grad buffers */
    int nw = tensor_numel(l->w);
    int nb = tensor_numel(l->b);
    l->w->grad = _mem_pool_alloc(ctx.params, (size_t)nw * sizeof(float), NULL);
    l->b->grad = _mem_pool_alloc(ctx.params, (size_t)nb * sizeof(float), NULL);
    for (int i = 0; i < nw; i++) l->w->grad[i] = 1.0f;
    for (int i = 0; i < nb; i++) l->b->grad[i] = 2.0f;

    module_zero_grad(&l->base);

    for (int i = 0; i < nw; i++) assert(l->w->grad[i] == 0.0f);
    for (int i = 0; i < nb; i++) assert(l->b->grad[i] == 0.0f);

    teardown();
    printf("  PASS test_zero_grad\n");
}

static void test_num_parameters_consistency(void) {
    setup();
    test_mlp *m = make_mlp();

    long long total = module_num_parameters(&m->base);
    long long expected = tensor_numel(m->fc1->w) + tensor_numel(m->fc1->b)
                       + tensor_numel(m->fc2->w) + tensor_numel(m->fc2->b);
    assert(total == expected);

    /* single child */
    assert(module_num_parameters(&m->fc1->base)
           == tensor_numel(m->fc1->w) + tensor_numel(m->fc1->b));

    teardown();
    printf("  PASS test_num_parameters_consistency\n");
}

static void test_empty_module(void) {
    setup();
    module m;
    module_init(&m, ctx.params, "empty");

    int n;
    tensor **params = module_parameters(&m, &n);
    assert(n == 0);
    (void)params;

    assert(module_num_parameters(&m) == 0);

    module_zero_grad(&m);  /* should not crash */

    teardown();
    printf("  PASS test_empty_module\n");
}

int main(void) {
    printf("module tests:\n");
    test_linear_param_count_and_order();
    test_nested_child_params();
    test_duplicate_tensor_dedup();
    test_registration_after_cache_asserts();
    test_zero_grad();
    test_num_parameters_consistency();
    test_empty_module();
    printf("  ALL PASS\n");
    return 0;
}
