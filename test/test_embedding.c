#include "dnn.h"
#include "context.h"
#include "ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

#define EPS 1e-5f

/* helper: check tensor data elements */
static void check_data_ary(tensor *t, const float *exp, int n, const char *label) {
    float *d = tensor_data_ptr(t);
    for (int i = 0; i < n; i++) {
        if (fabsf(d[i] - exp[i]) > EPS) {
            printf("    FAIL: %s[%d]: got %.6f, expected %.6f\n", label, i, d[i], exp[i]);
            assert(0);
        }
    }
}

/* helper: check tensor grad elements */
static void check_grad_ary(tensor *t, const float *exp, int n, const char *label) {
    float *g = tensor_grad(t);
    if (!g) { printf("    FAIL: %s: grad is NULL\n", label); assert(0); }
    for (int i = 0; i < n; i++) {
        if (fabsf(g[i] - exp[i]) > EPS) {
            printf("    FAIL: %s[%d]: got %.6f, expected %.6f\n", label, i, g[i], exp[i]);
            assert(0);
        }
    }
}

/* ── Forward tests ── */

static void test_embedding_simple(void) {
    printf("  test_embedding_simple... ");
    /* table [4, 3], ids [2] */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){4, 3}, 1);
    float *td = tensor_data_ptr(table);
    /* row 0: [1, 2, 3] */
    td[0]=1; td[1]=2; td[2]=3;
    /* row 1: [4, 5, 6] */
    td[3]=4; td[4]=5; td[5]=6;
    /* row 2: [7, 8, 9] */
    td[6]=7; td[7]=8; td[8]=9;
    /* row 3: [10, 11, 12] */
    td[9]=10; td[10]=11; td[11]=12;

    /* ids as int tensor: store via memset wrapping */
    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){2});
    int *idp = (int*)ids->data;
    idp[0] = 1;  /* lookup row 1 */
    idp[1] = 3;  /* lookup row 3 */

    tensor *out = tensor_embedding(ctx.scratch, table, ids);
    assert(tensor_ndim(out) == 2);
    assert(tensor_shape(out, 0) == 2);
    assert(tensor_shape(out, 1) == 3);

    /* expected: [[4,5,6], [10,11,12]] */
    float exp[] = {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};
    check_data_ary(out, exp, 6, "out");
    printf("OK\n");
}

static void test_embedding_batch(void) {
    printf("  test_embedding_batch... ");
    /* table [3, 2], ids [4] — larger batch */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 1);
    float *td = tensor_data_ptr(table);
    td[0]=10; td[1]=100;
    td[2]=20; td[3]=200;
    td[4]=30; td[5]=300;

    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){4});
    int *idp = (int*)ids->data;
    idp[0]=0; idp[1]=2; idp[2]=1; idp[3]=0;

    tensor *out = tensor_embedding(ctx.scratch, table, ids);
    assert(tensor_shape(out, 0) == 4);
    assert(tensor_shape(out, 1) == 2);

    float exp[] = {10, 100, 30, 300, 20, 200, 10, 100};
    check_data_ary(out, exp, 8, "out");
    printf("OK\n");
}

static void test_embedding_single_id(void) {
    printf("  test_embedding_single_id... ");
    /* table [5, 4], single id */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){5, 4}, 1);
    float *td = tensor_data_ptr(table);
    for (int i = 0; i < 20; i++) td[i] = (float)(i + 1);

    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){1});
    ((int*)ids->data)[0] = 0;

    tensor *out = tensor_embedding(ctx.scratch, table, ids);
    assert(tensor_shape(out, 0) == 1);
    assert(tensor_shape(out, 1) == 4);

    float exp[] = {1, 2, 3, 4};
    check_data_ary(out, exp, 4, "out");
    printf("OK\n");
}

/* ── Backward / grad tests ── */

static void test_embedding_backward_simple(void) {
    printf("  test_embedding_backward_simple... ");
    /* table [3, 2], ids [2], backward with dloss/dy = ones */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 1);
    float *td = tensor_data_ptr(table);
    td[0]=1; td[1]=2;
    td[2]=3; td[3]=4;
    td[4]=5; td[5]=6;

    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){2});
    int *idp = (int*)ids->data;
    idp[0] = 0;   /* look up row 0 */
    idp[1] = 2;   /* look up row 2 */

    tensor *out = tensor_embedding(ctx.scratch, table, ids);
    dnn_backward(ctx.scratch, out);

    /* d_table gradients:
     *   row 0 gets d_out[0] = [1,1]  → d_table[0] += [1,1]
     *   row 2 gets d_out[1] = [1,1]  → d_table[2] += [1,1]
     *   row 1 gets nothing           → d_table[1] = [0,0]
     */
    float exp_dt[] = {1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    check_grad_ary(table, exp_dt, 6, "d_table");
    printf("OK\n");
}

static void test_embedding_backward_scaled(void) {
    printf("  test_embedding_backward_scaled... ");
    /* table [2, 3], ids [3], loss = sum(out) so d_out = ones(3,3) */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){2, 3}, 1);
    float *td = tensor_data_ptr(table);
    td[0]=0.1f; td[1]=0.2f; td[2]=0.3f;
    td[3]=0.4f; td[4]=0.5f; td[5]=0.6f;

    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){3});
    int *idp = (int*)ids->data;
    idp[0]=1; idp[1]=0; idp[2]=1;  /* row 1, 0, 1 */

    tensor *out = tensor_embedding(ctx.scratch, table, ids);
    tensor *loss = tensor_sum(ctx.scratch, out, 0);  /* sum all -> scalar */
    dnn_backward(ctx.scratch, loss);

    /* d_table accumulation:
     *   row 0 appears once (id[1]) → gets 1.0f across d_model
     *   row 1 appears twice (id[0], id[2]) → gets 2.0f across d_model
     */
    float exp_dt[] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    check_grad_ary(table, exp_dt, 6, "d_table");
    printf("OK\n");
}

static void test_embedding_backward_duplicate_ids(void) {
    printf("  test_embedding_backward_duplicate_ids... ");
    /* Same id multiple times — must accumulate grads */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){2, 2}, 1);
    float *td = tensor_data_ptr(table);
    td[0]=1; td[1]=2;
    td[2]=3; td[3]=4;

    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){4});
    int *idp = (int*)ids->data;
    idp[0]=0; idp[1]=0; idp[2]=1; idp[3]=0;  /* row 0 appears 3×, row 1 appears 1× */

    tensor *out = tensor_embedding(ctx.scratch, table, ids);
    dnn_backward(ctx.scratch, out);

    /* row 0 gets 3× d_out = [1,1] → [3,3] */
    /* row 1 gets 1× d_out = [1,1] → [1,1] */
    float exp_dt[] = {3.0f, 3.0f, 1.0f, 1.0f};
    check_grad_ary(table, exp_dt, 4, "d_table");
    printf("OK\n");
}

static void test_embedding_no_grad(void) {
    printf("  test_embedding_no_grad... ");
    /* no-grad mode: no gradients allocated */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 0);  /* no grad */
    float *td = tensor_data_ptr(table);
    td[0]=1; td[1]=2;
    td[2]=3; td[3]=4;
    td[4]=5; td[5]=6;

    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){2});
    ((int*)ids->data)[0] = 0;
    ((int*)ids->data)[1] = 1;

    {
        dnn_grad_ctx gc = dnn_no_grad_enter();
        tensor *out = tensor_embedding(ctx.scratch, table, ids);
        dnn_no_grad_exit(gc);

        /* forward still works */
        assert(tensor_shape(out, 0) == 2);
        assert(tensor_shape(out, 1) == 2);
        float exp[] = {1, 2, 3, 4};
        check_data_ary(out, exp, 4, "out");

        /* no grad_fn on output */
        assert(out->grad_fn == NULL && "no-grad: grad_fn should be NULL");
    }
    printf("OK\n");
}

static void test_embedding_chain(void) {
    printf("  test_embedding_chain... ");
    /* embed -> linear: y = embed(ids) @ W + b */
    tensor *table = tensor_zeros(ctx.params, 2, (int[]){3, 2}, 1);
    float *td = tensor_data_ptr(table);
    td[0]=1; td[1]=2;
    td[2]=3; td[3]=4;
    td[4]=5; td[5]=6;

    tensor *ids = tensor_zeros_data(ctx.data, 1, (int[]){2});
    int *idp = (int*)ids->data;
    idp[0]=0; idp[1]=2;

    /* linear layer [2 -> 2] */
    linear *l = linear_create(ctx.params, 2, 2);
    float *wp = tensor_data_ptr(l->weight);
    wp[0]=1; wp[1]=0; wp[2]=0; wp[3]=1;  /* identity */
    float *bp = tensor_data_ptr(l->bias);
    bp[0]=0; bp[1]=0;

    tensor *e = tensor_embedding(ctx.scratch, table, ids);  /* [2, 2] */
    tensor *y = linear_forward(ctx.scratch, l, e);           /* [2, 2] */
    dnn_backward(ctx.scratch, y);

    /* Gradients flow back to embedding table */
    float *d_table = tensor_grad(table);
    assert(d_table != NULL && "d_table should exist");
    assert(fabsf(d_table[0]) > 0.0f && "d_table[0] non-zero");
    assert(fabsf(d_table[5]) > 0.0f && "d_table[5] non-zero");
    printf("OK\n");
}

int main(void) {
    printf("test_embedding:\n");

    dnn_ctx_init(&ctx, 128 * 1024, 128 * 1024, 128 * 1024);

    test_embedding_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_embedding_batch();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_embedding_single_id();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_embedding_backward_simple();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_embedding_backward_scaled();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_embedding_backward_duplicate_ids();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_embedding_no_grad();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    test_embedding_chain();
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    printf("  ALL PASS\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
