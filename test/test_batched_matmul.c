#include "dnn.h"
#include "context.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

#define EPS 1e-4f

/* ── helpers ── */

static float max_abs_diff(const float *a, const float *b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(a[i] - b[i]);
        if (v > d) d = v;
    }
    return d;
}

static tensor *make_tensor(int ndim, const int *shape, const float *data, int requires_grad) {
    tensor *t = tensor_zeros(ctx.params, ndim, shape, requires_grad);
    if (data) {
        int n = tensor_numel(t);
        memcpy(tensor_data_ptr(t), data, n * sizeof(float));
    }
    return t;
}

static void check_grad_ary(tensor *t, const float *exp, int n, const char *label, int line) {
    float *g = tensor_grad(t);
    if (!g) {
        printf("    FAIL (line %d): %s: grad is NULL\n", line, label);
        assert(0);
    }
    float d = max_abs_diff(g, exp, n);
    if (d > EPS) {
        printf("    FAIL (line %d): %s: max diff %.6f\n", line, label, d);
        for (int i = 0; i < n; i++)
            printf("      [%d] got %.6f exp %.6f\n", i, g[i], exp[i]);
        assert(0);
    }
}

/* ──────────────────────────────────────────────
 *  1) 3D batched: [B, M, K] @ [B, K, N]
 * ────────────────────────────────────────────── */

static void test_3d_batched(void) {
    printf("  test_3d_batched... ");
    int B = 3, M = 2, K = 4, N = 5;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 3, (int[]){B, M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 3, (int[]){B, K, N}, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    assert(c->ndim == 3);
    assert(c->shape[0] == B && c->shape[1] == M && c->shape[2] == N);

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    float *bg = tensor_grad(b);
    assert(ag && bg && "grads not null");

    /* Verify grads are non-zero and finite */
    for (int i = 0; i < tensor_numel(a); i++) {
        assert(isfinite(ag[i]) && "da has inf/nan");
        assert(fabsf(ag[i]) > 0 && "da zero");
    }
    for (int i = 0; i < tensor_numel(b); i++) {
        assert(isfinite(bg[i]) && "db has inf/nan");
        assert(fabsf(bg[i]) > 0 && "db zero");
    }

    printf("OK  [B=%d,M=%d,K=%d,N=%d]\n", B, M, K, N);

}

/* ──────────────────────────────────────────────
 *  2) 3D with broadcast b: [B, M, K] @ [K, N]
 * ────────────────────────────────────────────── */

static void test_3d_broadcast_b(void) {
    printf("  test_3d_broadcast_b... ");
    int B = 2, M = 3, K = 4, N = 5;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 3, (int[]){B, M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 2, (int[]){K, N}, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    assert(c->ndim == 3);
    assert(c->shape[0] == B && c->shape[1] == M && c->shape[2] == N);

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    float *bg = tensor_grad(b);
    assert(ag && bg && "grads not null");
    assert(tensor_numel(b) == K * N);

    /* b is broadcast: gradient accumulated over batch dim.
     * Since a and grad_output are random, just check non-zero + finite */
    for (int i = 0; i < tensor_numel(a); i++)
        assert(isfinite(ag[i]) && "da inf/nan");
    for (int i = 0; i < tensor_numel(b); i++)
        assert(isfinite(bg[i]) && "db inf/nan");

    printf("OK\n");

}

/* ──────────────────────────────────────────────
 *  3) 3D with broadcast a: [M, K] @ [B, K, N]
 * ────────────────────────────────────────────── */

static void test_3d_broadcast_a(void) {
    printf("  test_3d_broadcast_a... ");
    int B = 3, M = 2, K = 4, N = 5;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 2, (int[]){M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 3, (int[]){B, K, N}, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    assert(c->ndim == 3);
    assert(c->shape[0] == B && c->shape[1] == M && c->shape[2] == N);

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    float *bg = tensor_grad(b);
    assert(ag && bg && "grads not null");

    for (int i = 0; i < tensor_numel(a); i++)
        assert(isfinite(ag[i]) && "da inf/nan");
    for (int i = 0; i < tensor_numel(b); i++)
        assert(isfinite(bg[i]) && "db inf/nan");

    printf("OK\n");

}

/* ──────────────────────────────────────────────
 *  4) 4D broadcast both ways
 * ────────────────────────────────────────────── */

static void test_4d_broadcast(void) {
    printf("  test_4d_broadcast... ");
    int B1 = 2, B2 = 3, M = 4, K = 5, N = 6;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 4, (int[]){B1, B2, M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 4, (int[]){1, B2, K, N}, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    assert(c->ndim == 4);
    assert(c->shape[0] == B1 && c->shape[1] == B2);
    assert(c->shape[2] == M && c->shape[3] == N);

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    float *bg = tensor_grad(b);
    assert(ag && bg && "grads not null");

    for (int i = 0; i < tensor_numel(a); i++)
        assert(isfinite(ag[i]) && "da inf/nan");
    for (int i = 0; i < tensor_numel(b); i++)
        assert(isfinite(bg[i]) && "db inf/nan");

    printf("OK  [B1=%d,B2=%d,M=%d,K=%d,N=%d]\n", B1, B2, M, K, N);

}

/* ──────────────────────────────────────────────
 *  5) 4D broadcast a: [1, B2, M, K] @ [B1, B2, K, N]
 * ────────────────────────────────────────────── */

static void test_4d_broadcast_a(void) {
    printf("  test_4d_broadcast_a... ");
    int B1 = 2, B2 = 3, M = 4, K = 5, N = 6;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 4, (int[]){1, B2, M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 4, (int[]){B1, B2, K, N}, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    assert(c->ndim == 4);
    assert(c->shape[0] == B1 && c->shape[1] == B2);
    assert(c->shape[2] == M && c->shape[3] == N);

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    float *bg = tensor_grad(b);
    assert(ag && bg && "grads not null");

    for (int i = 0; i < tensor_numel(a); i++)
        assert(isfinite(ag[i]) && "da inf/nan");
    for (int i = 0; i < tensor_numel(b); i++)
        assert(isfinite(bg[i]) && "db inf/nan");

    printf("OK\n");

}

/* ──────────────────────────────────────────────
 *  6) Self-matmul: [B, M, M] @ [B, M, M] (a == b)
 * ────────────────────────────────────────────── */

static void test_self_matmul(void) {
    printf("  test_self_matmul... ");
    int B = 2, M = 3;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 3, (int[]){B, M, M}, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, a);
    assert(c->ndim == 3);
    assert(c->shape[0] == B && c->shape[1] == M && c->shape[2] == M);

    dnn_backward(ctx.scratch, c);
    float *ag = tensor_grad(a);
    assert(ag && "self-matmul grad not null");
    for (int i = 0; i < tensor_numel(a); i++) {
        assert(isfinite(ag[i]) && "da inf/nan");
        assert(fabsf(ag[i]) > 0 && "da zero");
    }

    printf("OK\n");

}

/* ──────────────────────────────────────────────
 *  7) Single batch: [1, M, K] @ [1, K, N]
 *     Compare with equivalent 2D matmul
 * ────────────────────────────────────────────── */

static void test_single_batch_vs_2d(void) {
    printf("  test_single_batch_vs_2d... ");
    int M = 3, K = 4, N = 2;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    /* 3D batched version */
    tensor *a3 = tensor_randn(ctx.params, 3, (int[]){1, M, K}, 1);
    tensor *b3 = tensor_randn(ctx.params, 3, (int[]){1, K, N}, 1);
    tensor *c3 = tensor_matmul(ctx.scratch, a3, b3);

    /* Save forward output */
    int n_out = tensor_numel(c3);
    float *c3_data = malloc(n_out * sizeof(float));
    memcpy(c3_data, tensor_data_ptr(c3), n_out * sizeof(float));

    dnn_backward(ctx.scratch, c3);
    float *a3_grad = malloc(tensor_numel(a3) * sizeof(float));
    float *b3_grad = malloc(tensor_numel(b3) * sizeof(float));
    memcpy(a3_grad, tensor_grad(a3), tensor_numel(a3) * sizeof(float));
    memcpy(b3_grad, tensor_grad(b3), tensor_numel(b3) * sizeof(float));

    /* Reset and do 2D version */
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);

    srand(42);
    tensor *a2 = tensor_randn(ctx.params, 2, (int[]){M, K}, 1);
    tensor *b2 = tensor_randn(ctx.params, 2, (int[]){K, N}, 1);
    tensor *c2 = tensor_matmul(ctx.scratch, a2, b2);

    dnn_backward(ctx.scratch, c2);

    /* Compare forward: 3D [1,M,N] vs 2D [M,N] */
    float *c2_data = tensor_data_ptr(c2);
    float fd = max_abs_diff(c3_data, c2_data, M * N);
    assert(fd < EPS && "3D vs 2D forward mismatch");

    /* Compare grads: squeeze batch dim */
    float *a2g = tensor_grad(a2);
    float *b2g = tensor_grad(b2);
    float da_diff = max_abs_diff(a3_grad, a2g, M * K);
    float db_diff = max_abs_diff(b3_grad, b2g, K * N);
    assert(da_diff < EPS && "3D vs 2D da mismatch");
    assert(db_diff < EPS && "3D vs 2D db mismatch");

    free(c3_data);
    free(a3_grad);
    free(b3_grad);

    printf("OK  (fwd=%.2e da=%.2e db=%.2e)\n", fd, da_diff, db_diff);

}

/* ──────────────────────────────────────────────
 *  8) Reference values: compare against PyTorch
 * ────────────────────────────────────────────── */

static void test_reference_values(void) {
    printf("  test_reference_values... ");

    /* Values from ref_batched_matmul.py, seed=42, B=2, M=2, K=3, N=2 */
    int B = 2, M = 2, K = 3, N = 2;

    float a_data[] = {0.33669037f, 0.12880941f, 0.23446237f,
                      0.23033303f, -1.12285638f, -0.18632829f,
                      2.20820141f, -0.63799703f, 0.46165723f,
                      0.26735088f, 0.53490466f, 0.80935723f};
    float b_data[] = {1.11029029f, -1.68979895f,
                      -0.98895991f, 0.95797181f,
                      1.32213509f, 0.81718975f,
                      -0.76583862f, -0.75062233f,
                      1.35254776f, 0.68632191f,
                      -0.32775864f, 0.79496872f};
    float exp_c[]  = {0.55642760f, -0.25394303f,
                      1.11984539f, -1.61714685f,
                      -2.70535946f, -1.72839367f,
                      0.25346267f, 0.80985093f};
    float exp_da[] = {-0.57950866f, -0.03098810f, 2.13932490f,
                      -0.57950866f, -0.03098810f, 2.13932490f,
                      -1.51646090f, 2.03886962f, 0.46721008f,
                      -1.51646090f, 2.03886962f, 0.46721008f};
    float exp_db[] = {0.56702340f, 0.56702340f,
                      -0.99404699f, -0.99404699f,
                      0.04813407f, 0.04813407f,
                      2.47555232f, 2.47555232f,
                      -0.10309237f, -0.10309237f,
                      1.27101445f, 1.27101445f};

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);  /* ensure reproducible, though data overwritten */
    tensor *a = make_tensor(3, (int[]){B, M, K}, a_data, 1);
    tensor *b = make_tensor(3, (int[]){B, K, N}, b_data, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    float *cd = tensor_data_ptr(c);

    float cf = max_abs_diff(cd, exp_c, B * M * N);
    assert(cf < EPS && "forward ref mismatch");
    printf("  forward: OK (%.2e) ", cf);

    dnn_backward(ctx.scratch, c);

    check_grad_ary(a, exp_da, B * M * K, "da", __LINE__);
    check_grad_ary(b, exp_db, B * K * N, "db", __LINE__);
    printf("backward: OK\n");

}

/* ──────────────────────────────────────────────
 *  9) No-grad mode
 * ────────────────────────────────────────────── */

static void test_no_grad(void) {
    printf("  test_no_grad... ");
    int B = 2, M = 3, K = 4, N = 5;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 3, (int[]){B, M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 3, (int[]){B, K, N}, 1);

    dnn_grad_ctx gc = dnn_no_grad_enter();
    tensor *c = tensor_matmul(ctx.scratch, a, b);
    dnn_no_grad_exit(gc);

    assert(c->grad_fn == NULL && "c has grad_fn despite no_grad");
    assert(c->ndim == 3);
    assert(c->shape[0] == B && c->shape[1] == M && c->shape[2] == N);
    float *cd = tensor_data_ptr(c);
    for (int i = 0; i < tensor_numel(c); i++)
        assert(isfinite(cd[i]) && "no-grad output inf/nan");

    printf("OK\n");

}

/* ──────────────────────────────────────────────
 *  10) Numerical gradient (finite difference)
 * ────────────────────────────────────────────── */

static void test_numerical_grad(void) {
    printf("  test_numerical_grad...\n");
    int B = 1, M = 2, K = 3, N = 2;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 3, (int[]){B, M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 3, (int[]){B, K, N}, 1);

    int na = tensor_numel(a);
    int nb = tensor_numel(b);
    float *a_orig = malloc(na * sizeof(float));
    float *b_orig = malloc(nb * sizeof(float));
    memcpy(a_orig, tensor_data_ptr(a), na * sizeof(float));
    memcpy(b_orig, tensor_data_ptr(b), nb * sizeof(float));

    /* Autograd gradients */
    tensor *c = tensor_matmul(ctx.scratch, a, b);
    dnn_backward(ctx.scratch, c);
    float *a_grad_auto = malloc(na * sizeof(float));
    float *b_grad_auto = malloc(nb * sizeof(float));
    memcpy(a_grad_auto, tensor_grad(a), na * sizeof(float));
    memcpy(b_grad_auto, tensor_grad(b), nb * sizeof(float));

    /* Finite difference for a few elements */
    float h = 1e-4f;
    int n_checks = 0;

    /* Check a[0,0,0] */
    {
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        srand(42);
        tensor *a_p = make_tensor(3, (int[]){B, M, K}, a_orig, 1);
        tensor *b_p = make_tensor(3, (int[]){B, K, N}, b_orig, 1);

        float *ap = tensor_data_ptr(a_p);
        float orig = ap[0];
        ap[0] = orig + h;
        tensor *c_p = tensor_matmul(ctx.scratch, a_p, b_p);
        float loss_p = 0;
        float *cp = tensor_data_ptr(c_p);
        for (int i = 0; i < tensor_numel(c_p); i++) loss_p += cp[i];

        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        srand(42);
        tensor *a_n = make_tensor(3, (int[]){B, M, K}, a_orig, 1);
        tensor *b_n = make_tensor(3, (int[]){B, K, N}, b_orig, 1);
        ap = tensor_data_ptr(a_n);
        ap[0] = orig - h;
        tensor *c_n = tensor_matmul(ctx.scratch, a_n, b_n);
        float loss_n = 0;
        float *cn = tensor_data_ptr(c_n);
        for (int i = 0; i < tensor_numel(c_n); i++) loss_n += cn[i];

        float fd = (loss_p - loss_n) / (2.0f * h);
        float diff = fabsf(fd - a_grad_auto[0]);
        printf("    a[0,0,0]: fd=%.6f auto=%.6f diff=%.2e %s\n",
               fd, a_grad_auto[0], diff, diff < 0.01f ? "OK" : "FAIL");
        assert(diff < 0.01f);
        n_checks++;
    }

    /* Check b[0,0,0] */
    {
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        srand(42);
        tensor *a_p = make_tensor(3, (int[]){B, M, K}, a_orig, 1);
        tensor *b_p = make_tensor(3, (int[]){B, K, N}, b_orig, 1);

        float *bp = tensor_data_ptr(b_p);
        float orig = bp[0];
        bp[0] = orig + h;
        tensor *c_p = tensor_matmul(ctx.scratch, a_p, b_p);
        float loss_p = 0;
        float *cp = tensor_data_ptr(c_p);
        for (int i = 0; i < tensor_numel(c_p); i++) loss_p += cp[i];

        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        srand(42);
        tensor *a_n = make_tensor(3, (int[]){B, M, K}, a_orig, 1);
        tensor *b_n = make_tensor(3, (int[]){B, K, N}, b_orig, 1);
        bp = tensor_data_ptr(b_n);
        bp[0] = orig - h;
        tensor *c_n = tensor_matmul(ctx.scratch, a_n, b_n);
        float loss_n = 0;
        float *cn = tensor_data_ptr(c_n);
        for (int i = 0; i < tensor_numel(c_n); i++) loss_n += cn[i];

        float fd = (loss_p - loss_n) / (2.0f * h);
        float diff = fabsf(fd - b_grad_auto[0]);
        printf("    b[0,0,0]: fd=%.6f auto=%.6f diff=%.2e %s\n",
               fd, b_grad_auto[0], diff, diff < 0.01f ? "OK" : "FAIL");
        assert(diff < 0.01f);
        n_checks++;
    }

    printf("  numerical gradient: OK (%d checks)\n", n_checks);

    free(a_orig); free(b_orig);
    free(a_grad_auto); free(b_grad_auto);

}

/* ──────────────────────────────────────────────
 *  11) Test 2D still works (regression)
 * ────────────────────────────────────────────── */

static void test_2d_still_works(void) {
    printf("  test_2d_still_works... ");
    int M = 2, K = 3, N = 2;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {7, 8, 9, 10, 11, 12};

    tensor *a = make_tensor(2, (int[]){M, K}, a_data, 1);
    tensor *b = make_tensor(2, (int[]){K, N}, b_data, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    float *cd = tensor_data_ptr(c);
    assert(fabsf(cd[0] - 58.0f) < EPS);
    assert(fabsf(cd[1] - 64.0f) < EPS);
    assert(fabsf(cd[2] - 139.0f) < EPS);
    assert(fabsf(cd[3] - 154.0f) < EPS);

    dnn_backward(ctx.scratch, c);

    float exp_a[] = {15.0f, 19.0f, 23.0f, 15.0f, 19.0f, 23.0f};
    float exp_b[] = {5.0f, 5.0f, 7.0f, 7.0f, 9.0f, 9.0f};
    check_grad_ary(a, exp_a, M * K, "da", __LINE__);
    check_grad_ary(b, exp_b, K * N, "db", __LINE__);

    printf("OK\n");

}

/* ──────────────────────────────────────────────
 *  12) Test 2D self-matmul still works (regression)
 * ────────────────────────────────────────────── */

static void test_2d_self_still_works(void) {
    printf("  test_2d_self_still_works... ");
    int M = 2;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    float a_data[] = {1, 2, 3, 4};

    tensor *a = make_tensor(2, (int[]){M, M}, a_data, 1);
    tensor *c = tensor_matmul(ctx.scratch, a, a);
    float *cd = tensor_data_ptr(c);
    assert(fabsf(cd[0] - 7.0f) < EPS);
    assert(fabsf(cd[1] - 10.0f) < EPS);
    assert(fabsf(cd[2] - 15.0f) < EPS);
    assert(fabsf(cd[3] - 22.0f) < EPS);

    dnn_backward(ctx.scratch, c);
    float exp_a[] = {7.0f, 11.0f, 9.0f, 13.0f};
    check_grad_ary(a, exp_a, M * M, "da", __LINE__);

    printf("OK\n");

}

/* ──────────────────────────────────────────────
 *  13) Larger contiguous ND with check
 * ────────────────────────────────────────────── */

static void test_larger_batch(void) {
    printf("  test_larger_batch... ");
    int B = 7, M = 8, K = 16, N = 10;

    dnn_ctx_init(&ctx, 256 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024);

    srand(42);
    tensor *a = tensor_randn(ctx.params, 3, (int[]){B, M, K}, 1);
    tensor *b = tensor_randn(ctx.params, 3, (int[]){B, K, N}, 1);

    tensor *c = tensor_matmul(ctx.scratch, a, b);
    dnn_backward(ctx.scratch, c);

    float *cd = tensor_data_ptr(c);
    for (int i = 0; i < tensor_numel(c); i++)
        assert(isfinite(cd[i]) && "output inf/nan");

    float *ag = tensor_grad(a);
    float *bg = tensor_grad(b);
    for (int i = 0; i < tensor_numel(a); i++)
        assert(isfinite(ag[i]) && "da inf/nan");
    for (int i = 0; i < tensor_numel(b); i++)
        assert(isfinite(bg[i]) && "db inf/nan");

    printf("OK  [%d,%d,%d,%d]\n", B, M, K, N);

}

/* ──────────────────────────────────────────────
 *  Main
 * ────────────────────────────────────────────── */

int main(void) {
    printf("=== test_batched_matmul ===\n\n");

    test_2d_still_works();
    test_2d_self_still_works();
    test_3d_batched();
    test_3d_broadcast_b();
    test_3d_broadcast_a();
    test_4d_broadcast();
    test_4d_broadcast_a();
    test_self_matmul();
    test_single_batch_vs_2d();
    test_reference_values();
    test_no_grad();
    test_numerical_grad();
    test_larger_batch();

    printf("\nAll batched matmul tests passed.\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
