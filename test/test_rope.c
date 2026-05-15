#include "dnn.h"
#include "context.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

#define EPS 1e-5f

static float *_load_bin(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { *n_out = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);
    int n = (int)(bytes / sizeof(float));
    float *data = malloc(bytes);
    if (!data) { fclose(f); *n_out = 0; return NULL; }
    size_t nr = fread(data, sizeof(float), n, f);
    fclose(f);
    if ((int)nr != n) { free(data); *n_out = 0; return NULL; }
    *n_out = n;
    return data;
}

static int _float_eq(float a, float b) {
    return fabsf(a - b) < EPS;
}

static int _check_tensor(const float *got, const float *expected, int n,
                          const char *label) {
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (!_float_eq(got[i], expected[i])) {
            printf("    MISMATCH %s[%d]: got %.6f, expected %.6f (diff %.2e)\n",
                   label, i, got[i], expected[i], fabsf(got[i] - expected[i]));
            ok = 0;
            if (i >= 5) { printf("    (showing first 6 errors)\n"); break; }
        }
    }
    return ok;
}

static float *_tf(tensor *t) { return (float*)t->data + t->offset; }

/* ══════════════════════════════════════════════════════════════════
 *  Forward
 * ══════════════════════════════════════════════════════════════════ */

static void test_identity(void) {
    printf("  test_identity... ");
    int shape[4] = {1, 1, 1, 2};
    tensor *x = tensor_zeros(ctx.params, 4, shape, 0);
    _tf(x)[0] = 3.0f; _tf(x)[1] = 4.0f;
    int fshape[2] = {1, 1};
    tensor *cos_t = tensor_zeros(ctx.params, 2, fshape, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, fshape, 0);
    _tf(cos_t)[0] = 1.0f; _tf(sin_t)[0] = 0.0f;
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    assert(_float_eq(_tf(out)[0], 3.0f));
    assert(_float_eq(_tf(out)[1], 4.0f));
    printf("OK\n");
}

static void test_rotate_90(void) {
    printf("  test_rotate_90... ");
    int shape[4] = {1, 1, 1, 2};
    tensor *x = tensor_zeros(ctx.params, 4, shape, 0);
    _tf(x)[0] = 1.0f; _tf(x)[1] = 0.0f;
    int fshape[2] = {1, 1};
    tensor *cos_t = tensor_zeros(ctx.params, 2, fshape, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, fshape, 0);
    _tf(cos_t)[0] = 0.0f; _tf(sin_t)[0] = 1.0f;
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    assert(_float_eq(_tf(out)[0], 0.0f));
    assert(_float_eq(_tf(out)[1], 1.0f));
    printf("OK\n");
}

static void test_rotate_45(void) {
    printf("  test_rotate_45... ");
    float s2 = sqrtf(2.0f) / 2.0f;
    int shape[4] = {1, 1, 1, 2};
    tensor *x = tensor_zeros(ctx.params, 4, shape, 0);
    _tf(x)[0] = 1.0f; _tf(x)[1] = 0.0f;
    int fshape[2] = {1, 1};
    tensor *cos_t = tensor_zeros(ctx.params, 2, fshape, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, fshape, 0);
    _tf(cos_t)[0] = s2; _tf(sin_t)[0] = s2;
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    assert(_float_eq(_tf(out)[0], s2));
    assert(_float_eq(_tf(out)[1], s2));
    printf("OK\n");
}

static void test_multi_pos_multi_pair(void) {
    printf("  test_multi_pos_multi_pair... ");
    int N = 3, d = 4;
    int shape[4] = {1, 1, N, d};
    tensor *x = tensor_zeros(ctx.params, 4, shape, 0);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < d/2; k++) {
            _tf(x)[n * d + 2*k]     = (float)(n * 10 + k);
            _tf(x)[n * d + 2*k + 1] = (float)(n * 10 + k + 100);
        }
    int fshape[2] = {N, d/2};
    tensor *cos_t = tensor_zeros(ctx.params, 2, fshape, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, fshape, 0);
    for (int i = 0; i < N * d/2; i++) { _tf(cos_t)[i] = 1.0f; _tf(sin_t)[i] = 0.0f; }
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < d/2; k++) {
            assert(_float_eq(_tf(out)[n * d + 2*k],     (float)(n * 10 + k)));
            assert(_float_eq(_tf(out)[n * d + 2*k + 1], (float)(n * 10 + k + 100)));
        }
    printf("OK\n");
}

static void test_3d_input(void) {
    printf("  test_3d_input... ");
    int B = 2, N = 3, d = 4;
    tensor *x = tensor_zeros(ctx.params, 3, (int[]){B, N, d}, 0);
    for (int i = 0; i < B * N * d; i++) _tf(x)[i] = (float)i;
    int fshape[2] = {N, d/2};
    tensor *cos_t = tensor_zeros(ctx.params, 2, fshape, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, fshape, 0);
    for (int i = 0; i < N * d/2; i++) { _tf(cos_t)[i] = 1.0f; _tf(sin_t)[i] = 0.0f; }
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    for (int i = 0; i < B * N * d; i++) assert(_float_eq(_tf(out)[i], (float)i));
    printf("OK\n");
}

static void test_gold_data(void) {
    printf("  test_gold_data... ");
    int n_in, n_cos, n_sin, n_out;
    float *in_data  = _load_bin("rope_fwd_in.bin", &n_in);
    float *cos_data = _load_bin("rope_fwd_cos.bin", &n_cos);
    float *sin_data = _load_bin("rope_fwd_sin.bin", &n_sin);
    float *out_data = _load_bin("rope_fwd_out.bin", &n_out);
    if (!in_data || !cos_data || !sin_data || !out_data) {
        printf("SKIP\n");
        free(in_data); free(cos_data); free(sin_data); free(out_data); return;
    }
    int B = 2, H = 3, N = 5, d = 8;
    tensor *x = tensor_zeros(ctx.params, 4, (int[]){B, H, N, d}, 0);
    memcpy(_tf(x), in_data, n_in * sizeof(float));
    tensor *cos_t = tensor_zeros(ctx.params, 2, (int[]){N, d/2}, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, (int[]){N, d/2}, 0);
    memcpy(_tf(cos_t), cos_data, n_cos * sizeof(float));
    memcpy(_tf(sin_t), sin_data, n_sin * sizeof(float));
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    int ok = _check_tensor(_tf(out), out_data, n_out, "rope_fwd");
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(in_data); free(cos_data); free(sin_data); free(out_data);
}

/* ══════════════════════════════════════════════════════════════════
 *  Backward
 * ══════════════════════════════════════════════════════════════════ */

/* Manual backward: set upstream gradient on rope output,
 * call rope_backward directly, check x->grad matches gold. */
static void test_backward_gold(void) {
    printf("  test_backward_gold... ");
    int n_in, n_cos, n_sin, n_grad, n_dx;
    float *in_data  = _load_bin("rope_fwd_in.bin", &n_in);
    float *cos_data = _load_bin("rope_fwd_cos.bin", &n_cos);
    float *sin_data = _load_bin("rope_fwd_sin.bin", &n_sin);
    float *grad_data = _load_bin("rope_bwd_grad.bin", &n_grad);
    float *dx_data   = _load_bin("rope_bwd_dxin.bin", &n_dx);
    if (!in_data || !cos_data || !sin_data || !grad_data || !dx_data) {
        printf("SKIP\n");
        free(in_data); free(cos_data); free(sin_data); free(grad_data); free(dx_data); return;
    }
    int B = 2, H = 3, N = 5, d = 8;
    tensor *x = tensor_zeros(ctx.params, 4, (int[]){B, H, N, d}, 1);
    memcpy(_tf(x), in_data, n_in * sizeof(float));
    tensor *cos_t = tensor_zeros(ctx.params, 2, (int[]){N, d/2}, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, (int[]){N, d/2}, 0);
    memcpy(_tf(cos_t), cos_data, n_cos * sizeof(float));
    memcpy(_tf(sin_t), sin_data, n_sin * sizeof(float));
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    assert(out->grad_fn);
    out->grad = _mem_pool_alloc(ctx.scratch, tensor_numel(out) * sizeof(float), NULL);
    memcpy(out->grad, grad_data, n_grad * sizeof(float));
    tensor gv;
    memset(&gv, 0, sizeof(gv));
    gv.data = (void*)out->grad;
    gv.ndim = out->ndim;
    memcpy(gv.shape, out->shape, out->ndim * sizeof(int));
    int stride = 1;
    for (int d_i = out->ndim - 1; d_i >= 0; d_i--) { gv.strides[d_i] = stride; stride *= out->shape[d_i]; }
    gv.offset = 0; gv.contiguous = 1;
    out->grad_fn->backward(out->grad_fn, &gv);
    float *x_grad = tensor_grad(x);
    int ok = (x_grad != NULL) && _check_tensor(x_grad, dx_data, n_dx, "rope_bwd");
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(in_data); free(cos_data); free(sin_data); free(grad_data); free(dx_data);
}

/* Autograd-based backward: x → rope → sum → dnn_backward */
static void test_backward_autograd(void) {
    printf("  test_backward_autograd... ");
    int B = 1, H = 1, N = 2, d = 4;
    tensor *x = tensor_zeros(ctx.params, 4, (int[]){B, H, N, d}, 1);
    for (int i = 0; i < B * H * N * d; i++) _tf(x)[i] = (float)(i + 1);
    int fshape[2] = {N, d/2};
    tensor *cos_t = tensor_zeros(ctx.params, 2, fshape, 0);
    tensor *sin_t = tensor_zeros(ctx.params, 2, fshape, 0);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < d/2; k++) {
            float angle = (float)(n + 1) * 1.0f;
            _tf(cos_t)[n * (d/2) + k] = cosf(angle);
            _tf(sin_t)[n * (d/2) + k] = sinf(angle);
        }
    /* Build graph */
    tensor *out = tensor_rope(ctx.scratch, x, cos_t, sin_t);
    tensor *flat = tensor_flatten(ctx.scratch, out);
    tensor *loss = tensor_sum(ctx.scratch, flat, 0); /* [8] → scalar */
    dnn_backward(ctx.scratch, loss);
    float *g = tensor_grad(x);
    assert(g != NULL);
    for (int i = 0; i < B * H * N * d; i++) assert(isfinite(g[i]));
    float total = 0.0f;
    for (int i = 0; i < B * H * N * d; i++) total += fabsf(g[i]);
    assert(total > 0.0f);
    printf("OK (finite, non-zero)\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Frequency table
 * ══════════════════════════════════════════════════════════════════ */

static void test_freqs_init(void) {
    printf("  test_freqs_init... ");
    tensor *cos_t, *sin_t;
    tensor_rope_freqs_init(ctx.scratch, &cos_t, &sin_t, 8, 10, 10000.0f);
    assert(cos_t->shape[0] == 10 && cos_t->shape[1] == 4);
    assert(sin_t->shape[0] == 10 && sin_t->shape[1] == 4);
    for (int k = 0; k < 4; k++) {
        assert(_float_eq(_tf(cos_t)[k], 1.0f));
        assert(_float_eq(_tf(sin_t)[k], 0.0f));
    }
    for (int k = 0; k < 4; k++) {
        float theta = powf(10000.0f, -2.0f * k / 8.0f);
        for (int n = 0; n < 10; n++) {
            float angle = (float)n * theta;
            assert(_float_eq(_tf(cos_t)[n * 4 + k], cosf(angle)));
            assert(_float_eq(_tf(sin_t)[n * 4 + k], sinf(angle)));
        }
    }
    printf("OK\n");
}

static void test_freqs_init_custom_base(void) {
    printf("  test_freqs_init_custom_base... ");
    tensor *cos_t, *sin_t;
    tensor_rope_freqs_init(ctx.scratch, &cos_t, &sin_t, 4, 5, 500.0f);
    float theta0 = powf(500.0f, 0.0f);
    float theta1 = powf(500.0f, -2.0f / 4.0f);
    for (int n = 0; n < 5; n++) {
        assert(_float_eq(_tf(cos_t)[n * 2 + 0], cosf((float)n * theta0)));
        assert(_float_eq(_tf(sin_t)[n * 2 + 0], sinf((float)n * theta0)));
        assert(_float_eq(_tf(cos_t)[n * 2 + 1], cosf((float)n * theta1)));
        assert(_float_eq(_tf(sin_t)[n * 2 + 1], sinf((float)n * theta1)));
    }
    printf("OK\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {

    dnn_ctx_init(&ctx, 64 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    printf("=== RoPE tests ===\n\n");

    printf("-- Forward --\n");
    test_identity();
    test_rotate_90();
    test_rotate_45();
    test_multi_pos_multi_pair();
    test_3d_input();
    test_gold_data();

    printf("\n-- Backward --\n");
    test_backward_autograd();
    test_backward_gold();

    printf("\n-- Frequency table --\n");
    test_freqs_init();
    test_freqs_init_custom_base();

    printf("\n=== All tests done ===\n");

    dnn_ctx_destroy(&ctx);


    return 0;
}
