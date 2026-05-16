/* Isolated RMSNorm backward precision test.
 *
 * Feeds non-uniform gradient through the graph (via mul+sum) so we're
 * only testing rms_backward with known grad_output.
 * Compares against analytical formula.
 */
#include "dnn.h"
#include "context.h"
#include "norm.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

int main(void) {

    dnn_ctx_init(&ctx, 512 * 1024, 512 * 1024, 8*1024*1024);

    int pass = 1;

    /* ── 1D test: single slice, easy to verify ── */
    printf("── 1D RMSNorm backward ──\n");
    srand(42);
    int d = 6;

    tensor *x = tensor_randn(ctx.params, 1, (int[]){d}, 1);
    tensor *w = tensor_zeros(ctx.params, 1, (int[]){d}, 1);
    float *wp = tensor_data_ptr(w);
    for (int j = 0; j < d; j++) wp[j] = 1.0f + 0.5f * (float)rand()/(float)RAND_MAX;

    /* non-uniform grad via scale vector */
    tensor *scale = tensor_zeros(ctx.params, 1, (int[]){d}, 0);
    float *sp = tensor_data_ptr(scale);
    for (int j = 0; j < d; j++) sp[j] = 1.0f + 2.0f * (float)rand()/(float)RAND_MAX;

    tensor *out   = tensor_rms_norm(ctx.scratch, x, w, 1e-5f);
    tensor *gated = tensor_mul(ctx.scratch, out, scale);
    tensor *loss  = tensor_sum(ctx.scratch, gated, 0);
    dnn_backward(ctx.scratch, loss);

    float *dx = tensor_grad(x);
    float *dw = tensor_grad(w);
    float *xd = tensor_data_ptr(x);

    /* analytical expectation */
    double sum_x2 = 0.0;
    for (int j = 0; j < d; j++) sum_x2 += (double)(xd[j] * xd[j]);
    float mean_x2 = (float)(sum_x2 / d);
    float rs = 1.0f / sqrtf(mean_x2 + 1e-5f);
    float rs2 = rs * rs;

    float S = 0.0f;
    for (int j = 0; j < d; j++) {
        float dy = sp[j] * wp[j];  /* d(loss)/d(y) = gd * w */
        S += dy * xd[j];
    }
    float S_scaled = rs2 / (float)d * S;

    float max_err_dx = 0.0f, max_err_dw = 0.0f;
    for (int j = 0; j < d; j++) {
        float dy  = sp[j] * wp[j];
        float exp_dx = rs * dy - rs * S_scaled * xd[j];
        float exp_dw = sp[j] * xd[j] * rs;

        float edx = fabsf(dx[j] - exp_dx);
        float edw = fabsf(dw[j] - exp_dw);
        if (edx > max_err_dx) max_err_dx = edx;
        if (edw > max_err_dw) max_err_dw = edw;
    }
    printf("  |dx-expected|_max = %.2e  (tol=1e-5)\n", max_err_dx);
    printf("  |dw-expected|_max = %.2e  (tol=1e-5)\n", max_err_dw);
    if (max_err_dx > 1e-5f || max_err_dw > 1e-5f) {
        printf("  FAIL\n"); pass = 0;
    } else printf("  PASS\n");
    printf("\n");

    /* reset for next test */
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    /* ── 2D batch test ── */
    printf("── 2D batch RMSNorm backward ──\n");
    srand(43);
    int B = 4;

    tensor *x2 = tensor_randn(ctx.params, 2, (int[]){B, d}, 1);
    float *x2d = tensor_data_ptr(x2);
    tensor *w2 = tensor_zeros(ctx.params, 1, (int[]){d}, 1);
    float *w2p = tensor_data_ptr(w2);
    for (int j = 0; j < d; j++) w2p[j] = 1.0f + 0.3f * (float)rand()/(float)RAND_MAX;

    tensor *s2 = tensor_zeros(ctx.params, 1, (int[]){d}, 0);
    float *s2p = tensor_data_ptr(s2);
    for (int j = 0; j < d; j++) s2p[j] = 1.0f + 0.5f * (float)rand()/(float)RAND_MAX;

    tensor *o2 = tensor_rms_norm(ctx.scratch, x2, w2, 1e-5f);
    tensor *g2 = tensor_mul(ctx.scratch, o2, s2);
    tensor *l2 = tensor_sum(ctx.scratch, g2, 0);
    dnn_backward(ctx.scratch, l2);

    float *dx2 = tensor_grad(x2);
    float *dw2 = tensor_grad(w2);

    float max_err_dx2 = 0.0f, max_err_dw2 = 0.0f;
    float exp_dw_buf[6]; memset(exp_dw_buf, 0, 6 * sizeof(float));

    for (int b = 0; b < B; b++) {
        double sx2 = 0.0;
        for (int j = 0; j < d; j++) sx2 += (double)(x2d[b * d + j] * x2d[b * d + j]);
        float m_x2 = (float)(sx2 / d);
        float rs = 1.0f / sqrtf(m_x2 + 1e-5f);
        float rs2 = rs * rs;

        /* pass 1: compute S */
        float S = 0.0f;
        for (int j = 0; j < d; j++) {
            float dyj = s2p[j] * w2p[j];
            S += dyj * x2d[b * d + j];
        }
        float S_scaled = rs2 / (float)d * S;

        /* pass 2: compute dx and accumulate dγ */
        for (int j = 0; j < d; j++) {
            float dyj = s2p[j] * w2p[j];
            float exp_dx = rs * dyj - rs * S_scaled * x2d[b * d + j];
            float err = fabsf(dx2[b * d + j] - exp_dx);
            if (err > max_err_dx2) max_err_dx2 = err;

            exp_dw_buf[j] += s2p[j] * x2d[b * d + j] * rs;
        }
    }

    for (int j = 0; j < d; j++) {
        float err = fabsf(dw2[j] - exp_dw_buf[j]);
        if (err > max_err_dw2) max_err_dw2 = err;
    }

    printf("  |dx-expected|_max = %.2e  (tol=2e-5)\n", max_err_dx2);
    printf("  |dw-expected|_max = %.2e  (tol=2e-5)\n", max_err_dw2);
    if (max_err_dx2 > 2e-5f || max_err_dw2 > 2e-5f) {
        printf("  FAIL\n"); pass = 0;
    } else printf("  PASS\n");
    printf("\n");

    /* reset */
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    /* ── 3D test: higher-dim input ── */
    printf("── 3D RMSNorm backward ──\n");
    srand(44);
    int B3 = 2, T = 3;
    int d3 = 5;

    tensor *x3 = tensor_randn(ctx.params, 3, (int[]){B3, T, d3}, 1);
    tensor *w3 = tensor_zeros(ctx.params, 1, (int[]){d3}, 1);
    float *w3p = tensor_data_ptr(w3);
    for (int j = 0; j < d3; j++) w3p[j] = 0.8f + 0.4f * (float)rand()/(float)RAND_MAX;

    tensor *o3 = tensor_rms_norm(ctx.scratch, x3, w3, 1e-5f);
    tensor *l3 = tensor_sum(ctx.scratch, o3, 0);
    dnn_backward(ctx.scratch, l3);

    float *dx3 = tensor_grad(x3);
    float *x3d = tensor_data_ptr(x3);
    float *dw3 = tensor_grad(w3);

    int n3 = tensor_numel(x3) / d3;  /* = B3 * T */
    float exp_dw3[5]; memset(exp_dw3, 0, 5 * sizeof(float));
    float max_err_dx3 = 0.0f;

    for (int s = 0; s < n3; s++) {
        double sx2 = 0.0;
        for (int j = 0; j < d3; j++) sx2 += (double)(x3d[s * d3 + j] * x3d[s * d3 + j]);
        float m_x2 = (float)(sx2 / d3);
        float rs = 1.0f / sqrtf(m_x2 + 1e-5f);
        float rs2 = rs * rs;

        float S = 0.0f;
        for (int j = 0; j < d3; j++) {
            float dyj = 1.0f * w3p[j];  /* gd=1 from sum backward */
            S += dyj * x3d[s * d3 + j];
        }
        float S_scaled = rs2 / (float)d3 * S;

        for (int j = 0; j < d3; j++) {
            float dyj = 1.0f * w3p[j];
            float exp_dx = rs * dyj - rs * S_scaled * x3d[s * d3 + j];
            float err = fabsf(dx3[s * d3 + j] - exp_dx);
            if (err > max_err_dx3) max_err_dx3 = err;
            exp_dw3[j] += 1.0f * x3d[s * d3 + j] * rs;
        }
    }

    float max_err_dw3 = 0.0f;
    for (int j = 0; j < d3; j++) {
        float err = fabsf(dw3[j] - exp_dw3[j]);
        if (err > max_err_dw3) max_err_dw3 = err;
    }

    printf("  |dx-expected|_max = %.2e  (tol=2e-5)\n", max_err_dx3);
    printf("  |dw-expected|_max = %.2e  (tol=2e-5)\n", max_err_dw3);
    if (max_err_dx3 > 2e-5f || max_err_dw3 > 2e-5f) {
        printf("  FAIL\n"); pass = 0;
    } else printf("  PASS\n");
    printf("\n");

    /* reset */
    mem_pool_reset(ctx.params);
    mem_pool_reset(ctx.scratch);

    /* ── no-weight test ── */
    printf("── 1D RMSNorm no-weight backward ──\n");
    srand(45);

    tensor *x4 = tensor_randn(ctx.params, 1, (int[]){d}, 1);
    float *x4d = tensor_data_ptr(x4);

    tensor *o4 = tensor_rms_norm(ctx.scratch, x4, NULL, 1e-5f);
    tensor *l4 = tensor_sum(ctx.scratch, o4, 0);
    dnn_backward(ctx.scratch, l4);

    float *dx4 = tensor_grad(x4);

    double sx4 = 0.0;
    for (int j = 0; j < d; j++) sx4 += (double)(x4d[j] * x4d[j]);
    float m_x4 = (float)(sx4 / d);
    float rs4 = 1.0f / sqrtf(m_x4 + 1e-5f);
    float rs42 = rs4 * rs4;

    float S4 = 0.0f;
    for (int j = 0; j < d; j++) S4 += 1.0f * x4d[j];  /* γ=1, dy=1 */
    float S4_scaled = rs42 / (float)d * S4;

    float max_err_dx4 = 0.0f;
    for (int j = 0; j < d; j++) {
        float exp_dx = rs4 * 1.0f - rs4 * S4_scaled * x4d[j];
        float err = fabsf(dx4[j] - exp_dx);
        if (err > max_err_dx4) max_err_dx4 = err;
    }
    printf("  |dx-expected|_max = %.2e  (tol=1e-5)\n", max_err_dx4);
    if (max_err_dx4 > 1e-5f) {
        printf("  FAIL\n"); pass = 0;
    } else printf("  PASS\n");
    printf("\n");

    printf(pass ? "ALL PASS\n" : "SOME FAILED\n");
    dnn_ctx_destroy(&ctx);

    return pass ? 0 : 1;
}
