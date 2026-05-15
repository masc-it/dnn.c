/* Isolated LayerNorm backward precision test.
 *
 * Feeds a non-uniform gradient through the graph (via mul+sum) so we're
 * not testing sum_backward + ln_backward, just the ln_backward itself
 * with a known grad_output.  Compares against the analytical formula.
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
    printf("── 1D LayerNorm backward ──\n");
    srand(42);
    int d = 6;

    tensor *x = tensor_randn(ctx.params, 1, (int[]){d}, 1);
    tensor *w = tensor_zeros(ctx.params, 1, (int[]){d}, 1);
    float *wp = tensor_data_ptr(w);
    for (int j = 0; j < d; j++) wp[j] = 1.0f + 0.5f * (float)rand()/(float)RAND_MAX;

    /* non-uniform grad by gating with a scale vector */
    tensor *scale = tensor_zeros(ctx.params, 1, (int[]){d}, 0);
    float *sp = tensor_data_ptr(scale);
    for (int j = 0; j < d; j++) sp[j] = 1.0f + 2.0f * (float)rand()/(float)RAND_MAX;

    tensor *out   = tensor_layer_norm(ctx.scratch, x, w, NULL, 1e-5f);
    tensor *gated = tensor_mul(ctx.scratch, out, scale);
    tensor *loss  = tensor_sum(ctx.scratch, gated, 0);
    dnn_backward(ctx.scratch, loss);

    float *dx = tensor_grad(x);
    float *dw = tensor_grad(w);
    float *xd = tensor_data_ptr(x);

    /* analytical expectation */
    double sum_x = 0.0;
    for (int j = 0; j < d; j++) sum_x += xd[j];
    float mean = (float)(sum_x / d);

    double sum_v = 0.0;
    for (int j = 0; j < d; j++) { float dd = xd[j] - mean; sum_v += (double)(dd*dd); }
    float rstd = 1.0f / sqrtf((float)(sum_v / d) + 1e-5f);

    float sum_dy = 0.0f, sum_dy_xmu = 0.0f;
    for (int j = 0; j < d; j++) {
        float xmu = xd[j] - mean;
        float dy  = sp[j] * wp[j];
        sum_dy     += dy;
        sum_dy_xmu += dy * xmu;
    }
    float mean_dy     = sum_dy / d;
    float mean_dy_xmu = sum_dy_xmu / d;

    float max_err_dx = 0.0f, max_err_dw = 0.0f;
    for (int j = 0; j < d; j++) {
        float xmu  = xd[j] - mean;
        float gd_j = sp[j];  /* d(loss)/d(out) — NOT multiplied by w */
        float dy   = gd_j * wp[j];  /* d(loss)/d(y) = d(loss)/d(out) * w */
        float y    = (xd[j] - mean) * rstd;
        float exp_dx  = rstd * (dy - mean_dy - xmu * mean_dy_xmu * rstd * rstd);
        float exp_dw  = gd_j * y;   /* d(loss)/d(w) = gd * y */

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
    printf("── 2D batch LayerNorm backward ──\n");
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

    tensor *o2 = tensor_layer_norm(ctx.scratch, x2, w2, NULL, 1e-5f);
    tensor *g2 = tensor_mul(ctx.scratch, o2, s2);
    tensor *l2 = tensor_sum(ctx.scratch, g2, 0);
    dnn_backward(ctx.scratch, l2);

    float *dx2 = tensor_grad(x2);
    float *dw2 = tensor_grad(w2);

    float max_err_dx2 = 0.0f, max_err_dw2 = 0.0f;
    float exp_dw_buf[6]; memset(exp_dw_buf, 0, 6 * sizeof(float));

    for (int b = 0; b < B; b++) {
        double sx = 0.0;
        for (int j = 0; j < d; j++) sx += x2d[b * d + j];
        float m = (float)(sx / d);
        double sv = 0.0;
        for (int j = 0; j < d; j++) { float dd = x2d[b * d + j] - m; sv += (double)(dd*dd); }
        float rs = 1.0f / sqrtf((float)(sv / d) + 1e-5f);

        /* first pass over j to compute sum_dy and sum_dy_xmu */
        float sum_dy = 0.0f, sum_dy_xmu = 0.0f;
        for (int j = 0; j < d; j++) {
            float xmu = x2d[b * d + j] - m;
            float dyj = s2p[j] * w2p[j];
            sum_dy     += dyj;
            sum_dy_xmu += dyj * xmu;
        }
        float md  = sum_dy / (float)d;
        float mdx = sum_dy_xmu / (float)d;

        /* second pass to compute dx and accumulate dγ */
        for (int j = 0; j < d; j++) {
            float dyj = s2p[j] * w2p[j];
            float xmu = x2d[b * d + j] - m;
            float exp_dx = rs * (dyj - md - xmu * mdx * rs * rs);
            float err = fabsf(dx2[b * d + j] - exp_dx);
            if (err > max_err_dx2) max_err_dx2 = err;

            float y = (x2d[b * d + j] - m) * rs;
            exp_dw_buf[j] += s2p[j] * y;  /* dγ = gd * y */
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

    printf(pass ? "ALL PASS\n" : "SOME FAILED\n");
    dnn_ctx_destroy(&ctx);

    return pass ? 0 : 1;
}
