#include "rope.h"
#include "pool.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define EPS 1e-5f

/* helper: confirm float close */
static int close(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

/* ── Test: basic shape and contiguity ── */

static void test_rope_freqs_shape(void) {
    printf("  test_rope_freqs_shape... ");
    int d = 64;
    tensor *freqs = tensor_rope_freqs(d, 10000.0f);
    assert(freqs != NULL);
    assert(tensor_ndim(freqs) == 1);
    assert(tensor_shape(freqs, 0) == d / 2);
    assert(tensor_is_contiguous(freqs));
    assert(!tensor_requires_grad(freqs));
    printf("OK\n");
}

/* ── Test: first value theta_0 = base^{-0/d} = 1.0 ── */

static void test_rope_freqs_first(void) {
    printf("  test_rope_freqs_first... ");
    tensor *freqs = tensor_rope_freqs(64, 10000.0f);
    float *fp = tensor_data_ptr(freqs);
    /* theta_0 = 10000^(-0/64) = 1.0 */
    assert(close(fp[0], 1.0f, EPS));
    printf("OK\n");
}

/* ── Test: last value for standard d=64, base=10000 ──
 *
 * theta_{31} = 10000^{-62/64} = 10000^{-0.96875}
 *            = exp(-0.96875 * ln(10000))
 *            = exp(-0.96875 * 9.210340371976184)
 *            = exp(-8.9227662765)
 *            ≈ 0.000133
 */

static void test_rope_freqs_last(void) {
    printf("  test_rope_freqs_last... ");
    tensor *freqs = tensor_rope_freqs(64, 10000.0f);
    float *fp = tensor_data_ptr(freqs);
    /* theta_{31} for d=64, k=31 */
    float expected = expf(-2.0f * 31 * logf(10000.0f) / 64.0f);
    assert(close(fp[31], expected, 1e-4f));
    printf("OK\n");
}

/* ── Test: all values computed correctly ── */

static void test_rope_freqs_all(void) {
    printf("  test_rope_freqs_all... ");
    int d = 128;
    float base = 10000.0f;
    tensor *freqs = tensor_rope_freqs(d, base);
    float *fp = tensor_data_ptr(freqs);
    int half = d / 2;

    for (int k = 0; k < half; k++) {
        float expected = powf(base, -2.0f * k / (float)d);
        if (!close(fp[k], expected, 1e-5f)) {
            printf("  FAIL at k=%d: got %.10f, expected %.10f\n",
                   k, fp[k], expected);
            assert(0);
        }
    }
    printf("OK\n");
}

/* ── Test: different base values ── */

static void test_rope_freqs_base(void) {
    printf("  test_rope_freqs_base... ");
    int d = 32;
    float bases[] = {10000.0f, 50000.0f, 1000.0f};
    int nb = sizeof(bases) / sizeof(bases[0]);

    for (int bi = 0; bi < nb; bi++) {
        float base = bases[bi];
        tensor *freqs = tensor_rope_freqs(d, base);
        float *fp = tensor_data_ptr(freqs);
        int half = d / 2;

        for (int k = 0; k < half; k++) {
            float expected = powf(base, -2.0f * k / (float)d);
            if (!close(fp[k], expected, 1e-5f)) {
                printf("  FAIL at base=%.0f, k=%d: got %.10f, expected %.10f\n",
                       base, k, fp[k], expected);
                assert(0);
            }
        }
    }
    printf("OK\n");
}

/* ── Test: monotonic decreasing ──
 *
 * theta_k = base^{-2k/d}  decreases monotonically as k increases
 * (since base > 1, exponent becomes more negative).
 */

static void test_rope_freqs_monotonic(void) {
    printf("  test_rope_freqs_monotonic... ");
    int d = 64;
    tensor *freqs = tensor_rope_freqs(d, 10000.0f);
    float *fp = tensor_data_ptr(freqs);
    int half = d / 2;

    for (int k = 1; k < half; k++) {
        if (fp[k] >= fp[k-1]) {
            printf("  FAIL: theta[%d]=%.10f >= theta[%d]=%.10f (should decrease)\n",
                   k, fp[k], k-1, fp[k-1]);
            assert(0);
        }
    }
    printf("OK\n");
}

/* ── Test: small head dim (d=2) ── */

static void test_rope_freqs_min_d(void) {
    printf("  test_rope_freqs_min_d... ");
    tensor *freqs = tensor_rope_freqs(2, 10000.0f);
    assert(tensor_shape(freqs, 0) == 1);
    float *fp = tensor_data_ptr(freqs);
    /* theta_0 = 10000^0 = 1.0 */
    assert(close(fp[0], 1.0f, EPS));
    printf("OK\n");
}

/* ── Test: multi-res (different d values produce different lengths) ── */

static void test_rope_freqs_dims(void) {
    printf("  test_rope_freqs_dims... ");
    int ds[] = {2, 16, 64, 128, 256};

    for (int di = 0; di < 5; di++) {
        int d = ds[di];
        tensor *freqs = tensor_rope_freqs(d, 10000.0f);
        assert(tensor_shape(freqs, 0) == d / 2);
        /* spot check k=0 always 1.0 */
        assert(close(tensor_data_ptr(freqs)[0], 1.0f, EPS));
    }
    printf("OK\n");
}

int main(void) {
    printf("test_rope:\n");

    mem_pool params  = mem_pool_create(64 * 1024);
    mem_pool scratch = mem_pool_create(64 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    test_rope_freqs_shape();
    test_rope_freqs_first();
    test_rope_freqs_last();
    test_rope_freqs_all();
    test_rope_freqs_base();
    test_rope_freqs_monotonic();
    test_rope_freqs_min_d();
    test_rope_freqs_dims();

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);

    printf("    PASS\n");
    return 0;
}
