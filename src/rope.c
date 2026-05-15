#include "rope.h"
#include "autograd.h"
#include "pool.h"
#include "autograd_int.h"
#include "broadcast.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── Internal: in-place RoPE forward ──
 *
 * Applies 2D rotation to each pair (2k, 2k+1) along the last dimension
 * of x.  x is contiguous with shape [..., N, d] where d is even.
 * freqs_cos/sin are contiguous [N, d/2].
 *
 * The number of outer dims (batch × heads) is num_outer = numel / (N * d).
 */

static void _rope_apply(float *x_data, int offset, int N, int d, int num_outer,
                         const float *fc, const float *fs) {
    int d_half = d / 2;
    for (int o = 0; o < num_outer; o++) {
        for (int n = 0; n < N; n++) {
            float *row  = x_data + offset + (o * N + n) * d;
            const float *c = fc + n * d_half;
            const float *s = fs + n * d_half;
            for (int k = 0; k < d_half; k++) {
                float e = row[2 * k];
                float o_val = row[2 * k + 1];
                row[2 * k]     = e * c[k] - o_val * s[k];
                row[2 * k + 1] = e * s[k] + o_val * c[k];
            }
        }
    }
}

/* ── Internal: inverse RoPE applied to gradient ──
 *
 * Applies the transpose rotation (inverse) to each pair:
 *   grad[2k]   += g[2k] * cos + g[2k+1] * sin
 *   grad[2k+1] += -g[2k] * sin + g[2k+1] * cos
 *
 * where g[2k], g[2k+1] are the incoming gradient values.
 */

static void _rope_apply_grad(float *grad, int offset, int N, int d, int num_outer,
                              const float *fc, const float *fs,
                              const float *g_data) {
    int d_half = d / 2;
    for (int o = 0; o < num_outer; o++) {
        for (int n = 0; n < N; n++) {
            const float *g_row = g_data + (o * N + n) * d;
            float *grad_row   = grad + offset + (o * N + n) * d;
            const float *c = fc + n * d_half;
            const float *s = fs + n * d_half;
            for (int k = 0; k < d_half; k++) {
                float ge = g_row[2 * k];
                float go = g_row[2 * k + 1];
                grad_row[2 * k]     += ge * c[k] + go * s[k];
                grad_row[2 * k + 1] += -ge * s[k] + go * c[k];
            }
        }
    }
}

/* ── rope_backward ── */

static void rope_backward(grad_fn *fn, tensor *grad_output) {
    tensor *x         = fn->inputs[0];
    tensor *freqs_cos = fn->saved_tensors[0];
    tensor *freqs_sin = fn->saved_tensors[1];

    if (!(x->grad_fn || x->requires_grad)) return;

    int ndim    = x->ndim;
    int d       = x->shape[ndim - 1];
    int N       = x->shape[ndim - 2];
    int numel   = tensor_numel(x);
    int num_outer = numel / (N * d);

    float *grad = _grad_ensure(x);
    float *g_data = (float*)grad_output->data;  /* grad_output is contiguous wrapper */
    float *fc = (float*)freqs_cos->data + freqs_cos->offset;
    float *fs = (float*)freqs_sin->data + freqs_sin->offset;

    _rope_apply_grad(grad, x->offset, N, d, num_outer, fc, fs, g_data + grad_output->offset);
}

/* ── tensor_rope ── */

tensor *tensor_rope(struct mem_pool *scratch, tensor *x, const tensor *freqs_cos, const tensor *freqs_sin) {
    assert(x);
    assert(freqs_cos && freqs_sin);
    assert(tensor_is_contiguous(x) && "tensor_rope: x must be contiguous");
    assert(tensor_is_contiguous(freqs_cos) && "tensor_rope: freqs_cos must be contiguous");
    assert(tensor_is_contiguous(freqs_sin) && "tensor_rope: freqs_sin must be contiguous");
    assert(x->ndim >= 2 && "tensor_rope: x must have at least 2 dims");
    int ndim = x->ndim;
    int d    = x->shape[ndim - 1];
    int d_half  = d / 2;
    assert(d % 2 == 0 && "tensor_rope: last dim must be even");
    int N    = x->shape[ndim - 2];
    assert(freqs_cos->ndim == 2 && freqs_cos->shape[0] == N && freqs_cos->shape[1] == d_half);
    assert(freqs_sin->ndim == 2 && freqs_sin->shape[0] == N && freqs_sin->shape[1] == d_half);

    int numel   = tensor_numel(x);
    int num_outer = numel / (N * d);

    float *xd = (float*)x->data;
    float *fc = (float*)freqs_cos->data + freqs_cos->offset;
    float *fs = (float*)freqs_sin->data + freqs_sin->offset;

    /* Apply rotation in-place */
    _rope_apply(xd, x->offset, N, d, num_outer, fc, fs);

    /* Create lightweight view tensor sharing x's data buffer.
     * Copied from x, but without a new data allocation. */
    tensor *out = _mem_pool_alloc(scratch, sizeof(tensor), x);
    out->data        = x->data;
    out->offset      = x->offset;
    out->pool        = x->pool;    /* inherit pool from input (for _grad_ensure) */
    out->parent      = x;          /* view: parent chain for _grad_ensure */
    out->grad_fn     = NULL;
    out->grad        = NULL;

    /* Wire autograd */
    if (dnn_grad_enabled() && tensor_requires_grad(x)) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = rope_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = x;
        fn->n_saved = 2;
        fn->saved_tensors = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)freqs_cos;
        fn->saved_tensors[1] = (tensor*)freqs_sin;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── tensor_rope_freqs_init ── */

void tensor_rope_freqs_init(struct mem_pool *params, tensor **freqs_cos, tensor **freqs_sin,
                             int dim, int max_seq_len, float base) {
    assert(dim > 0 && dim % 2 == 0 && "dim must be positive and even");
    assert(max_seq_len > 0);
    if (base == 0.0f) base = 10000.0f;

    int d_half = dim / 2;
    int shape[2] = {max_seq_len, d_half};

    *freqs_cos = tensor_zeros(params, 2, shape, 0);
    *freqs_sin = tensor_zeros(params, 2, shape, 0);

    float *cos_data = (float*)(*freqs_cos)->data;
    float *sin_data = (float*)(*freqs_sin)->data;

    for (int k = 0; k < d_half; k++) {
        float theta = powf(base, -2.0f * k / (float)dim);
        for (int n = 0; n < max_seq_len; n++) {
            float angle = (float)n * theta;
            cos_data[n * d_half + k] = cosf(angle);
            sin_data[n * d_half + k] = sinf(angle);
        }
    }
}
