#include "multihead.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
#include "tensor_int.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* # of iterations below which OMP overhead dominates. */
#define OMP_MIN_ITERS 8

/* ── Internal helpers ── */

/* Copy data from [B,N,H,d_k] layout (src) to [B,H,N,d_k] layout (dst).
 * Both dst and src are contiguous flat arrays.
 * Parallelized over (batch × heads) — each thread copies a subset of
 * sequence positions for its assigned (b,h) pair. */
static void _to_bhnd(int B, int N, int H, int d_k,
                      const float *src, float *dst) {
    int Hd_k = H * d_k;
    int Nd_k = N * d_k;

    #pragma omp parallel for collapse(2) if (B * H >= OMP_MIN_ITERS)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            const float *b_src = src + b * N * Hd_k;
            float *b_dst = dst + b * H * Nd_k;
            for (int n = 0; n < N; n++) {
                const float *s = b_src + n * Hd_k + h * d_k;
                float *d = b_dst + h * Nd_k + n * d_k;
                for (int i = 0; i < d_k; i++)
                    d[i] = s[i];
            }
        }
}

/* Copy data from [B,H,N,d_k] layout (src) to [B,N,H,d_k] layout (dst).
 * Parallelized over (batch × heads). */
static void _to_bnhd(int B, int H, int N, int d_k,
                      const float *src, float *dst) {
    int Nd_k = N * d_k;
    int Hd_k = H * d_k;

    #pragma omp parallel for collapse(2) if (B * H >= OMP_MIN_ITERS)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            const float *b_src = src + b * H * Nd_k;
            float *b_dst = dst + b * N * Hd_k;
            for (int n = 0; n < N; n++) {
                const float *s = b_src + h * Nd_k + n * d_k;
                float *d = b_dst + n * Hd_k + h * d_k;
                for (int i = 0; i < d_k; i++)
                    d[i] = s[i];
            }
        }
}

/* ── split_heads_backward ──
 *
 *   grad_output: [B, H, N, d_k]  (contiguous)
 *   grad flows to: input (3D [B, N, H*d_k])
 *   gradient transform: invert the split → merge transform
 */
static void split_heads_backward(grad_fn *fn, tensor *grad_output) {
    tensor *input = fn->inputs[0];
    int B = input->shape[0];
    int N = input->shape[1];
    int D = input->shape[2];
    int saved_H = *(int*)fn->saved_tensors[0];

    float *gd = (float*)grad_output->data + grad_output->offset;
    float *ig = _grad_ensure(input);
    if (!ig) return;

    /* gradient for input: merge heads (inverse of split)
     * grad_input[b,n,h,d] = grad_output[b,h,n,d]
     * Write to input's grad buffer (which is same shape as input).
     * NOTE: input may be a view (e.g., slice of fused QKV).
     * _grad_ensure returns root->grad; use input->offset to target
     * the correct region of the root's grad buffer. */
    int numel = tensor_numel(input);
    memset(ig + input->offset, 0, numel * sizeof(float));

    int d_k = D / saved_H;
    /* Collapse over (b, h): each thread handles one (batch, head) pair */
    #pragma omp parallel for collapse(2) if (B * saved_H >= OMP_MIN_ITERS)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < saved_H; h++) {
            float *ig_off = ig + input->offset + b * N * D + h * d_k;
            float *gd_off = gd + ((b * saved_H + h) * N) * d_k;
            for (int n = 0; n < N; n++) {
                float *ig_row = ig_off + n * D;
                float *gd_row = gd_off + n * d_k;
                for (int d = 0; d < d_k; d++)
                    ig_row[d] += gd_row[d];
            }
        }
}

/* ── tensor_split_heads forward ──
 *
 *   Input:  [B, N, H*d_k], contiguous 3D
 *   Output: [B, H, N, d_k], contiguous 4D (new alloc)
 */
tensor *tensor_split_heads(tensor *t, int H) {
    assert(t->ndim == 3 && "split_heads: input must be 3D [B, N, H*d_k]");

    int B = t->shape[0];
    int N = t->shape[1];
    int D = t->shape[2];  /* H * d_k */
    assert(D % H == 0 && "split_heads: last dim must be divisible by H");
    int d_k = D / H;

    /* Allocate contiguous output in scratch pool */
    tensor *out = _tensor_scratch_create(4, (int[]){B, H, N, d_k}, 0);
    float *od = (float*)out->data + out->offset;
    float *td = (float*)t->data + t->offset;

    /* Copy with layout transform: [B,N,H,d_k] → [B,H,N,d_k] */
    _to_bhnd(B, N, H, d_k, td, od);

    /* ── Autograd ── */
    int needs_grad = dnn_grad_enabled() && tensor_requires_grad(t);
    if (needs_grad) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = split_heads_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = t;

        fn->n_saved = 1;
        fn->saved_tensors = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        int *saved_H = mem_scratch_alloc(sizeof(int), NULL);
        *saved_H = H;
        fn->saved_tensors[0] = (tensor*)saved_H;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── merge_heads_backward ──
 *
 *   grad_output: [B, N, H*d_k]  (contiguous)
 *   grad flows to: input (4D [B, H, N, d_k])
 *   gradient transform: invert the merge → split transform
 */
static void merge_heads_backward(grad_fn *fn, tensor *grad_output) {
    tensor *input = fn->inputs[0];
    int B = input->shape[0];
    int H = input->shape[1];
    int N = input->shape[2];
    int d_k = input->shape[3];

    float *gd = (float*)grad_output->data + grad_output->offset;
    float *ig = _grad_ensure(input);
    if (!ig) return;

    /* gradient for input: split heads (inverse of merge)
     * grad_input[b,h,n,d] = grad_output[b,n,h,d]
     * Write to input's grad buffer.
     * NOTE: input may be a view — use input->offset for root grad. */
    int numel = tensor_numel(input);
    memset(ig + input->offset, 0, numel * sizeof(float));

    /* Collapse over (b, h): each thread handles one (batch, head) pair */
    #pragma omp parallel for collapse(2) if (B * H >= OMP_MIN_ITERS)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            float *ig_off = ig + input->offset + ((b * H + h) * N) * d_k;
            float *gd_off = gd + b * N * H * d_k + h * d_k;
            for (int n = 0; n < N; n++) {
                float *ig_row = ig_off + n * d_k;
                float *gd_row = gd_off + n * H * d_k;
                for (int d = 0; d < d_k; d++)
                    ig_row[d] += gd_row[d];
            }
        }
}

/* ── tensor_merge_heads forward ──
 *
 *   Input:  [B, H, N, d_k], contiguous 4D
 *   Output: [B, N, H*d_k], contiguous 3D (new alloc)
 */
tensor *tensor_merge_heads(tensor *t) {
    assert(t->ndim == 4 && "merge_heads: input must be 4D [B, H, N, d_k]");

    int B   = t->shape[0];
    int H   = t->shape[1];
    int N   = t->shape[2];
    int d_k = t->shape[3];

    /* Allocate contiguous output in scratch pool */
    tensor *out = _tensor_scratch_create(3, (int[]){B, N, H * d_k}, 0);
    float *od = (float*)out->data + out->offset;
    float *td = (float*)t->data + t->offset;

    /* Copy with layout transform: [B,H,N,d_k] → [B,N,H,d_k] */
    _to_bnhd(B, H, N, d_k, td, od);

    /* ── Autograd ── */
    int needs_grad = dnn_grad_enabled() && tensor_requires_grad(t);
    if (needs_grad) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = merge_heads_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = t;

        fn->n_saved = 0;
        fn->saved_tensors = NULL;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}
