#include "multihead.h"
#include "autograd.h"
#include "autograd_int.h"
#include "pool.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* # of iterations below which OMP overhead dominates. */
#define OMP_MIN_ITERS 8

/* ── Internal helpers ── */

/* Copy data from [B,N,H,d_k] layout (src_t) to [B,H,N,d_k] layout (dst).
 * src_t may be a non-contiguous view (e.g. slice of fused QKV) — uses its
 * strides to index correctly.  dst is always flat contiguous.
 * Parallelized over (batch × heads). */
static void _to_bhnd(const tensor *src_t, int H, float *dst) {
    int B   = src_t->shape[0];
    int N   = src_t->shape[1];
    int D   = src_t->shape[2];  /* H * d_k */
    int d_k = D / H;
    int Nd_k = N * d_k;
    float *src = (float*)src_t->data;
    int off    = src_t->offset;
    int s0     = src_t->strides[0];
    int s1     = src_t->strides[1];
    int s2     = src_t->strides[2];

    #pragma omp parallel for collapse(2) if (B * H >= OMP_MIN_ITERS)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            float *b_dst = dst + b * H * Nd_k;
            for (int n = 0; n < N; n++) {
                float *d = b_dst + h * Nd_k + n * d_k;
                float *s = src + off + b * s0 + n * s1 + h * d_k * s2;
                for (int i = 0; i < d_k; i++)
                    d[i] = s[i * s2];
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
tensor *tensor_split_heads(struct mem_pool *scratch, tensor *t, int H) {
    assert(t->ndim == 3 && "split_heads: input must be 3D [B, N, H*d_k]");

    int B = t->shape[0];
    int N = t->shape[1];
    int D = t->shape[2];  /* H * d_k */
    assert(D % H == 0 && "split_heads: last dim must be divisible by H");
    int d_k = D / H;

    /* Allocate contiguous output in scratch pool */
    tensor *out = tensor_scratch(scratch, 4, (int[]){B, H, N, d_k}, 0);
    float *od = (float*)out->data + out->offset;

    /* Copy with layout transform: [B,N,H,d_k] → [B,H,N,d_k]
     * Uses tensor strides — works for both contiguous inputs and
     * non-contiguous views (e.g. slices of fused QKV). */
    _to_bhnd(t, H, od);

    /* ── Autograd ── */
    int needs_grad = dnn_grad_enabled() && tensor_requires_grad(t);
    if (needs_grad) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = split_heads_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = t;

        fn->n_saved = 1;
        fn->saved_tensors = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        int *saved_H = _mem_pool_alloc(scratch, sizeof(int), NULL);
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
tensor *tensor_merge_heads(struct mem_pool *scratch, tensor *t) {
    assert(t->ndim == 4 && "merge_heads: input must be 4D [B, H, N, d_k]");

    int B   = t->shape[0];
    int H   = t->shape[1];
    int N   = t->shape[2];
    int d_k = t->shape[3];

    /* Allocate contiguous output in scratch pool */
    tensor *out = tensor_scratch(scratch, 3, (int[]){B, N, H * d_k}, 0);
    float *od = (float*)out->data + out->offset;
    float *td = (float*)t->data + t->offset;

    /* Copy with layout transform: [B,H,N,d_k] → [B,N,H,d_k] */
    _to_bnhd(B, H, N, d_k, td, od);

    /* ── Autograd ── */
    int needs_grad = dnn_grad_enabled() && tensor_requires_grad(t);
    if (needs_grad) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = merge_heads_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = t;

        fn->n_saved = 0;
        fn->saved_tensors = NULL;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── Fused QKV split + split heads ──
 *
 * Forward: read contiguous [B,N,3*H*d_k], write 3× contiguous [B,H,N,d_k].
 * One pass, no strided intermediates.
 */

static void _split_qkv_to_bhnd(const tensor *qkv, int H,
                                 float *qd, float *kd, float *vd) {
    int B   = qkv->shape[0];
    int N   = qkv->shape[1];
    int D   = qkv->shape[2];  /* 3 * H * d_k */
    int d_k = D / (3 * H);
    float *src = (float*)qkv->data + qkv->offset;

    #pragma omp parallel for collapse(2) if (B * H >= OMP_MIN_ITERS)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            float *q_row = qd + (b * H + h) * N * d_k;
            float *k_row = kd + (b * H + h) * N * d_k;
            float *v_row = vd + (b * H + h) * N * d_k;
            for (int n = 0; n < N; n++) {
                float *s = src + b * N * D + n * D + h * d_k;
                for (int i = 0; i < d_k; i++) {
                    q_row[n * d_k + i] = s[i];
                    k_row[n * d_k + i] = s[H * d_k + i];
                    v_row[n * d_k + i] = s[2 * H * d_k + i];
                }
            }
        }
}

/* Backward for one output (Q, K, or V) of split_qkv_heads.
 * Adds (does not zero) gradient to the fused qkv tensor.
 * saved_idx = 0→Q, 1→K, 2→V selects which third of fused data. */
static void split_qkv_heads_backward(grad_fn *fn, tensor *grad_output) {
    tensor *qkv = fn->inputs[0];
    int B   = qkv->shape[0];
    int N   = qkv->shape[1];
    int D   = qkv->shape[2];  /* 3 * H * d_k */
    int saved_idx = *(int*)fn->saved_tensors[0];
    float *gd = (float*)grad_output->data + grad_output->offset;
    float *ig = _grad_ensure(qkv);
    if (!ig) return;

    /* grad_output is [B, H, N, d_k] (contiguous), saved_idx selects offset
     * into the fused qkv gradient [B, N, 3*H*d_k]. */
    int H_   = grad_output->shape[1];
    int d_k  = grad_output->shape[3];
    int Hd_k = H_ * d_k;

    #pragma omp parallel for collapse(2) if (B * H_ >= OMP_MIN_ITERS)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H_; h++) {
            float *g_row = gd + (b * H_ + h) * N * d_k;
            int qkv_off = qkv->offset + b * N * D + h * d_k + saved_idx * Hd_k;
            for (int n = 0; n < N; n++) {
                float *ig_off = ig + qkv_off + n * D;
                float *gd_off = g_row + n * d_k;
                for (int i = 0; i < d_k; i++)
                    ig_off[i] += gd_off[i];
            }
        }
}

/* ── Public API ── */

void tensor_split_qkv_heads(struct mem_pool *scratch,
                            tensor *qkv, int H,
                            tensor **Qh_out, tensor **Kh_out, tensor **Vh_out) {
    assert(qkv->ndim == 3 && "split_qkv_heads: qkv must be 3D [B,N,3*H*d_k]");
    assert(qkv->contiguous && "split_qkv_heads: qkv must be contiguous");
    int B   = qkv->shape[0];
    int N   = qkv->shape[1];
    int D   = qkv->shape[2];  /* 3 * H * d_k */
    assert(D % (3 * H) == 0 && "split_qkv_heads: D must be 3*H*d_k");
    int d_k = D / (3 * H);

    /* Allocate three contiguous output tensors */
    int shape4[] = {B, H, N, d_k};
    tensor *Qh = tensor_scratch(scratch, 4, shape4, 0);
    tensor *Kh = tensor_scratch(scratch, 4, shape4, 0);
    tensor *Vh = tensor_scratch(scratch, 4, shape4, 0);

    /* Fused layout transform: contiguous read from qkv, contiguous writes */
    _split_qkv_to_bhnd(qkv, H,
                        (float*)Qh->data + Qh->offset,
                        (float*)Kh->data + Kh->offset,
                        (float*)Vh->data + Vh->offset);

    /* ── Autograd — wire three separate grad_fn instances, one per output.
     * Each backward adds to the qkv gradient (no zero, so they accumulate). */
    int needs_grad = dnn_grad_enabled() && tensor_requires_grad(qkv);
    if (needs_grad) {
        tensor *outs[3] = {Qh, Kh, Vh};
        int saved_ids[3] = {0, 1, 2};
        for (int i = 0; i < 3; i++) {
            grad_fn *fn = _grad_fn_create(scratch);
            fn->backward = split_qkv_heads_backward;
            fn->n_inputs = 1;
            fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
            fn->inputs[0] = qkv;
            fn->n_saved = 1;
            fn->saved_tensors = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
            int *p_idx = _mem_pool_alloc(scratch, sizeof(int), NULL);
            *p_idx = saved_ids[i];
            fn->saved_tensors[0] = (tensor*)p_idx;
            outs[i]->requires_grad = 1;
            outs[i]->grad_fn = fn;
        }
    }

    *Qh_out = Qh;
    *Kh_out = Kh;
    *Vh_out = Vh;
}
