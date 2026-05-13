#ifndef DNN_BROADCAST_H
#define DNN_BROADCAST_H

#include "tensor.h"

/* ── Broadcast shape computation ── */
/* returns out ndim, 0 = incompatible */
int _bcast_ndim(int ndim_a, const int *shape_a,
                int ndim_b, const int *shape_b,
                int *out_shape);

/* ── Strided offset helpers (inline with contiguity fast path) ── */

static inline int _flat_off(const tensor *t, int flat) {
    if (t->contiguous)
        return t->offset + flat;
    int off = t->offset;
    for (int d = t->ndim - 1; d >= 0; d--) {
        int c = flat % t->shape[d];
        flat /= t->shape[d];
        off += c * t->strides[d];
    }
    return off;
}

static inline int _bcast_off(const tensor *t, int out_ndim, const int *coord) {
    int off = t->offset;
    int lead = out_ndim - t->ndim;
    if (t->contiguous && t->ndim == out_ndim) {
        /* fast path: contiguous, same ndim, no broadcasting.
         * Check that no dim has shape-1-broadcasting. */
        int i;
        for (i = 0; i < t->ndim; i++)
            if (t->shape[i] == 1) break;  /* broadcast dim */
        if (i == t->ndim) {
            /* no broadcasting → flat index */
            int flat = 0;
            for (int d = 0; d < t->ndim; d++)
                flat = flat * t->shape[d] + coord[d];
            return t->offset + flat;
        }
    }
    for (int d = 0; d < t->ndim; d++) {
        int c = t->shape[d] == 1 ? 0 : coord[lead + d];
        off += c * t->strides[d];
    }
    return off;
}

/* ── Element count from raw shape ── */
int _numel(int ndim, const int *shape);

/* ── Fast-path helpers: same contiguous shape (no broadcast) ── */

/* Both tensors have the same shape and are contiguous → flat indexing */
static inline int _same_contiguous(const tensor *a, const tensor *b) {
    if (a->ndim != b->ndim) return 0;
    if (!a->contiguous || !b->contiguous) return 0;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

/* Tensor has same shape as grad_output and is contiguous → flat grad path */
static inline int _grad_contiguous(const tensor *t, const tensor *grad) {
    if (t->ndim != grad->ndim) return 0;
    if (!t->contiguous) return 0;
    for (int i = 0; i < t->ndim; i++)
        if (t->shape[i] != grad->shape[i]) return 0;
    return 1;
}

#endif /* DNN_BROADCAST_H */
