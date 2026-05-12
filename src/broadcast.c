#include "broadcast.h"
#include <assert.h>

/* compute broadcast shape, return out ndim (0 = incompatible) */
int _bcast_ndim(int ndim_a, const int *shape_a,
                int ndim_b, const int *shape_b,
                int *out_shape) {
    int max = ndim_a > ndim_b ? ndim_a : ndim_b;
    for (int i = 0; i < max; i++) {
        int da = i < ndim_a ? shape_a[ndim_a - 1 - i] : 1;
        int db = i < ndim_b ? shape_b[ndim_b - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) return 0;
        out_shape[max - 1 - i] = da > db ? da : db;
    }
    return max;
}

/* strided element offset for tensor at logical flat index (row-major) */
int _flat_off(const tensor *t, int flat) {
    int off = t->offset;
    for (int d = t->ndim - 1; d >= 0; d--) {
        int c = flat % t->shape[d];
        flat /= t->shape[d];
        off += c * t->strides[d];
    }
    return off;
}

/* strided offset in input tensor given an output coord (broadcast-aware) */
int _bcast_off(const tensor *t, int out_ndim, const int *coord) {
    int off = t->offset;
    int lead = out_ndim - t->ndim;
    for (int d = 0; d < t->ndim; d++) {
        int c = t->shape[d] == 1 ? 0 : coord[lead + d];
        off += c * t->strides[d];
    }
    return off;
}

/* numel from raw shape (no tensor object) */
int _numel(int ndim, const int *shape) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}
