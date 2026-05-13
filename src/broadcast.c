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

/* numel from raw shape (no tensor object) */
int _numel(int ndim, const int *shape) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}
