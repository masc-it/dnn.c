#ifndef DNN_BROADCAST_H
#define DNN_BROADCAST_H

#include "tensor.h"

/* ── Broadcast shape computation ── */
/* returns out ndim, 0 = incompatible */
int _bcast_ndim(int ndim_a, const int *shape_a,
                int ndim_b, const int *shape_b,
                int *out_shape);

/* ── Strided offset helpers ── */
int _flat_off(const tensor *t, int flat);
int _bcast_off(const tensor *t, int out_ndim, const int *coord);

/* ── Element count from raw shape ── */
int _numel(int ndim, const int *shape);

#endif /* DNN_BROADCAST_H */
