#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_activation_int.h"
#include "simd.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

/* ── relu_backward ── */

static void relu_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    int n = _numel(a->ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *ad = (float*)a->data;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        if (tensor_is_contiguous(a) && tensor_is_contiguous(grad_output)) {
            float *ap = ad + a->offset;
            simd_relu_bwd(ag + a->offset, ap, g_data, n);
        } else {
            for (int i = 0; i < n; i++) {
                int off = _flat_off(a, i);
                if (ad[off] > 0.0f)
                    ag[off] += g_data[i];
            }
        }
    }
}

tensor *tensor_relu(const tensor *t) {
    assert(t);

    tensor *out = _tensor_scratch_create(t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    if (tensor_is_contiguous(t)) {
        float *tp = td + t->offset;
        simd_relu_fwd(od, tp, numel);
    } else {
        for (int i = 0; i < numel; i++) {
            int off = _flat_off(t, i);
            float v = td[off];
            od[out->offset + i] = v > 0.0f ? v : 0.0f;
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = relu_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── sigmoid_backward ── */

static void sigmoid_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *saved_out = fn->saved_tensors[0];
    int n = _numel(a->ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *sd = (float*)saved_out->data;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        if (tensor_is_contiguous(a) && tensor_is_contiguous(grad_output)) {
            simd_sigmoid_bwd(ag + a->offset, sd, g_data, n);
        } else {
            for (int i = 0; i < n; i++) {
                int off = _flat_off(a, i);
                float sig = sd[i];
                ag[off] += sig * (1.0f - sig) * g_data[i];
            }
        }
    }
}

tensor *tensor_sigmoid(const tensor *t) {
    assert(t);

    tensor *out = _tensor_scratch_create(t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    if (tensor_is_contiguous(t)) {
        float *tp = td + t->offset;
        simd_sigmoid_fwd(od, tp, numel);
    } else {
        for (int i = 0; i < numel; i++) {
            int off = _flat_off(t, i);
            od[out->offset + i] = 1.0f / (1.0f + expf(-td[off]));
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = sigmoid_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 1;
        fn->saved_tensors = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = out;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── silu_backward ──
 *
 *   silu(x) = x * sigmoid(x)
 *   silu'(x) = sigmoid(x) * (1 + x - x*sigmoid(x))
 *
 * Recomputes sigmoid(x) from saved input — no intermediate tensor needed.
 */

static void silu_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    int n = _numel(a->ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *ad = (float*)a->data;

    if (a->grad_fn || a->requires_grad) {
        float *ag = _grad_ensure(a);
        if (tensor_is_contiguous(a) && tensor_is_contiguous(grad_output)) {
            simd_silu_bwd(ag + a->offset, ad + a->offset, g_data, n);
        } else {
            for (int i = 0; i < n; i++) {
                int off = _flat_off(a, i);
                float x = ad[off];
                float sig = 1.0f / (1.0f + expf(-x));
                ag[off] += sig * (1.0f + x - x * sig) * g_data[i];
            }
        }
    }
}

/* ── silu (Swish) ──
 *
 *   silu(x) = x * sigmoid(x)
 *
 * Fused single-pass: computes x*sigmoid(x) in one loop over input,
 * no intermediate sigmoid tensor allocated.  Autograd wired via
 * silu_backward (no saved tensors needed).
 */

tensor *tensor_silu(const tensor *t) {
    assert(t);

    tensor *out = _tensor_scratch_create(t->ndim, t->shape, 0);
    int numel = _numel(t->ndim, t->shape);
    float *od = (float*)out->data;
    float *td = (float*)t->data;

    if (tensor_is_contiguous(t)) {
        float *tp = td + t->offset;
        simd_silu_fwd(od, tp, numel);
    } else {
        for (int i = 0; i < numel; i++) {
            int off = _flat_off(t, i);
            float x = td[off];
            od[out->offset + i] = x / (1.0f + expf(-x));
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = silu_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── swiglu_backward ──
 *
 *   out = SiLU(gate) * up
 *   d_gate = d_out * SiLU'(gate) * up
 *   d_up   = d_out * SiLU(gate)
 *
 * Gate and up shapes match the output shape (broadcast resolved in forward).
 * Recomputes sigmoid(gate) from saved gate — no intermediate saved tensors
 * aside from gate pointer (used in both d_gate and d_up paths).
 */

static void swiglu_backward(grad_fn *fn, tensor *grad_output) {
    tensor *gate = fn->inputs[0];
    tensor *up   = fn->inputs[1];

    int out_numel = tensor_numel(grad_output);
    float *g_data = tensor_data_ptr(grad_output);
    float *gd = (float*)gate->data;
    float *ud = (float*)up->data;

    int need_gate = (gate->grad_fn || gate->requires_grad);
    int need_up   = (up->grad_fn   || up->requires_grad);

    float *ag = need_gate ? _grad_ensure(gate) : NULL;
    float *bg = need_up   ? _grad_ensure(up)   : NULL;

    /* Fast path: both same contiguous shape as grad_output */
    if (_grad_contiguous(gate, grad_output) && _grad_contiguous(up, grad_output)) {
        simd_swiglu_bwd(
            ag ? ag + gate->offset : NULL,
            bg ? bg + up->offset   : NULL,
            gd + gate->offset,
            ud + up->offset,
            g_data, out_numel);
        return;
    }

    /* General: precompute offsets then scatter */
    int *gate_offs = need_gate ? mem_scratch_alloc(out_numel * sizeof(int), NULL) : NULL;
    int *up_offs   = need_up   ? mem_scratch_alloc(out_numel * sizeof(int), NULL) : NULL;
    int out_ndim = grad_output->ndim;
    for (int i = 0; i < out_numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            coord[d] = r % grad_output->shape[d];
            r /= grad_output->shape[d];
        }
        if (gate_offs) gate_offs[i] = _bcast_off(gate, out_ndim, coord);
        if (up_offs)   up_offs[i]   = _bcast_off(up,   out_ndim, coord);
    }

    if (ag && gate_offs) {
        for (int i = 0; i < out_numel; i++) {
            float g_val = gd[gate_offs[i]];
            float sig = 1.0f / (1.0f + expf(-g_val));
            float silu_deriv = sig * (1.0f + g_val - g_val * sig);
            ag[gate_offs[i]] += g_data[i] * silu_deriv * ud[up_offs[i]];
        }
    }
    if (bg && up_offs) {
        for (int i = 0; i < out_numel; i++) {
            float g_val = gd[gate_offs[i]];
            float sig = 1.0f / (1.0f + expf(-g_val));
            float silu = g_val * sig;
            bg[up_offs[i]] += g_data[i] * silu;
        }
    }
}

/* ── tensor_swiglu ──
 *
 *   out = SiLU(gate) ⊗ up
 *
 * Fused single-pass gated activation: computes silu(gate) * up in one
 * loop over the broadcast output.  No intermediate SiLU tensor allocated.
 * Both inputs must have compatible shapes (NumPy-style broadcast).
 *
 * Autograd: saves neither — gate and up are the inputs, recompute
 * sigmoid(gate) in backward from the saved input pointers.
 */

tensor *tensor_swiglu(const tensor *gate, const tensor *up) {
    assert(gate && up);

    int ndim_out;
    int shape_out[DNN_MAX_DIMS];
    ndim_out = _bcast_ndim(gate->ndim, gate->shape, up->ndim, up->shape, shape_out);
    assert(ndim_out > 0 && "tensor_swiglu: incompatible shapes");

    tensor *out = _tensor_scratch_create(ndim_out, shape_out, 0);
    int numel = _numel(ndim_out, shape_out);
    float *od = (float*)out->data;
    float *gd = (float*)gate->data;
    float *ud = (float*)up->data;

    if (_same_contiguous(gate, up)) {
        /* Fast path: same contiguous shapes, no broadcasting */
        float *gp = gd + gate->offset;
        float *up_p = ud + up->offset;
        simd_swiglu_fwd(od, gp, up_p, numel);
    } else {
        /* General broadcast path */
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim_out - 1; d >= 0; d--) {
                coord[d] = r % shape_out[d];
                r /= shape_out[d];
            }
            int g_off = _bcast_off(gate, ndim_out, coord);
            int u_off = _bcast_off(up,   ndim_out, coord);
            float g_val = gd[g_off];
            od[out->offset + i] = (g_val / (1.0f + expf(-g_val))) * ud[u_off];
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() &&
        (tensor_requires_grad(gate) || tensor_requires_grad(up))) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = swiglu_backward;
        fn->n_inputs = 2;
        fn->inputs = mem_scratch_alloc(2 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)gate;
        fn->inputs[1] = (tensor*)up;
        fn->n_saved = 0;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── softmax_backward ── */

static void softmax_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *saved_out = fn->saved_tensors[0];
    int dim = *(int*)fn->saved_tensors[1];

    int ndim = a->ndim;
    int numel = _numel(ndim, a->shape);
    float *g_data = (float*)grad_output->data;
    float *sm_data = (float*)saved_out->data;
    int dim_size = a->shape[dim];
    int n_slices = numel / dim_size;

    if (!(a->grad_fn || a->requires_grad)) return;

    float *ag = _grad_ensure(a);

    /* ── 2D fast path (N, C), dim=1, contiguous ── */
    if (ndim == 2 && dim == 1
        && tensor_is_contiguous(a)
        && tensor_is_contiguous(saved_out)
        && tensor_is_contiguous(grad_output)) {
        int N = a->shape[0], C = a->shape[1];
        for (int n = 0; n < N; n++) {
            float *sm_row = sm_data + n * C;
            float *g_row  = g_data + n * C;
            float *ag_row = ag + n * C;

            /* dot = sum(sm * g) over this row */
            float dot = 0.0f;
#if DNN_HAVE_NEON
            float32x4_t vdot = vdupq_n_f32(0.0f);
            int c = 0;
            for (; c + 4 <= C; c += 4)
                vdot = vfmaq_f32(vdot, vld1q_f32(sm_row + c),
                                          vld1q_f32(g_row + c));
            dot = vaddvq_f32(vdot);
            for (; c < C; c++) dot += sm_row[c] * g_row[c];
#else
            for (int c = 0; c < C; c++) dot += sm_row[c] * g_row[c];
#endif

            /* dL/dx_i = sm_i * (g_i - dot) */
#if DNN_HAVE_NEON
            float32x4_t vdot_vec = vdupq_n_f32(dot);
            c = 0;
            for (; c + 4 <= C; c += 4) {
                float32x4_t vsm = vld1q_f32(sm_row + c);
                float32x4_t vg  = vld1q_f32(g_row + c);
                float32x4_t vag = vld1q_f32(ag_row + c);
                vst1q_f32(ag_row + c,
                    vfmaq_f32(vag, vsm, vsubq_f32(vg, vdot_vec)));
            }
            for (; c < C; c++)
                ag_row[c] += sm_row[c] * (g_row[c] - dot);
#else
            for (int c = 0; c < C; c++)
                ag_row[c] += sm_row[c] * (g_row[c] - dot);
#endif
        }
        return;
    }

    /* ── General nD: fuse coord decompose — compute slice_idx once ── */
    float *dot = mem_scratch_alloc(n_slices * sizeof(float), NULL);
    int   *sid = mem_scratch_alloc(numel * sizeof(int), NULL);

    for (int i = 0; i < numel; i++) {
        int coord[DNN_MAX_DIMS];
        int r = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coord[d] = r % a->shape[d];
            r /= a->shape[d];
        }

        int slice_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d != dim) {
                slice_idx += coord[d] * stride;
                stride *= a->shape[d];
            }
        }

        sid[i] = slice_idx;
        dot[slice_idx] += sm_data[_flat_off(saved_out, i)] * g_data[i];
    }

    for (int i = 0; i < numel; i++) {
        int   si    = sid[i];
        float sm_val = sm_data[_flat_off(saved_out, i)];
        int   off   = _flat_off(a, i);
        ag[off] += sm_val * (g_data[i] - dot[si]);
    }
}

/* ── softmax forward ── */

tensor *tensor_softmax(const tensor *t, int dim) {
    assert(t);
    int ndim = t->ndim;
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "tensor_softmax: dim out of range");

    int numel = _numel(ndim, t->shape);
    int dim_size = t->shape[dim];
    int n_slices = numel / dim_size;
    float *td = (float*)t->data;

    /* per-slice max and sum of exp(x - max) */
    float *max_vals = mem_scratch_alloc(n_slices * sizeof(float), NULL);
    float *sum_vals = mem_scratch_alloc(n_slices * sizeof(float), NULL);

    /* fast path: 2D contiguous, dim=1 (most common — classification) */
    if (ndim == 2 && dim == 1 && tensor_is_contiguous(t)) {
        int N = t->shape[0], C = t->shape[1];
        for (int n = 0; n < N; n++) {
            float *row = (float*)t->data + n * C;
            float mx = simd_reduce_max_f32(row, C);
            float se = simd_exp_sum_shifted_f32(row, C, mx);
            max_vals[n] = mx;
            sum_vals[n] = se;
        }
    } else {
        for (int s = 0; s < n_slices; s++) max_vals[s] = -INFINITY;

        /* Pass 1: find max along dim for each slice */
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % t->shape[d];
                r /= t->shape[d];
            }
            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= t->shape[d];
                }
            }
            float val = td[_bcast_off(t, ndim, coord)];
            if (val > max_vals[slice_idx]) max_vals[slice_idx] = val;
        }

        /* Pass 2: sum of exp(x - max) for each slice */
        for (int s = 0; s < n_slices; s++) sum_vals[s] = 0.0f;
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % t->shape[d];
                r /= t->shape[d];
            }
            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= t->shape[d];
                }
            }
            float val = td[_bcast_off(t, ndim, coord)];
            sum_vals[slice_idx] += expf(val - max_vals[slice_idx]);
        }
    }

    /* create output and compute softmax */
    tensor *out = _tensor_scratch_create(ndim, t->shape, 0);
    float *od = (float*)out->data;

    /* fast path: 2D contiguous output write */
    if (ndim == 2 && dim == 1 && tensor_is_contiguous(t)) {
        int N = t->shape[0], C = t->shape[1];
#if DNN_HAVE_NEON
        for (int n = 0; n < N; n++) {
            float *row = (float*)t->data + n * C;
            float *out_row = od + n * C;
            float mx = max_vals[n];
            float32x4_t vmax = vdupq_n_f32(mx);
            float32x4_t vse  = vdupq_n_f32(sum_vals[n]);
            int c = 0;
            for (; c + 4 <= C; c += 4) {
                float32x4_t shifted = vsubq_f32(vld1q_f32(row + c), vmax);
                vst1q_f32(out_row + c, vdivq_f32(simd_expf_f32(shifted), vse));
            }
            for (; c < C; c++)
                out_row[c] = expf(row[c] - mx) / sum_vals[n];
        }
#else
        for (int n = 0; n < N; n++) {
            float *row = (float*)t->data + n * C;
            float *out_row = od + n * C;
            float mx = max_vals[n];
            float se = sum_vals[n];
            for (int c = 0; c < C; c++)
                out_row[c] = expf(row[c] - mx) / se;
        }
#endif
    } else {
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % t->shape[d];
                r /= t->shape[d];
            }
            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= t->shape[d];
                }
            }
            float val = td[_bcast_off(t, ndim, coord)];
            od[out->offset + i] = expf(val - max_vals[slice_idx]) / sum_vals[slice_idx];
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = softmax_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 2;
        fn->saved_tensors = mem_scratch_alloc(2 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = out;
        int *dim_saved = mem_scratch_alloc(sizeof(int), NULL);
        *dim_saved = dim;
        fn->saved_tensors[1] = (tensor*)dim_saved;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── causal_softmax_backward ──
 *
 * Backward gradient for fused causal softmax.
 * Same formula as softmax backward:
 *   dL/dx[i][j] = sm[i][j] * (g[i][j] - dot_i)  for j <= i
 *   dL/dx[i][j] = 0                               for j > i
 *
 * The sm[i][j] = 0 for j > i naturally zeros masked positions.
 * dot_i = sum_{j <= i} sm[i][j] * g[i][j]  (only visible positions).
 */

static void causal_softmax_backward(grad_fn *fn, tensor *grad_output) {
    tensor *a = fn->inputs[0];
    tensor *saved_out = fn->saved_tensors[0];
    int ndim = a->ndim;
    int N = a->shape[ndim - 1];  /* last dim size, also = second-to-last */
    float *g_data = (float*)grad_output->data;
    float *sm_data = (float*)saved_out->data;

    if (!(a->grad_fn || a->requires_grad)) return;
    float *ag = _grad_ensure(a);

    /* ── 2D fast path (N, N), contiguous ── */
    if (ndim == 2 && tensor_is_contiguous(a) && tensor_is_contiguous(grad_output)) {
        for (int i = 0; i < N; i++) {
            float *sm_row = sm_data + i * N;
            float *g_row  = g_data + i * N;
            float *ag_row = ag + i * N;

            /* dot_i = sum_{j <= i} sm[i][j] * g[i][j] */
            float dot;
#if DNN_HAVE_NEON
            {
                float32x4_t vdot = vdupq_n_f32(0.0f);
                int j = 0;
                for (; j + 4 <= i; j += 4) {
                    vdot = vfmaq_f32(vdot, vld1q_f32(sm_row + j),
                                           vld1q_f32(g_row + j));
                }
                dot = vaddvq_f32(vdot);
                for (; j <= i; j++) dot += sm_row[j] * g_row[j];
            }
#else
            dot = 0.0f;
            for (int j = 0; j <= i; j++)
                dot += sm_row[j] * g_row[j];
#endif

            /* dL/dx[i][j] = sm[i][j] * (g[i][j] - dot_i) for j <= i */
            for (int j = 0; j <= i; j++)
                ag_row[j] += sm_row[j] * (g_row[j] - dot);
            /* j > i: sm=0 → gradient 0, no-op */
        }
        return;
    }

    /* ── General nD (batch dims before last 2) ── */
    int n_matrices = 1;
    for (int d = 0; d < ndim - 2; d++)
        n_matrices *= a->shape[d];

    for (int m = 0; m < n_matrices; m++) {
        float *sm_mat = sm_data + m * N * N;
        float *g_mat  = g_data  + m * N * N;
        float *ag_mat = ag      + m * N * N;

        for (int i = 0; i < N; i++) {
            float *sm_row = sm_mat + i * N;
            float *g_row  = g_mat  + i * N;
            float *ag_row = ag_mat + i * N;

            float dot;
#if DNN_HAVE_NEON
            {
                float32x4_t vdot = vdupq_n_f32(0.0f);
                int j = 0;
                for (; j + 4 <= i; j += 4) {
                    vdot = vfmaq_f32(vdot, vld1q_f32(sm_row + j),
                                           vld1q_f32(g_row + j));
                }
                dot = vaddvq_f32(vdot);
                for (; j <= i; j++) dot += sm_row[j] * g_row[j];
            }
#else
            dot = 0.0f;
            for (int j = 0; j <= i; j++)
                dot += sm_row[j] * g_row[j];
#endif

            for (int j = 0; j <= i; j++)
                ag_row[j] += sm_row[j] * (g_row[j] - dot);
        }
    }
}

/* ── tensor_causal_softmax ──
 *
 * Fused causal softmax over the last dimension.
 * Input shape: (..., N, N) where last two dims are equal.
 * For query position i (dim ndim-2), key positions j > i (dim ndim-1)
 * are masked with -inf before softmax.
 *
 * No mask materialized — the triangular mask is applied implicitly
 * by only computing softmax over visible positions j=0..i per row.
 */

tensor *tensor_causal_softmax(const tensor *t) {
    assert(t);
    int ndim = t->ndim;
    assert(ndim >= 2 && "causal_softmax: need at least 2 dims");
    assert(t->shape[ndim - 2] == t->shape[ndim - 1]
           && "causal_softmax: last two dims must be equal (square)");

    int N = t->shape[ndim - 1];
    int n_matrices = 1;
    for (int d = 0; d < ndim - 2; d++)
        n_matrices *= t->shape[d];

    float *td = (float*)t->data;

    tensor *out = _tensor_scratch_create(ndim, t->shape, 0);
    float *od = (float*)out->data;

    /* ── Fused causal softmax with online max+sum_exp + NEON SIMD ──
     *
     * Both 2D fast path and general nD path share the same fused logic.
     * The inner loop body is factored inline to avoid function call overhead
     * while keeping the fused pattern (1 pass max+sum_exp, 1 pass write).
     */
    if (ndim == 2 && tensor_is_contiguous(t)) {
        for (int i = 0; i < N; i++) {
            float *row_in = td + i * N;
            float *row_out = od + i * N;

            /* ── Fused pass: online max + sum_exp ── */
            float mx = -INFINITY;
            float se = 0.0f;
#if DNN_HAVE_NEON
            {
                int j = 0;
                for (; j + 4 <= i + 1; j += 4) {
                    float32x4_t v = vld1q_f32(row_in + j);
                    float group_max = vmaxvq_f32(v);
                    if (group_max > mx) {
                        se *= expf(mx - group_max);
                        mx = group_max;
                    }
                    float32x4_t shifted = vsubq_f32(v, vdupq_n_f32(mx));
                    se += vaddvq_f32(simd_expf_f32(shifted));
                }
                for (; j <= i; j++) {
                    float old_mx = mx;
                    if (row_in[j] > mx) mx = row_in[j];
                    if (mx != old_mx) se *= expf(old_mx - mx);
                    se += expf(row_in[j] - mx);
                }
            }
#else
            for (int j = 0; j <= i; j++) {
                float old_mx = mx;
                if (row_in[j] > mx) mx = row_in[j];
                if (mx != old_mx) se *= expf(old_mx - mx);
                se += expf(row_in[j] - mx);
            }
#endif

            /* ── Write softmax weights (1 pass) ── */
            float inv_se = 1.0f / se;
#if DNN_HAVE_NEON
            {
                int j = 0;
                float32x4_t vmx = vdupq_n_f32(mx);
                float32x4_t vinv_se = vdupq_n_f32(inv_se);
                for (; j + 4 <= i + 1; j += 4) {
                    float32x4_t v = vld1q_f32(row_in + j);
                    float32x4_t exp_v = simd_expf_f32(vsubq_f32(v, vmx));
                    vst1q_f32(row_out + j, vmulq_f32(exp_v, vinv_se));
                }
                for (; j <= i; j++)
                    row_out[j] = expf(row_in[j] - mx) * inv_se;
            }
#else
            for (int j = 0; j <= i; j++)
                row_out[j] = expf(row_in[j] - mx) * inv_se;
#endif
            for (int j = i + 1; j < N; j++)
                row_out[j] = 0.0f;
        }
    } else {
        /* ── General nD ── */
        for (int m = 0; m < n_matrices; m++) {
            float *mat_in  = td + m * N * N;
            float *mat_out = od + m * N * N;

            for (int i = 0; i < N; i++) {
                float *row_in  = mat_in  + i * N;
                float *row_out = mat_out + i * N;

                /* ── Fused pass: online max + sum_exp ── */
                float mx = -INFINITY;
                float se = 0.0f;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    for (; j + 4 <= i + 1; j += 4) {
                        float32x4_t v = vld1q_f32(row_in + j);
                        float group_max = vmaxvq_f32(v);
                        if (group_max > mx) {
                            se *= expf(mx - group_max);
                            mx = group_max;
                        }
                        float32x4_t shifted = vsubq_f32(v, vdupq_n_f32(mx));
                        se += vaddvq_f32(simd_expf_f32(shifted));
                    }
                    for (; j <= i; j++) {
                        float old_mx = mx;
                        if (row_in[j] > mx) mx = row_in[j];
                        if (mx != old_mx) se *= expf(old_mx - mx);
                        se += expf(row_in[j] - mx);
                    }
                }
#else
                for (int j = 0; j <= i; j++) {
                    float old_mx = mx;
                    if (row_in[j] > mx) mx = row_in[j];
                    if (mx != old_mx) se *= expf(old_mx - mx);
                    se += expf(row_in[j] - mx);
                }
#endif

                /* ── Write softmax weights (1 pass) ── */
                float inv_se = 1.0f / se;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    float32x4_t vmx = vdupq_n_f32(mx);
                    float32x4_t vinv_se = vdupq_n_f32(inv_se);
                    for (; j + 4 <= i + 1; j += 4) {
                        float32x4_t v = vld1q_f32(row_in + j);
                        float32x4_t exp_v = simd_expf_f32(vsubq_f32(v, vmx));
                        vst1q_f32(row_out + j, vmulq_f32(exp_v, vinv_se));
                    }
                    for (; j <= i; j++)
                        row_out[j] = expf(row_in[j] - mx) * inv_se;
                }
#else
                for (int j = 0; j <= i; j++)
                    row_out[j] = expf(row_in[j] - mx) * inv_se;
#endif
                for (int j = i + 1; j < N; j++)
                    row_out[j] = 0.0f;
            }
        }
    }

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = causal_softmax_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;
        fn->n_saved = 1;
        fn->saved_tensors = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = out;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}


/* ── cross_entropy_backward ── */

static void cross_entropy_backward(grad_fn *fn, tensor *grad_output) {
    tensor *logits   = fn->inputs[0];
    float  *max_vals = (float*)fn->saved_tensors[0];
    float  *sum_exp  = (float*)fn->saved_tensors[1];
    tensor *target   = fn->saved_tensors[2];
    int     dim      = *(int*)fn->saved_tensors[3];
    float   inv_N    = *(float*)fn->saved_tensors[4];

    int ndim    = logits->ndim;
    int numel   = _numel(ndim, logits->shape);
    float *ld   = (float*)logits->data;
    float *gd   = (float*)grad_output->data;
    int   *td   = (int*)target->data;
    float  gout = gd[0];  /* scalar grad_output for a loss */

    if (logits->grad_fn || logits->requires_grad) {
        float *ag = _grad_ensure(logits);

        /* fast path: 2D contiguous logits, dim=1 (most common — classification) */
        if (ndim == 2 && dim == 1 && tensor_is_contiguous(logits)) {
            int N = logits->shape[0], C = logits->shape[1];
            float scale = gout * inv_N;
            for (int n = 0; n < N; n++) {
                float *row = ld + n * C;
                float *ag_row = ag + n * C;
#if DNN_HAVE_NEON
                simd_ce_bwd_row_kernel(ag_row, row, max_vals[n], sum_exp[n],
                                       td[n], scale, C);
#else
                float mx = max_vals[n];
                float se = sum_exp[n];
                int tgt = td[n];
                for (int c = 0; c < C; c++) {
                    float sm = expf(row[c] - mx) / se;
                    ag_row[c] += (sm - (c == tgt ? 1.0f : 0.0f)) * scale;
                }
#endif
            }
        } else {
            for (int i = 0; i < numel; i++) {
                int coord[DNN_MAX_DIMS];
                int r = i;
                for (int d = ndim - 1; d >= 0; d--) {
                    coord[d] = r % logits->shape[d];
                    r /= logits->shape[d];
                }

                int slice_idx = 0;
                int stride = 1;
                for (int d = ndim - 1; d >= 0; d--) {
                    if (d != dim) {
                        slice_idx += coord[d] * stride;
                        stride *= logits->shape[d];
                    }
                }

                float val   = ld[_bcast_off(logits, ndim, coord)];
                float sm    = expf(val - max_vals[slice_idx]) / sum_exp[slice_idx];
                int is_tgt  = (coord[dim] == td[slice_idx]) ? 1 : 0;

                int off = _flat_off(logits, i);
                ag[off] += (sm - (float)is_tgt) * gout * inv_N;
            }
        }
    }
}

/* ── cross_entropy forward ── */

tensor *tensor_cross_entropy(const tensor *logits, const tensor *target, int dim) {
    assert(logits && target);
    int ndim = logits->ndim;
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "tensor_cross_entropy: dim out of range");

    int numel    = _numel(ndim, logits->shape);
    int dim_size = logits->shape[dim];
    int n_slices = numel / dim_size;
    float *ld = (float*)logits->data;
    int   *td = (int*)target->data;

    /* per-slice max and sum of exp(x - max) */
    float *max_vals = mem_scratch_alloc(n_slices * sizeof(float), NULL);
    float *sum_exp  = mem_scratch_alloc(n_slices * sizeof(float), NULL);

    for (int s = 0; s < n_slices; s++) max_vals[s] = -INFINITY;

    float total_loss = 0.0f;

    /* fast path: 2D contiguous logits, dim=1 (most common — classification)
     * 1 fused pass: online max + sum_exp via online softmax (running max,
     * adaptive sum_exp).  Saves 1 full C-element read per row.
     */
    if (ndim == 2 && dim == 1 && tensor_is_contiguous(logits)) {
        int N = logits->shape[0], C = logits->shape[1];
        for (int n = 0; n < N; n++) {
            float *row = ld + n * C;
            float mx, se;
#if DNN_HAVE_NEON
            mx = -INFINITY;
            se = 0.0f;
            int j = 0;
            for (; j + 4 <= C; j += 4) {
                float32x4_t v = vld1q_f32(row + j);
                float chunk_max = vmaxvq_f32(v);
                if (chunk_max > mx) {
                    se *= expf(mx - chunk_max);
                    mx = chunk_max;
                }
                float32x4_t shifted = vsubq_f32(v, vdupq_n_f32(mx));
                se += vaddvq_f32(simd_expf_f32(shifted));
            }
            for (; j < C; j++) {
                float v = row[j];
                if (v > mx) { se *= expf(mx - v); mx = v; }
                se += expf(v - mx);
            }
#else
            mx = -INFINITY;
            se = 0.0f;
            for (int j = 0; j < C; j++) {
                float v = row[j];
                if (v > mx) { se *= expf(mx - v); mx = v; }
                se += expf(v - mx);
            }
#endif
            max_vals[n] = mx;
            sum_exp[n]  = se;
            total_loss += logf(se) + mx - row[td[n]];
        }
    } else {
        /* general nD fallback — 2 full passes over logits (was 3).
         * Pass 1 finds max per slice.  Pass 2 computes sum_exp and saves
         * the target logit value.  Loss computed per-slice from saved values.
         */
        float *target_logits = mem_scratch_alloc(n_slices * sizeof(float), NULL);

        /* Pass 1: find max along dim for each slice */
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % logits->shape[d];
                r /= logits->shape[d];
            }

            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= logits->shape[d];
                }
            }

            float val = ld[_bcast_off(logits, ndim, coord)];
            if (val > max_vals[slice_idx]) max_vals[slice_idx] = val;
        }

        /* Pass 2: sum of exp(x - max) + save target logit per slice */
        for (int s = 0; s < n_slices; s++) { sum_exp[s] = 0.0f; target_logits[s] = 0.0f; }
        for (int i = 0; i < numel; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % logits->shape[d];
                r /= logits->shape[d];
            }

            int slice_idx = 0;
            int stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d != dim) {
                    slice_idx += coord[d] * stride;
                    stride *= logits->shape[d];
                }
            }

            float val = ld[_bcast_off(logits, ndim, coord)];
            sum_exp[slice_idx] += expf(val - max_vals[slice_idx]);
            if (coord[dim] == td[slice_idx])
                target_logits[slice_idx] = val;
        }

        /* compute loss per slice from saved max, sum_exp, target_logit */
        total_loss = 0.0f;
        for (int s = 0; s < n_slices; s++) {
            float lse = max_vals[s] + logf(sum_exp[s]);
            total_loss += lse - target_logits[s];
        }
    }

    float inv_N = 1.0f / (float)n_slices;
    float loss_val = total_loss * inv_N;

    tensor *out = _tensor_scratch_create(1, (int[]){1}, 0);
    ((float*)out->data)[0] = loss_val;

    /* autograd tape */
    if (dnn_grad_enabled() && tensor_requires_grad(logits)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward = cross_entropy_backward;
        fn->n_inputs = 1;
        fn->inputs = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)logits;
        fn->n_saved = 5;
        fn->saved_tensors = mem_scratch_alloc(5 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)max_vals;
        fn->saved_tensors[1] = (tensor*)sum_exp;
        fn->saved_tensors[2] = (tensor*)target;
        int *dim_saved = mem_scratch_alloc(sizeof(int), NULL);
        *dim_saved = dim;
        fn->saved_tensors[3] = (tensor*)dim_saved;
        float *inv_n_saved = mem_scratch_alloc(sizeof(float), NULL);
        *inv_n_saved = inv_N;
        fn->saved_tensors[4] = (tensor*)inv_n_saved;
        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

/* ── dropout_backward ── */

static void dropout_backward(grad_fn *fn, tensor *grad_output) {
    unsigned char *mask = (unsigned char*)fn->saved_tensors[0];
    float   p    = *(float*)fn->saved_tensors[1];
    int     n    = *(int*)fn->saved_tensors[2];
    tensor *input = fn->inputs[0];

    float scale = 1.0f / (1.0f - p);
    float *gd   = (float*)grad_output->data;
    int    ndim = input->ndim;

    if (tensor_requires_grad(input)) {
        float *ig = _grad_ensure(input);
        if (tensor_is_contiguous(input)) {
            int off = input->offset;
            for (int i = 0; i < n; i++)
                ig[off + i] += gd[i] * (float)mask[i] * scale;
        } else {
            for (int i = 0; i < n; i++) {
                int coord[DNN_MAX_DIMS];
                int r = i;
                for (int d = ndim - 1; d >= 0; d--) {
                    coord[d] = r % input->shape[d];
                    r /= input->shape[d];
                }
                ig[_bcast_off(input, ndim, coord)] += gd[i] * (float)mask[i] * scale;
            }
        }
    }
}

/* ── dropout forward ── */

tensor *tensor_dropout(const tensor *t, float p) {
    assert(t);
    assert(p >= 0.0f && p < 1.0f);

    /* eval mode: identity */
    if (!dnn_grad_enabled()) return (tensor*)t;

    int ndim = t->ndim;
    int n    = tensor_numel(t);
    float scale = 1.0f / (1.0f - p);

    tensor *out = _tensor_scratch_create(ndim, t->shape, 0);
    float  *od  = (float*)out->data;
    float  *td  = (float*)t->data;

    /* generate mask and apply: out = mask * t / (1-p)
     * mask stored as byte array (75% less memory than float mask). */
    unsigned char *mask = mem_scratch_alloc((size_t)n * sizeof(unsigned char), NULL);
    if (tensor_is_contiguous(t)) {
        float *tp = td + t->offset;
        for (int i = 0; i < n; i++) {
            mask[i] = ((float)rand() / (float)RAND_MAX) >= p ? 1 : 0;
            od[i]   = (float)mask[i] * tp[i] * scale;
        }
    } else {
        for (int i = 0; i < n; i++) {
            int coord[DNN_MAX_DIMS];
            int r = i;
            for (int d = ndim - 1; d >= 0; d--) {
                coord[d] = r % t->shape[d];
                r /= t->shape[d];
            }
            mask[i] = ((float)rand() / (float)RAND_MAX) >= p ? 1 : 0;
            od[i]   = (float)mask[i] * td[_bcast_off(t, ndim, coord)] * scale;
        }
    }

    /* autograd tape */
    if (tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create();
        fn->backward  = dropout_backward;
        fn->n_inputs  = 1;
        fn->inputs    = mem_scratch_alloc(1 * sizeof(tensor*), NULL);
        fn->inputs[0] = (tensor*)t;

        fn->n_saved = 3;
        fn->saved_tensors = mem_scratch_alloc(3 * sizeof(tensor*), NULL);
        fn->saved_tensors[0] = (tensor*)mask;

        float *p_saved = mem_scratch_alloc(sizeof(float), NULL);
        *p_saved = p;
        fn->saved_tensors[1] = (tensor*)p_saved;

        int *n_saved = mem_scratch_alloc(sizeof(int), NULL);
        *n_saved = n;
        fn->saved_tensors[2] = (tensor*)n_saved;

        out->requires_grad = 1;
        out->grad_fn = fn;
    }

    return out;
}

tensor *tensor_tanh(const tensor *t) {
    (void)t;
    return NULL;
}
