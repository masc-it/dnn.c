#include "rope.h"
#include "pool.h"
#include "tensor_int.h"
#include <assert.h>
#include <math.h>

tensor *tensor_rope_freqs(int d, float base) {
    assert(d > 0 && "rope_freqs: d must be positive");
    assert(d % 2 == 0 && "rope_freqs: d must be even");
    assert(base > 0.0f && "rope_freqs: base must be positive");

    int half = d / 2;

    /* output shape [half] */
    tensor *out = _tensor_scratch_create(1, (int[]){half}, 0);
    float *od = (float*)out->data + out->offset;

    /* theta_k = base^{-2k/d}   for k = 0 .. half-1
     *
     * Compute as: exp(-2k * ln(base) / d)
     *
     * Faster and avoids powf() per element — one log, then linear exponent.
     * powf(base, -2k/d) = exp((-2k/d) * ln(base))
     *                    = exp(k * (-2 * ln(base) / d))
     *
     * Let step = -2.0f * logf(base) / (float)d
     * Then theta_k = exp(k * step)
     */
    float step = -2.0f * logf(base) / (float)d;

    for (int k = 0; k < half; k++) {
        od[k] = expf(k * step);
    }

    return out;
}
