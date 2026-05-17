#ifndef DNN_RNG_H
#define DNN_RNG_H

#include <stdint.h>
#include <stddef.h>

/* ── xorshift128+ PRNG ──
 *
 * Fast, seedable, thread-safe if each thread has its own instance.
 * Passes BigCrush.  Period 2^128 - 1.
 */
typedef struct dnn_rng {
    uint64_t s[2];
} dnn_rng;

/* Seed from a single 64-bit value (SplitMix64 mixer). */
dnn_rng dnn_rng_seed(uint64_t seed);

/* Uniform float in [0, 1).  53-bit mantissa via 64-bit extraction. */
float dnn_rng_uniform(dnn_rng *rng);

/* Standard normal via Box-Muller.  Re-rolls u1 == 0 (no bias clamp). */
float dnn_rng_normal(dnn_rng *rng);

/* Uniform integer in [0, n).  n > 0.  Rejection-bias-free. */
int dnn_rng_uniform_int(dnn_rng *rng, int n);

/* ── Global thread-local RNG ──
 *
 * Convenience default for library ops (tensor_randn, dropout, sampling).
 * Seeded with {1, 2} on first access.  Override via dnn_seed().
 */
dnn_rng *dnn_get_rng(void);
void      dnn_seed(uint64_t seed);

#endif /* DNN_RNG_H */
