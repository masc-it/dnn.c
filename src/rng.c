#include "rng.h"
#include <math.h>

/* ── xorshift128+ core ── */

static inline uint64_t xorshift128plus(dnn_rng *rng) {
    uint64_t s1 = rng->s[0];
    uint64_t s0 = rng->s[1];
    rng->s[0] = s0;
    s1 ^= s1 << 23;
    rng->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return rng->s[1] + s0;
}

/* ── Seed ── */

dnn_rng dnn_rng_seed(uint64_t seed) {
    dnn_rng rng;
    /* SplitMix64: two rounds to fill both state words */
    uint64_t z = (seed + 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    rng.s[0] = z ^ (z >> 31);
    z = (rng.s[0] + 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    rng.s[1] = z ^ (z >> 31);
    return rng;
}

/* ── Uniform ── */

float dnn_rng_uniform(dnn_rng *rng) {
    /* Upper 53 bits → float in [0, 1) */
    return (float)(xorshift128plus(rng) >> 11) * 0x1.0p-53f;
}

/* ── Normal (Box-Muller, single sample) ── */

float dnn_rng_normal(dnn_rng *rng) {
    float u1, u2;
    do {
        u1 = dnn_rng_uniform(rng);
    } while (u1 == 0.0f);          /* re-roll zero — no log(0) bias */
    u2 = dnn_rng_uniform(rng);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

/* ── Uniform integer [0, n) — rejection-bias-free ── */

int dnn_rng_uniform_int(dnn_rng *rng, int n) {
    if (n <= 1) return 0;
    /* 32-bit range, rejection-sample to avoid mod bias */
    uint32_t limit = (uint32_t)n;
    uint32_t mask = limit;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    uint32_t v;
    do {
        v = (uint32_t)(xorshift128plus(rng) >> 32) & mask;
    } while (v >= limit);
    return (int)v;
}

/* ── Global thread-local RNG ── */

static _Thread_local dnn_rng _dnn_default_rng = { .s = { 1, 2 } };

dnn_rng *dnn_get_rng(void) {
    return &_dnn_default_rng;
}

void dnn_seed(uint64_t seed) {
    _dnn_default_rng = dnn_rng_seed(seed);
}
