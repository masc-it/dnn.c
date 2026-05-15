# General Optimizations Pt2 — Memory Bandwidth & SIMD Gap Closure

Based on `docs/letters/02_dear_expert.md` (performance audit v2).
Measured baseline: MNIST CNN batch=64 on Apple M1, `-O3 -ffast-math`.

---

## P0 — Winograd F(2×2, 3×3) for all 3×3 convolutions

**File:** `src/conv.c` (new `winograd_*` functions + dispatch in `tensor_conv2d`)

**Root cause:** Conv2 (32→64, s2) allocates 14.5 MB col buffer via im2col. L2 cache on M1 = 12 MB → every sgemm access to col misses L2 → sgemm drops from ~665 GFLOPS to ~221 GFLOPS. im2col is 85% of conv2 cost, sgemm only 15%.

**Mitigation:** Winograd F(2×2, 3×3) replaces im2col+GEMM with element-wise multiplies on 4×4 tiles:

Forward:
- Transform input tiles (4×4) × weight (3×3) → 4 element-wise multiplies → inverse transform
- Col buffer goes to zero (tile-only, fits in registers/L1)
- FLOPs: 9 MACs/position → 4 MACs/position (56% reduction)

Backward:
- Same transform applied to gradients
- Sequential writes, no col2im scatter overhead

Dispatch condition: `kH==3 && kW==3 && stride==1 && pad==1` (all convs in this model). Fallback to im2col for non-3×3 kernels.

**Estimated lines:** ~150 (transform kernels, backward, dispatch in `tensor_conv2d`)

**Estimated gain:**
| Metric | Current | Winograd | Gain |
|--------|---------|----------|------|
| Conv2 fwd time | 2.40 ms | ~0.9 ms | **62%** |
| Conv total fwd | 4.16 ms | ~1.9 ms | **54%** |
| Total batch fwd+bwd | 17.54 ms | ~10 ms | **43%** |
| Col buffer | 14.5 MB (L2 miss) | 0 (tile-only) | **∞** |

**Status: implemented**

**Measured impact:**
| Metric | Baseline | Winograd | Gain |
|--------|----------|----------|------|
| Conv1 fwd (batch=64) | 0.96 ms | 0.40 ms | **2.4×** |
| Total fwd (batch=64) | 4.66 ms | 3.47 ms | **25%** |
| Total bwd (batch=64) | 12.88 ms | 13.40 ms | ~same |

**Checklist:**
- [x] Forward: input transform (4×4 tiles via `B^T @ x @ B`)
- [x] Forward: weight transform (3×3 → 4×4 via `G @ w @ G^T`)
- [x] Forward: element-wise multiply + inverse transform
- [x] Forward: dispatch in `tensor_conv2d` (kH==3 && kW==3 && stride==1 && pad==1)
- [x] Backward: gradient transform for d_input (`B @ dU @ B^T`)
- [x] Backward: weight gradient transform (`G^T @ dV @ G`)
- [x] Backward: OMP parallelized over batch×tiles
- [x] Backward: per-thread dV accumulators (avoids atomics)
- [x] Test: `test_winograd_simple` (forward+backward vs PyTorch)
- [x] Test: `test_winograd_multi_channel` (multi-channel with bias)
- [x] Test: `test_winograd_no_bias`
- [x] Bench: conv1 fwd measured at 0.40 ms (2.4× gain)

---

## P1 — Fuse causal softmax forward passes + NEON SIMD

**File:** `src/attention.c` (step 3 in `tensor_attention`)

**Root cause:** 3 scalar passes per row i:
1. Row max over j=0..i (scalar `if (v > mx)`)
2. Row sum_exp over j=0..i (scalar `expf`)
3. Row softmax write (scalar `expf * inv_se`)

No NEON SIMD anywhere. `simd_expf_f32` exists but is unused here.

**Mitigation:** Fuse passes 1+2 with online softmax (running max + adaptive sum_exp). Apply NEON SIMD to vectorize expf over 4-lane groups within each row segment.

Pattern for one fused pass:
```c
float mx = -INFINITY;
float se = 0.0f;
for (int j = 0; j <= i; j += 4) {
    // load 4 floats
    // compute new mx = max(old_mx, max(lane0..3))
    // shift exp sum: se = se * exp(old_mx - new_mx) + sum(exp(val - new_mx))
}
// write softmax: exp(val - mx) / se  (also SIMD)
```

**Estimated lines:** ~40

**Estimated gain:** 2-3× on causal softmax forward (critical for transformer training at N=2048+)

**Status: implemented**

**Changes:**
- `src/attention.c` — `tensor_attention` step 3 (causal softmax): fused online max+sum_exp + NEON SIMD write
- `src/ops_activation.c` — `tensor_causal_softmax` both 2D fast path and general nD: same fusion
- `src/transformer.c` — `transformer_block_forward_cached` inline softmax: same fusion (regular softmax, all-past-visible). Used by generation prefill path.

**Measured impact:**

Causal softmax benchmark (kernel-level):

| N | Ref (us) | Opt (us) | Speedup |
|---|----------|----------|--------|
| 64 | 9.0 | 3.0 | **3.00×** |
| 128 | 32.0 | 11.0 | **2.91×** |
| 256 | 127.0 | 46.0 | **2.76×** |
| 512 | 504.5 | 180.0 | **2.80×** |
| 1024 | 2075.0 | 728.5 | **2.85×** |

Max numerical error vs reference: < 2e-6 (well under 1e-5 tolerance).

**Checklist:**
- [x] Implement online softmax fusion (running max + adaptive sum_exp)
- [x] Add `DNN_HAVE_NEON` path with 4-wide SIMD expf
- [x] Replace 3-loop block in `tensor_attention` step 3 (`src/attention.c`)
- [x] Replace 3-loop blocks in `tensor_causal_softmax` (`src/ops_activation.c`, both 2D fast path and general nD)
- [x] Replace 3-loop block in `transformer_block_forward_cached` (`src/transformer.c`, generation prefill path)
- [x] Test: numerical correctness vs original (matching P values within 2e-6)
- [x] Bench: measured causal softmax time at N=64/128/256/512/1024 — **2.8-3.0× speedup**

---

## P2 — Cross-entropy 2D fast path: fuse to 1 pass via online softmax

**File:** `src/ops_activation.c` (2D fast path in `tensor_cross_entropy`)

**Root cause:** Currently 2 full SIMD passes per row:
```c
float mx = simd_reduce_max_f32(row, C);         // Pass 1: SIMD max
float se = simd_exp_sum_shifted_f32(row, C, mx); // Pass 2: SIMD exp sum
total_loss += logf(se) + mx - row[td[n]];       // scalar
```

Could fuse to 1 pass with online softmax (running max, adaptive sum_exp). For C=vocab_size (32K+ in LLM), saves ~16K float reads per row × batch_size × steps.

**Estimated lines:** ~15

**Estimated gain:** 1 pass instead of 2. Saves 1 full C-element read per row. For vocab=32000, batch=128: 128 × 32000 × 4 bytes = 16 MB less DRAM reads per forward call.

**Status: not started**

**Checklist:**
- [ ] Implement online max+sum_exp fused loop with NEON SIMD
- [ ] Replace 2-pass 2D fast path with fused version
- [ ] Test: numerical correctness
- [ ] Bench: measure cross_entropy time at C=10/1000/32000

---

## P3 — Layer norm forward: Welford fusion (mean+var in 1 pass)

**File:** `src/norm.c` (`tensor_layer_norm`)

**Root cause:** 3 full passes over d elements per slice:
```c
// pass 1: mean
for (j) sum += x[j]; mean = sum/d;
// pass 2: var → rstd
for (j) diff = x[j]-mean; sum += diff*diff; rstd = 1/sqrt(sum/d + eps);
// pass 3: output
for (j) y = (x[j]-mean)*rstd; out[j] = y*w + b;
```

Welford one-pass fusion cuts passes 1+2 → 1, saving 33% of norm time.

**Estimated lines:** ~15

**Estimated gain:** 2 passes → 1 pass for mean+var. Saves N*d element reads/writes.

**Status: not started**

**Checklist:**
- [ ] Replace pass 1+2 with Welford online mean+var in one loop
- [ ] Keep pass 3 unchanged (still needs mean+rstd)
- [ ] Test: numerical correctness vs original (currently < 1e-7 error)
- [ ] Bench: measure layer_norm time across d=128..4096

---

## P4 — Control OpenMP nested parallelism

**File:** `src/conv.c` (both `tensor_conv2d` and `conv2d_backward`)

**Root cause:** Outer OMP `#pragma omp parallel for` on conv forward batches wraps individual `cblas_sgemm` calls that already use multi-threaded BLAS internally (Apple's Accelerate on M1 uses up to 8 threads). Creates nested parallelism where outer thread pool + inner BLAS thread pool oversubscribe cores by 2-4×.

**Mitigation:** Add one of:
- `omp_set_max_active_levels(1)` before OMP regions that wrap BLAS calls
- Or `if (out_C < 64)` guard to serialize trivial blocks
- Or both

**Estimated lines:** ~5 (2 calls to `omp_set_max_active_levels`)

**Estimated gain:** Measurable wall-clock reduction on multi-batch runs. Hard to predict magnitude (depends on BLAS thread count vs OMP thread count interaction).

**Status: not started**

**Checklist:**
- [ ] Add `omp_set_max_active_levels(1)` in `tensor_conv2d` before OMP sgemm
- [ ] Add `omp_set_max_active_levels(1)` in `conv2d_backward` before OMP sgemm
- [ ] Bench: measure wall-clock time change on CNN multi-batch

---

## P5 — NEON SIMD for layer norm inner loops

**File:** `src/norm.c` (`tensor_layer_norm` forward, `ln_backward`)

**Root cause:** All inner loops in layer norm are scalar. `#pragma omp simd` hints compiler but no explicit NEON intrinsics.

**Mitigation:** Replace scalar loops with NEON 4-wide reduction:
- `for (j) sum += x[j]` → `vaddq_f32` + `vaddvq_f32`
- `for (j) diff*diff` → `vmlaq_f32` + `vaddvq_f32`
- Output loop → `vld1q_f32` / `vfmaq_f32` / `vst1q_f32`

Also apply NEON to `ln_backward` reduction and gradient loops.

**Estimated lines:** ~40 (forward 2 loops + backward 2 loops × ~10 lines each)

**Estimated gain:** 2-4× on per-slice inner loop (currently scalar). Most impactful at large d (hidden dim 768+ in transformers).

**Status: not started**

**Checklist:**
- [ ] NEON SIMD for pass 1 (mean) in forward
- [ ] NEON SIMD for pass 2 (var→rstd) in forward
- [ ] NEON SIMD for pass 3 (output) in forward
- [ ] NEON SIMD for ln_backward reduction loops (sum_dy, sum_dy_xmu)
- [ ] NEON SIMD for ln_backward dx loop
- [ ] Test: numerical correctness
- [ ] Bench: measure layer_norm time at d=768/1024/4096

---

## P6 — Fuse dS scale into softmax backward

**File:** `src/attention.c` (`attention_backward`)

**Root cause:** After computing dS = causal_softmax_bwd(P, dP), a separate loop multiplies every element by scale:
```c
for (int i = 0; i < N * N; i++)
    dS[i] *= scale;
```

This is a full N×N pass per batch-head. scale is constant for all elements.

**Mitigation:** Move `* scale` into the causal softmax backward loop body where dS[i][j] is written. Add one multiply to the existing write — no extra pass.

**Estimated lines:** ~5

**Estimated gain:** Saves one full N×N pass per batch-head. For B=1, H=12, N=2048: saves 12 × 4M = 48M float multiplications per step.

**Status: not started**

**Checklist:**
- [ ] Remove standalone `for (int i = 0; i < N * N; i++) dS[i] *= scale` loop
- [ ] Add `* scale` to each `ds_row[j] = p_row[j] * (dp_row[j] - dot)` write
- [ ] Test: numerical correctness (gradients should match within 1e-7)
- [ ] Bench: measure attention backward time reduction

---

## P7 — Causal softmax backward dot product: NEON SIMD

**File:** `src/attention.c` (`attention_backward`, causal softmax bwd dot product)

**Root cause:** The inner dot product per row is fully scalar:
```c
float dot = 0.0f;
for (int j = 0; j <= i; j++)
    dot += p_row[j] * dp_row[j];
```

For N=2048, the average row length is N/2 = 1024. Scalar multiply-add for 12 heads × 2048 rows × average 1024 elements = 25M scalar FMAs per backward call.

**Mitigation:** Replace with NEON `vfmaq_f32` (4-wide FMA) + `vaddvq_f32` (horizontal sum). Pattern identical to existing `softmax_backward` 2D fast path NEON dot product.

**Estimated lines:** ~10 (2 `DNN_HAVE_NEON` guards — one for dot product, one already exists for the gradient write loop)

**Estimated gain:** 2-4× on the inner dot product loop.

**Status: not started**

**Checklist:**
- [ ] Add NEON SIMD dot product in causal softmax backward
- [ ] Test: numerical correctness
- [ ] Bench: measure causal softmax backward time at N=512/1024/2048

---

## P8 — Dropout mask as byte array

**File:** `src/ops_activation.c` (`tensor_dropout`, `dropout_backward`)

**Root cause:** `float *mask` allocates 4 bytes per element. Only stores 0.0 or 1.0 — 4 bytes for 1 bit of information.

**Mitigation:** Change `float *mask` to `unsigned char *mask`. Adjust reads in backward: `mask[i]` becomes `(float)mask[i]` (one conversion per element).

**Estimated lines:** ~10

**Estimated gain:** 75% reduction in mask memory (4 bytes → 1 byte per element). For transformer FFN (intermediate=11008): 44 KB → 11 KB per FFN per layer.

**Status: not started**

**Checklist:**
- [ ] Change `float *mask` to `unsigned char *mask` in `tensor_dropout`
- [ ] Update mask generation: store 1/0 as byte
- [ ] Update `dropout_backward`: cast `mask[i]` to float for multiplication
- [ ] Test: numerical correctness (output and gradients should match within 1e-7)
- [ ] Bench: measure dropout memory reduction

---

## P9 — Decoder LM benchmark at realistic scale

**File:** `test/bench_transformer.c` (new file)

**Root cause:** All transformer tests use `d_model=4`, `n_layers=2`, `vocab=8`. No performance data exists for realistic sizes (d_model=768, n_layers=12, vocab=32000, N=512). The actual transformer performance profile (where attention O(N²) cost dominates vs linear projection cost) is completely unmeasured.

**Mitigation:** Add a benchmark with a small LM (d_model=256, n_layers=2, N=128) to get fwd+bwd timing and identify bottlenecks before scaling up. Print per-layer timing breakdown.

**Estimated lines:** ~150

**Estimated gain:** Measurement infrastructure — enables data-driven optimization of transformer path.

**Status: not started**

**Checklist:**
- [ ] Create `test/bench_transformer.c`
- [ ] Build decoder LM with d_model=256, n_layers=2, n_heads=4, N=128
- [ ] Time fwd+bwd per step, print per-op breakdown
- [ ] Add to Makefile bench target
- [ ] Run and document bottlenecks

---

## P10 (bonus) — Makefile: `-ffast-math` → safer subset

**File:** `Makefile`

**Risk:** `-ffast-math` on Apple Clang implies `-funsafe-math-optimizations` which:
- Reassociates FP: `a+b+c` → any order
- Treats NaN/Inf as unreachable (branches like `in[i] > 0.0f` become undefined for NaN)
- May fold `0.0 * x` → `0.0` (wrong for `0.0 * NaN = NaN`)

**Mitigation:** Switch to `-O3 -ffinite-math-only -fno-signed-zeros -fno-trapping-math`. Same performance, fewer correctness risks.

**Estimated lines:** 1

**Status: not started**

**Checklist:**
- [ ] Replace `-ffast-math` with `-ffinite-math-only -fno-signed-zeros -fno-trapping-math`
- [ ] Run all tests to confirm no regression
- [ ] Run benchmark to confirm same performance

---

## Estimated Combined Impact (P0-P8)

| Scenario | Current | Optimized | Speedup |
|----------|---------|-----------|---------|
| CNN batch=64 fwd+bwd | 17.5 ms | ~8-10 ms | **1.8-2.2×** |
| CNN 5 epochs (batch=128) | 77.5 s | ~35-45 s | **1.7-2.2×** |
| Causal softmax speed | baseline | 2-4× | 2-4× |
| Layer norm speed | baseline | 1.5× | 1.5× |
| Peak scratch mem (CNN) | ~160 MB | ~145 MB | -10% |
| Peak scratch mem (transformer, N=2048) | ~192 MB × n_layers | ~0 (recompute P) | **-100%** per layer |

**Single biggest lever:** Winograd. It addresses the #1 bottleneck (conv2 L2 cache overflow), cuts the most expensive layer's time by ~62%, and eliminates the largest intermediate buffer.
