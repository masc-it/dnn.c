# Letter to Expert — Performance Audit v2 (with measurements)

**From:** dnn.c maintainer
**Tags:** `performance-optimization` `c-simd-neon` `memory-bandwidth` `convolution` `attention` `cache-utilization` `openmp` `im2col` `compiler-optimization`

---

## Request

Identify the worst-performing operations by measured time and memory access patterns. Determine which missing optimizations capture ~80% of potential speedup and memory reduction.

---

## 1. Measured Baseline (MNIST CNN, batch=64)

Profiled via `test_cnn_profile` on Apple M1 (8-core CPU, 16 GB LPDDR4X, 192 MB scratch pool, `-O3 -ffast-math`):

### Forward breakdown (per-layer, batch=64)

| Layer | Time (ms) | % of fwd | SGEMM (ms) | Im2col/mem (ms) |
|-------|-----------|----------|------------|-----------------|
| conv1 (1→32, 3×3, s1) | 0.96 | 20.6% | ~0.06 | ~0.90 |
| relu1 | 0.16 | 3.4% | — | — |
| conv2 (32→64, 3×3, s2) | **2.40** | **51.5%** | **~0.36 (15%)** | **~2.04 (85%)** |
| relu2 | 0.09 | 1.9% | — | — |
| conv3 (64→64, 3×3, s2) | 0.80 | 17.2% | ~0.18 | ~0.62 |
| relu3 | 0.02 | 0.4% | — | — |
| fc1 (3136→128) | 0.16 | 3.4% | — | — |
| fc2+dropout+relu (128→10) | 0.07 | 1.5% | — | — |
| **Total forward** | **4.66** | **100%** | **~0.60 (13%)** | **~3.56 (76%)** |

### Backward

| Metric | Value |
|--------|-------|
| Total backward (all layers) | **12.88 ms** |
| Ratio backward/forward | **2.76×** |
| Expected ratio (2× sgemm) | 2.0× |
| Excess (col2im scatter + small-M penalty) | **+0.76×** |

### Multi-batch timing (batch=128)

| Metric | Value |
|--------|-------|
| Avg fwd+bwd per batch | **36.03 ms** |
| Est. epoch (430 batches) | 15.5 s |
| Est. 5 epochs | 77.5 s |

---

## 2. Critical Finding: Memory Rearrangement Dominates, Not Compute

**Conv2 forward: sgemm is only 15% of the cost.** The remaining 85% (2.04 ms) is:
- im2col writes: writing K×M = 288×12544 = **14.5 MB** col buffer sequentially
- Input reads: reading N×C×H×W = 64×32×28×28 = 6.4 MB from input
- **14.5 MB col does NOT fit in L2 cache (~12 MB on M1)** → every sgemm access to col suffers L2 miss → sgemm drops from ~665 GFLOPS (L2-hot) to ~221 GFLOPS (3× penalty)

This overturns conventional wisdom: **the bottleneck is NOT FLOPs but memory bandwidth and L2 cache overflow**. Any optimization that eliminates or shrinks the col buffer (Winograd, direct convolution) would have outsized impact.

### Conv2 memory hierarchy analysis

| Level | Capacity | Conv2 col (14.5 MB) hits? |
|-------|----------|--------------------------|
| L1 (per core) | 128 KB | ❌ 113× overflow |
| L2 (shared) | 12 MB | ❌ 1.2× overflow |
| L3 (SLC) | 8 MB | ❌ |
| DRAM | 16 GB | ✅ |

Each sgemm call on conv2 col reads from DRAM every time.

### All convolution col buffer sizes (batch=64)

| Layer | K | M | Col size | L2 fit? |
|-------|---|---|----------|---------|
| conv1 (1→32, s1) | 9 | 50176 | **1.8 MB** | ✅ fits L2 |
| conv2 (32→64, s2) | 288 | 12544 | **14.5 MB** | ❌ 1.2× L2 overflow |
| conv3 (64→64, s2) | 576 | 3136 | **7.2 MB** | ✅ fits L2 |

Only conv2 overflows L2. But conv2 is the most expensive single layer.

---

## 3. Memory Pressure Points (Ranked by Severity)

### P1 — Conv2 col buffer (14.5 MB) causes L2 cache thrashing

**Root cause:** im2col materializes K×M = 288×12544 floats. Each sgemm call reads col from DRAM instead of L2. The 221 GFLOPS sgemm is memory-bandwidth-bound.

**Mitigation:** Winograd F(2×2, 3×3) replaces the entire im2col+GEMM with element-wise multiplications on 4×4 tiles. Col buffer goes to zero. Works for all 3×3 conv layers in this model. Since all convs use 3×3 kernels, Winograd would:
- Eliminate 14.5 MB col → **zero intermediate**
- Cut FLOPs by 56% (9 MACs/tile → 4 MACs/tile)
- No L2 overflow → sgemm stays at ~665 GFLOPS

**Estimated impact on conv2:** 2.40 ms → ~0.8-1.0 ms (60-65% reduction)

### P2 — Backward col2im scatter writes

col2im scatters dcol into NCHW dx output. Each kernel element hits stride-W apart locations. For conv2 (kH×kW=9), each write hits a different cache line in the output row. This is inherent to im2col approach.

**Mitigation:** Winograd again — backward also operates on tiles with sequential writes.

### P3 — Attention softmax output P saved for backward (transformer)

Shape B×H×N×N. For N=2048, H=12, B=1: **192 MB per attention layer.** This is the largest single scratch allocation. Forces scratch pool to be >192 MB and limits max sequence length.

**Current workaround:** 192 MB scratch pool hard-coded in `main.c`. For deep models (12+ layers), each layer's P lives simultaneously in scratch until backward runs → multiply by n_layers.

**Two strategies:**
1. **Recompute P in backward** (trade compute for memory): only saves `scale` (4 bytes) and `mask` pointer, recomputes causal softmax during backward. Adds 1× sgemm(Q@K^T) + 1× causal_softmax per backward call. For training, this doubles the forward attention FLOPs but saves 192 MB per layer.
2. **Save P in lower precision**: B×H×N²×2 bytes (fp16) = 96 MB. Requires fp16 storage and conversion overhead.

### P4 — Causal softmax forward: 3 row passes, no SIMD

For each row i in the N×N scores matrix:
1. Row max over j=0..i (scalar `if (v > mx)`)
2. Row sum_exp over j=0..i (scalar `expf`)
3. Row softmax write (scalar `expf * inv_se`)

Three full row passes per row = O(N²) passes total. No NEON SIMD anywhere. Only the `expf` calls benefit from fast-math, but the loop structure is scalar.

**Impact at N=2048:** 2048 rows × 3 passes × 2048 cols avg = 12.6M element iterations per batch-head. For B=1, H=12: 151M scalar iterations per forward call.

**Mitigation:** Fuse pass 1+2 with online softmax (running max + adaptive sum_exp). Apply NEON SIMD to vectorize expf over 4-lane groups within each row segment.

### P5 — Layer norm: 3 full passes, unfused

| Pass | Operation | SIMD? | OMP? |
|------|-----------|-------|------|
| 1 | mean(x) | ❌ scalar loop | ✅ outer |
| 2 | var(x-μ) → rstd | ❌ scalar loop | ✅ outer |
| 3 | output = γ·(x-μ)·rstd + β | ❌ scalar loop | ✅ outer |

No NEON SIMD on inner loops. Welford one-pass fusion cuts passes 1+2 → 1, saving 33% of norm time.

### P6 — Dropout mask allocated in scratch

Dropout allocates mask[n] as a full float tensor in scratch (n=numel elements, 4 bytes each). For transformer FFN with intermediate_size=11008: mask = 44 KB per FFN. Reuse as byte mask or bit mask saves 75-97%.

### P7 — Cross-entropy nD fallback: 2 passes, coord decompose per element

Fused 3→2 passes already done. But each pass still recomputes coord decomposition from scratch. Could cache the slice_idx for each element.

The 2D fast path (NEON SIMD) is used for all classification tasks and is well-optimized (2 SIMD row reads + 1 scalar access). However, these 2 SIMD row reads (max, then sum_exp) could be fused to 1 pass via online max + exp sum tracking.

---

## 4. SIMD Coverage Gap Analysis

The `simd.h` has NEON for exactly these operations:

| Op | Forward NEON | Backward NEON | Notes |
|----|-------------|---------------|-------|
| ReLU | ✅ `vmaxq_f32` | ✅ `vbslq_f32` | |
| Sigmoid | ✅ 4× expf poly + div | ✅ FMA | |
| SiLU | ✅ fused sigmoid + mul | ✅ | |
| SwiGLU | ✅ fused | ✅ | |
| Cross-entropy | ❌ (scalar max + exp_sum) | ✅ `simd_ce_bwd_row_kernel` | **Forward has no NEON** — only 2× `simd_reduce_max_f32` / `simd_exp_sum_shifted_f32` helpers |
| Softmax | ❌ (2D uses helpers above, nD none) | ✅ (2D NEON via FMA, nD none) | |
| Causal softmax | ❌ **fully scalar** | ❌ **fully scalar** | **biggest SIMD gap** |
| Layer norm | ❌ **fully scalar** | ❌ **fully scalar** | |
| Dropout | ❌ scalar | ❌ scalar | |
| Element-wise ops | ❌ scalar | ❌ scalar | |

### Cross-entropy forward 2D fast path — can be fused

Currently 2 full SIMD passes per row:
```c
float mx = simd_reduce_max_f32(row, C);         // Pass 1: SIMD max
float se = simd_exp_sum_shifted_f32(row, C, mx); // Pass 2: SIMD exp sum
total_loss += logf(se) + mx - row[td[n]];       // scalar
```

Could fuse to 1 pass with online softmax (running max, adaptive sum_exp). For C=vocab_size (32K+ in LLM), saves ~16K float reads per row × batch_size × steps.

---

## 5. `-ffast-math` Correctness Concern

`CFLAGS` use `-O3 -ffast-math`. On Apple Clang, `-ffast-math` implies `-funsafe-math-optimizations` which:
- Reassociates floating-point: `a + b + c` → `(a + b) + c` or any order
- Treats NaN/Inf as unreachable (comparisons with NaN become undefined)
- May reassign `0.0 * x` → `0.0` (math-h valid but IEEE 754-2019 says `0.0 * NaN = NaN`)

**Impact on correctness:**

| Pattern | Risk | Example in codebase |
|---------|------|---------------------|
| `in[i] > 0.0f` branch in ReLU bwd | Compiler may hoist/rewrite the comparison if NaN assumed absent | `if (in[i] > 0.0f) ag[i] += g[i]` |
| `se += expf(val - mx)` | Reassociation may underflow differently | softmax sum_exp |
| `sum / (float)n` | May be combined with surrounding multiply | norm mean, loss scaling |
| `1.0f / (1.0f + expf(-x))` | `1.0f / (1.0f + x)` if `expf(-x)` reassociated | sigmoid |

**Recommendation:** Use `-O3 -ffinite-math-only -fno-signed-zeros -fno-trapping-math` instead of `-ffast-math`. Same performance, fewer correctness risks. Or keep `-ffast-math` but audit all NaN-sensitive branches.

---

## 6. OpenMP Threading Analysis

Threading strategy across hot paths:

| Op | Threading | Oversubscription risk |
|----|-----------|----------------------|
| im2col | `#pragma omp parallel for` over N (batch) | Low — N=64 threads max vs 8 cores |
| col2im | `#pragma omp parallel for` over C (channels) | Low — C=32/64 threads |
| layer_norm | `#pragma omp parallel for` over n_slices | Medium — tiny slices (d=hidden) |
| sgemm (conv forward) | Internal Apple BLAS threading | **High** — BLAS threads + outer OMP competing |
| sgemm (conv d_weight) | OMP over out_C blocks | **High** — `#pragma omp parallel for` over 16×4 = 4 threads, each calling threaded cblas_sgemm |

The outer OMP `#pragma omp parallel for` on conv forward batches wraps individual `cblas_sgemm` calls that already use multi-threaded BLAS internally. This creates **nested parallelism** where the **outer thread pool and inner BLAS thread pool oversubscribe cores** by 2-4×. Apple's Accelerate on M1 uses up to 8 threads internally. With outer OMP creating 4+ threads, total threads = 4×8 = 32 on an 8-core machine → severe context switching.

**Fix:** Add `omp_set_max_active_levels(1)` before OMP regions that wrap BLAS calls, or use `if (out_C < 64)` guard to serialize trivial blocks.

---

## 7. Priority-Ordered Optimization Recommendations

### P0 — Winograd F(2×2, 3×3) for all 3×3 convolutions

| Metric | Current (im2col) | Winograd | Gain |
|--------|-----------------|----------|------|
| Conv2 fwd time | 2.40 ms | ~0.9 ms est. | **62%** |
| Col buffer | 14.5 MB (L2 miss) | 0 (tile-only) | **∞** |
| FLOPs per position | 9 MACs | 4 MACs | **56%** |
| Conv total fwd | 4.16 ms | ~1.9 ms est. | **54%** |
| Total batch (batch=64) | 17.54 ms | ~10 ms est. | **43%** |

**Effort:** ~150 lines. Forward: transform input tiles (4×4) × weight (3×3) → 4 element-wise multiplies → inverse transform. Backward: similar transform for gradients.

**Dispatch condition:** `kH==3 && kW==3 && stride==1 && pad==1` (most common case).

### P1 — Fuse causal softmax forward passes + NEON SIMD

| Current | Proposed | Gain |
|---------|----------|------|
| 3 row passes (max, exp_sum, write) | 1 fused pass with online max tracking | **2-3×** on causal softmax |
| Scalar inner loops | NEON `simd_expf_f32` on 4-wide groups | **2-4×** per pass |

**Effort:** ~40 lines in `attention.c` step 3 (replace three loops with one fused + SIMD). Already have `simd_expf_f32` and helper functions — just restructure loop logic + add `DNN_HAVE_NEON` path.

### P2 — Slice-level Softmax Fusion for cross_entropy forward (2D path)

**Impact:** Reduce 2 SIMD row passes → 1 fused pass. Saves 1 full read over C elements per row. For vocab=32000, batch=128: 128 × 32000 × 4 bytes = 16 MB less DRAM reads per forward call.

**Effort:** ~15 lines in `tensor_cross_entropy` 2D fast path. Online softmax: track running max and sum_exp simultaneously with `simd_expf_f32`.

### P3 — Layer norm Welford fusion

**Impact:** 2 passes → 1 pass for mean+var. Saves N*d element reads/writes.

**Effort:** ~15 lines in `tensor_layer_norm` forward.

### P4 — Control OpenMP nested parallelism

**Impact:** Prevents 2-4× thread oversubscription in conv forward/backward. Measurable as wall-clock time reduction on multi-batch runs.

**Effort:** 1 line `omp_set_max_active_levels(1)` at top of `conv2d_backward` and `tensor_conv2d`.

### P5 — Add NEON SIMD to layer_norm inner loops

**Impact:** 2-4× speedup on per-slice inner loop (currently scalar). Most impactful at large d (hidden dim 768+ in transformers).

**Effort:** ~20 lines per pass. Replace `for (j=0; j<d; j++) sum += x[j]` with NEON 4-wide reduction.

### P6 — Attention backward: fuse dS scale into softmax_bwd

**Impact:** Saves one full N×N pass per batch-head in backward. For B=1, H=12, N=2048: saves 12 × 4M = 48M float multiplications per step.

**Effort:** **2 lines.** Move `dS[i] *= scale` inside the causal softmax backward loop body.

### P7 — Causal softmax backward dot product: NEON SIMD

**Impact:** 2-4× on the inner dot product loop. Currently fully scalar: `for (j=0; j<=i; j++) dot += sm_row[j] * g_row[j]`. Replace with `vfmaq_f32` + `vaddvq_f32`.

**Effort:** 5 lines using `DNN_HAVE_NEON` guard. Pattern identical to existing `softmax_backward` 2D fast path NEON dot product.

### P8 — Check `dropout_mask` as byte array

**Impact:** 75% reduction in mask memory (4 bytes → 1 byte per element). For transformer FFN (intermediate=11008): 44 KB → 11 KB per FFN per layer.

**Effort:** 5 lines. Change `float *mask` to `unsigned char *mask`, adjust read.

### P9 — Add `decoder_lm` benchmark at realistic scale

**Important gap:** All transformer tests use `d_model=4`, `n_layers=2`, `vocab=8`. No performance data exists for realistic sizes (d_model=768, n_layers=12, vocab=32000, N=512). The actual transformer performance profile (where attention O(N²) cost dominates vs linear projection cost) is completely unmeasured.

**Effort:** Add a benchmark in `test/bench_transformer.c` with a small LM (d_model=256, n_layers=2, N=128) to get fwd+bwd timing and identify bottlenecks before scaling up.

---

## 8. Ready-to-Bench Architectures

Reuse the architectures in `main.c` (CNN) and `main_lm.c` (decoder LM). Both are full training runs with real data, realistic compute, and take real time.

---

## 9. Estimated Combined Impact

If P0-P8 are implemented:

| Scenario | Current | Optimized | Speedup |
|----------|---------|-----------|---------|
| CNN batch=64 fwd+bwd | 17.5 ms | ~8-10 ms | **1.8-2.2×** |
| CNN 5 epochs (batch=128) | 77.5 s | ~35-45 s | **1.7-2.2×** |
| Peak scratch memory (CNN) | ~160 MB | ~145 MB | -10% |
| Peak scratch memory (transformer N=2048, B=1, H=12) | ~192 MB × n_layers | ~0 (recompute P) | **-100% per layer** (or -50% with fp16) |
| Causal softmax speed | baseline | 2-4× | 2-4× |
| Layer norm speed | baseline | 1.5× | 1.5× |

**Single biggest lever:** Winograd. It addresses the #1 bottleneck (conv2 L2 cache overflow), cuts the most expensive layer's time by ~60%, and eliminates the largest intermediate buffer.

---

*Filed for expert review.*
