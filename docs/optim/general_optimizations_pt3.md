# General Optimizations Pt3 — Remaining Hot Paths & Architecture Gaps

Corrections to first-pass analysis after full code audit.
Measured baseline: Apple M1, `-O3 -ffinite-math-only`.

---

## Corrections to First-Pass Assumptions

| First-pass claim | Reality after code audit |
|-|-|
| "No NEON for element-wise ops" | **Partially wrong.** Contiguous non-broadcast paths are auto-vectorized by compiler at `-O3`. NEON intrinsics used for activations/softmax/layernorm/pool where compiler cannot auto-vectorize (online softmax, sigmoid/SiLU polynomial, horizontal reductions). |
| "NO_CBLAS triple loop is a production blocker" | **Overstated.** NO_CBLAS is never set in the build — Accelerate is linked unconditionally on macOS. The fallback exists for hypothetical non-BLAS builds. Adding a tiled fallback is still good engineering but not urgent. |
| "No GELU" | **Confirmed gap.** Only ReLU, SiLU, sigmoid, tanh, SwiGLU present. |
| "No RMS Norm" | **Confirmed gap.** Only LayerNorm present. |
| "No AVX2 fallback" | **Confirmed.** simd.h is pure NEON + scalar fallback. |
| "No FlashAttention" | **Confirmed.** Attention stores full [B, H, N, N] P matrix for backward. |
| "No NEON for tanh" | **Half wrong** — tanh is used in `tensor_tanh` via `tanhf()` which is libm on M1. But the tanh approximation via `expf(-2x)` formulation could benefit from `simd_expf_f32`. The function `tensor_tanh` exists in ops_activation.c. Let me check if it has NEON simd path. *After reading the code:* yes, `tensor_tanh` uses scalar `tanhf` libm call, no NEON intrinsic. |
| "No NEON for reduce ops" | **Partially wrong.** `simd_reduce_max_f32` and `simd_reduce_sum_f32` exist in simd.h but are NOT used by the public tensor ops. `tensor_sum` uses its own stride-based contiguous path in ops_reduce.c which is not explicitly NEON. `tensor_mean` delegates to `tensor_sum + tensor_mul`. |
| "No NEON for dropout" | **Confirmed after audit.** Dropout forward/backward use contiguous flat loops but no NEON intrinsics. However dropout is not compute-bound (just RNG + mask). |
| "No NEON for embedding" | **Correct but irrelevant.** Embedding is memory-bound (`memcpy` rows). NEON won't help. |

---

## P0 — GELU activation (GPT-2 / BERT compatibility)

**File:** `src/ops_activation.c`

**Why:** SiLU is present (used by Llama/Mistral). GELU is required for BERT/GPT-2 compatibility. The tanh-approximation matches PyTorch's `nn.GELU(approximate='tanh')`.

**Kernel (fwd):**
```
gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```
where `√(2/π) ≈ 0.7978845608028654`.

**Kernel (bwd):** Standard GELU derivative via `tanh` terms.

**NEON path:** Reuse `simd_expf_f32` for tanh: `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`. Or use the polynomial approximation directly.

**Estimated lines:** ~50 (fwd + bwd + NEON + autograd + dispatch in ops_activation.c + declaration in include/ops.h)

**Status:** not implemented

---

## P1 — RMS Norm (Llama/Mistral compatibility)

**File:** `src/norm.c` (new `tensor_rms_norm`)

**Why:** RMS Norm is the standard in modern LLMs. Compared to LayerNorm:
- No mean subtraction (saves 1 reduction per forward)
- `y = x * rsqrt(mean(x²) + eps) * γ`
- Backward is simpler: no mean gradient term

Backward formula:
```
y = x * rs * γ                  where rs = rsqrt(x² + eps)
                                where x² = mean(x²) over last dim
dy/dγ = rs * x
dx    = γ * rs * (1 - x² * rs² / d) * dy     — after some algebra
```

**Estimated lines:** ~70 (fwd + bwd + NEON for x² reduction and output write + autograd + header)

**NEON opportunity:** The `x²` reduction per slice is a sum of squares — vectorizable with `vfmaq_f32` + `vaddvq_f32`. The output write is `y = x * rs * γ` — element-wise multiply chain.

**Estimated gain:** ~30% fewer FLOPs than LayerNorm for the same d. Simpler backward.

**Status:** not implemented

---

## P2 — RoPE backward NEON SIMD

**File:** `src/rope.c` (`_rope_apply_grad`)

**Why:** The inner loop applies a 2×2 rotation matrix to each pair `(2k, 2k+1)`:
```c
grad[2k]     += ge * cos + go * sin
grad[2k+1]   += -ge * sin + go * cos
```
Each pair is independent and contiguous — perfect for 4-wide NEON:
- Load `ge, go, ge_next, go_next` into one `float32x4_t`
- Load `cos[k], cos[k+1]` and `sin[k], sin[k+1]`
- Compute with `vfmaq_f32`, `vmulq_f32`, interleave pairs

But: each element of `ge` and `go` is adjacent in memory (`g_row[2k]` and `g_row[2k+1]`). To load 4 elements (2 pairs) via NEON, we'd do:
```c
float32x4_t g4 = vld1q_f32(g_row + 2*k);
// g4 = [ge_pair0, go_pair0, ge_pair1, go_pair1]
```
Then reshape/extract to process pairs. The NEON intrinsics for this are somewhat involved (lots of `vget_low_f32`/`vget_high_f32`/`vcombine_f32` shuffles). Gain is ~2× on the inner loop.

**Estimated lines:** ~20 (NEON path in `_rope_apply` and `_rope_apply_grad`)

**Gain:** 2× on RoPE backward kernel. RoPE is ~1-2% of total transformer step time at d_model=256 (benchmarks needed to confirm). Low ROI unless running many RoPE-heavy layers.

**Status:** not implemented

---

## P3 — Direct 1×1 conv path (skip im2col entirely)

**File:** `src/conv.c` (dispatch in `tensor_conv2d`)

**Why:** For kH=kW=1, im2col is a no-op copy: input [N, C, H, W] reshapes to col [C, N*H*W] with zero transformations. The entire im2col pass (memset + loop) is wasted work.

**Mitigation:** When `kH==1 && kW==1`:
- Forward: reshape input to [N*H*W, C], call `cblas_sgemm` directly with weight [out_C, C]
- No col allocation
- Backward: reshape grad_output to [N*H*W, out_C], use standard sgemm grads
- d_bias: same as im2col path (sum over spatial dims)

**Edge case:** When the 1×1 layer is preceded by a non-contiguous operation, the reshape requires a contiguous view. Assert contiguous or copy.

**Estimated lines:** ~30 (dispatch in both forward + backward)

**Gain:** Saves im2col memset + copy for 1×1 layers. For a typical conv net with 1×1 layers (bottleneck blocks), this is ~O(HWC) per 1×1 layer. With MNIST CNN there are no 1×1 layers. Benefit appears in modern architectures (ResNet bottleneck, EfficientNet).

**Status:** not implemented

---

## P4 — Depthwise separable convolution

**File:** `src/conv.c` (new `tensor_depthwise_conv2d`)

**Why:** Depthwise conv (groups=in_channels, out_channels=groups*multiplier) is the foundation of MobileNet / EfficientNet. Each input channel has its own filter — no cross-channel mixing in the depthwise pass. This is a distinct operation from group conv.

**Forward:** Each channel c ∈ [0, C) does a 2D conv of x[:, c, :, :] with weight[c, :, :, :] → output[:, c*mult:(c+1)*mult, :, :]. No GEMM needed for the depthwise pass — it's C independent 2D convolutions. For small k (3×3), direct conv with NEON is faster than im2col.

**NEON opportunity:** For k=3, unrolled 3×3 kernel: 9 MACs per output position. Load 3 rows of 3 floats with NEON, compute all 9 products, accumulate. No im2col overhead.

**Backward:** Mirror structure: per-channel independent backward.

**Estimated lines:** ~120 (fwd + bwd + NEON 3×3 kernel + dispatch)

**Gain:** Enables MobileNet-style architectures. Not a speed gain for existing models but unlocks new model families.

**Status:** not implemented

---

## P5 — Small-matmul kernel for QKV projections

**File:** `src/ops_matrix.c` or new `src/sgemm_small.c`

**Why:** The decoder LM's QKV projection matmuls are small:
- Input: [B=1, N=128, d_model=256] → QKV weight: [256, 768] → output: [1, 128, 768]
- Per-head attention matmuls: [1, 4, 128, 64] × [1, 4, 64, 128] → [1, 4, 128, 128]
- These are ~25M flops and ~8M flops respectively

Apple's Accelerate cblas_sgemm for these sizes has significant call overhead and thread dispatch cost (~10-30µs overhead vs ~50-150µs compute). A hand-tuned micro-kernel with 4×4 or 6×4 tiles, loop-unrolled, would:
- Avoid BLAS dispatch overhead
- Run single-threaded (no OMP overhead for tiny matrices)
- Use NEON `fmla` directly for 4-wide FMA

**Design:** `sgemm_small(int M, int N, int K, float *A, float *B, float *C, uint8_t flags)` — specialized for M,N,K ≤ 256. 4×4 tile in NEON registers:
```c
float32x4_t c[4] = {vdupq_n_f32(0)};
for (int k = 0; k < K; k++) {
    float32x4_t b = vld1q_f32(B + j + k*N);  // N contiguous along j dim? depends on layout
    c[0] = vfmaq_laneq_f32(c[0], b, A[i+0][k], 0);
    ...
}
```

The actual layout depends on whether QKV weight is contiguous row-major (which it is). The inner loop loads 1 A-element broadcasted × 4 B-elements → FMA.

**Estimated lines:** ~80 (micro-kernel + dispatch + header)

**Gain:** Estimated 1.5-2× on QKV projection matmuls. For transformer forward with d_model=256, B=1, N=128: QKV matmul ~50µs with BLAS, ~25-30µs with custom micro-kernel. Total step time improvement ~5-10%.

**Status:** not implemented

---

## P6 — AVX2 fallback for simd.h

**File:** `src/simd.h`

**Why:** The library is currently Apple Silicon-only for SIMD. Adding AVX2 paths makes it portable to x86 machines with GCC/Clang. Functions need AVX2 equivalents:
- `simd_expf_f32` → `_mm256_exp_ps` (or same polynomial with `_mm256_*`)
- `simd_relu_fwd` → `_mm256_max_ps`
- `simd_sigmoid_fwd` → same polynomial
- `simd_silu_fwd/bwd` → same
- `simd_swiglu_fwd/bwd` → same
- `simd_ce_bwd_row_kernel` → `_mm256_*` equivalents
- `simd_reduce_max_f32` / `simd_reduce_sum_f32` → `_mm256_*`

**Precision note:** The fast `simd_expf_f32` polynomial works on AVX2 too — it's pure arithmetic, no NEON-specific intrinsics. Just change `float32x4_t` → `__m256`, `vdupq_n_f32` → `_mm256_set1_ps`, `vfmaq_f32` → `_mm256_fmadd_ps`, etc.

**Estimated lines:** ~120 (parallel NEON/AVX2 implementations in same header)

**Gain:** Enables x86 builds. Without this, x86 falls back to pure scalar loops on all activation ops.

**Status:** not implemented

---

## P7 — NO_CBLAS tiled matmul fallback

**File:** `src/ops_matrix_int.h` (new internal header)

**Why:** The NO_CBLAS fallback in `tensor_matmul` and `tensor_matmul_add` is a naive triple loop:
```c
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            C[i*N + j] += A[i*K + k] * B[k*N + j];
```
This achieves ~1-2% of peak FLOPS on modern CPUs. A simple 4×4 or 6×4 tiled version with NEON/SSE `fmla` achieves 40-60% of peak.

**Design:** 4×4 tile kernel (16 FMAs per 4×4×4 loop nest iteration). Outer loops over M and N with tile-size steps. Remainder handling for edge tiles.
```
for i in 0..M step 4:
  for j in 0..N step 4:
    C_tile[4][4] = {0}
    for k in 0..K step 4:
      C_tile += A_tile[4][4] @ B_tile[4][4]
    C[i:i+4, j:j+4] += C_tile
```

Remove NO_CBLAS dependency from all BLAS calls in the library (conv.c, attention.c, ops_matrix.c, transformer.c).

**Estimated lines:** ~60 (tile kernel + pack + dispatch in new internal file)

**Gain:** 10-50× on non-BLAS builds. Zero effect on macOS (already uses Accelerate). Enables Linux builds without cblas dependency.

**Status:** not implemented

---

## P8 — FlashAttention / memory-efficient attention backward

**File:** `src/attention.c`

**Why:** Current `tensor_attention` saves the full softmax output P of shape [B, H, N, N] in the scratch pool for backward. At N=2048, B=2, H=4: 2×4×2048×2048×4 = 128 MB per attention layer. With 12 layers that's 1.5 GB. At N=8192 it's ~2 GB per layer.

FlashAttention recomputes the softmax statistics (max, sum_exp) from Q and K in the backward pass, avoiding the O(N²) materialization. The tradeoff: ~2× more FLOPs in backward (recompute S = QK^T) but O(N·d) memory instead of O(N²).

**Forward remains unchanged** — QK^T is still computed and softmax applied. The difference is in what's saved for backward:
- Current: save P [B, H, N, N] — 4 bytes per element
- Flash: save per-row max and sum_exp [B, H, N, 2] — 8 bytes per row

**Backward adaptation:** Recompute `P = softmax(S)` from saved max/sum_exp + recomputed S = QK^T. Then proceed with existing `attention_backward` logic.

**Estimated lines:** ~150 (modified forward to save stats instead of P, modified backward to recompute P)

**Gain:** Memory: O(N²) → O(N·d) for saved tensors. At d=64 and N=2048: 128 MB → 512 KB per layer. Enables long-context training (N=16K+) within fixed scratch pool budget.

**Status:** not implemented

---

## Summary Table

| Item | Effort | Gain | Dependency | Status |
|------|--------|------|-----------|--------|
| P0 — GELU | 50 lines | New model support | None | not impl |
| P1 — RMS Norm | 70 lines | 30% fewer FLOPs vs LN; Llama compat | None | not impl |
| P2 — RoPE NEON | 20 lines | 2× on RoPE kernel | None | not impl |
| P3 — Direct 1×1 conv | 30 lines | Skip im2col for 1×1 | None | not impl |
| P4 — Depthwise conv | 120 lines | MobileNet arch support | None | not impl |
| P5 — Small matmul kernel | 80 lines | 1.5-2× on QKV proj | None | not impl |
| P6 — AVX2 fallback | 120 lines | x86 portability | None | not impl |
| P7 — Tiled NO_CBLAS sgemm | 60 lines | Linux builds | None | not impl |
| P8 — FlashAttention | 150 lines | Long-context training (N>2048) | None | not impl |

**Recommended order:** P1 (RMS Norm) → P0 (GELU) → P3 (1×1 conv) → P6 (AVX2) → P5 (small matmul) → P7 (tiled sgemm) → P8 (FlashAttention) → P2 (RoPE NEON) → P4 (depthwise conv)

The first three (RMS Norm, GELU, 1×1 conv) are small, focused additions that expand model compatibility with minimal risk. AVX2 unlocks x86 builds. Small matmul and FlashAttention target the transformer training bottleneck. RoPE NEON and depthwise conv are lower priority (narrow use case).
