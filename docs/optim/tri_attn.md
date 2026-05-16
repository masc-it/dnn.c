# Triangular Causal Attention Plan

Goal: stop paying dense `N×N` attention cost for causal masks. Current `src/attention.c` computes future-token scores/output that softmax later masks to zero. For long GPT contexts this wastes ~50% of attention FLOPs and creates large per-worker `N×N` scratch buffers.

Target implementation: **row-blocked triangular causal attention** using BLAS on rectangular prefixes.

Scope:
- Optimize `tensor_attention()` training/no-grad full-sequence path.
- Preserve public API and numerical behavior.
- Keep saved full `P` in first training implementation, so risk stays low.
- Defer full FlashAttention/recompute mode until triangular compute is correct and benchmarked.
- Leave `transformer_block_forward_cached()` alone first; cached generation already computes prefix attention for `N_new` tokens.

Non-goals for first patch:
- No mask gradient support. Current backward ignores mask grads too.
- No 2D/3D attention support unless colleague chooses to restore header promise.
- No row-tile parallelism; it creates overlapping `dK`/`dV` writes.

---

## Current hot path

Files:
- `src/attention.c`
- `src/gpt.c`
- `src/transformer.c` cached-generation path already has separate one-token/prefix attention; leave it alone for first pass.

Current `tensor_attention()` does this per `(B,H)` slice:

```c
scores = Q @ K^T              // full [N,N]
P      = causal_softmax(scores) // upper triangle becomes 0
O      = P @ V                // full [N,N] @ [N,d]
```

Backward is also dense:

```c
dV = P^T @ dO
dP = dO @ V^T
dS = causal_softmax_bwd(P, dP)
dQ = dS @ K
dK = dS^T @ Q
```

Problem: for causal attention, row `i` only uses columns `0..i`. Dense matmuls compute `j > i` anyway. That is roughly half of QK, PV, dP, dQ, dK, dV work wasted.

---

## Core idea

Process query rows in tiles. For tile rows `[r0, r1)`, only keys `[0, r1)` can be visible.

Example with tile size `TB=64`:

```c
for (r0 = 0; r0 < N; r0 += TB) {
    r1 = min(r0 + TB, N);
    M  = r1 - r0;   // query rows in tile
    S  = r1;        // visible key prefix length for this tile

    scores_tile = Q[r0:r1] @ K[0:S]^T;     // [M,S]
    mask scores_tile row-wise for j > global_i;
    p_tile = softmax(scores_tile);         // [M,S]
    O[r0:r1] = p_tile @ V[0:S];            // [M,d]
}
```

This still uses BLAS (`cblas_sgemm`) but skips later keys. Extra compute remains only inside each diagonal tile (`j > i` but `j < r1`) and is masked before softmax. Overhead ≈ `TB / N`; with `TB=64`, overhead is ~3% at `N=2048`.

Why this design instead of hand-written triangular microkernels:
- BLAS stays responsible for dense rectangular compute.
- Change is local to `src/attention.c`.
- Backward can mirror forward with same row tiles.
- Easy dense-reference comparison during rollout.

---

## Data layout assumptions

Current implementation asserts 4D contiguous inputs:

```c
q,k,v: [B,H,N,d]
```

For one `(b,h)` slice:

```c
q_slice = qd + bh * N * d;
k_slice = kd + bh * N * d;
v_slice = vd + bh * N * d;
o_slice = od + bh * N * d;
```

This mirrors current `src/attention.c` pointer math. It assumes contiguous tensors with offset behavior matching existing code; do not broaden tensor layout support in same patch.

Tile pointers:

```c
q_tile   = q_slice + r0 * d;  // [M,d], lda=d
k_prefix = k_slice;           // [S,d], lda=d
v_prefix = v_slice;           // [S,d], lda=d
o_tile   = o_slice + r0 * d;  // [M,d], ldc=d
```

Packed worker buffers:

```c
scores_tile: [TB,N] but active view [M,S], row stride S
p_tile:      [TB,N] but active view [M,S], row stride S
```

Do not pass saved full `P` directly to BLAS as `[M,S]`: saved `P` rows have stride `N`, not `S`. Pack row prefixes into `p_tile` in backward.

---

## Expected gain

Attention-only:
- Forward QK + PV: ~2× less FLOPs for long `N`
- Backward dV/dP/dQ/dK: ~2× less FLOPs for long `N`
- Scratch `scores_buf`: `workers * N*N` → `workers * TB*N`

End-to-end GPT:
- If attention dominates: likely **+50–90% throughput**
- If vocab projection dominates (`vocab=32k`, small `N`): less visible; LM head / cross-entropy becomes next bottleneck

Best cases:
- `N >= 512`
- many heads/layers
- moderate vocab or inference/no-grad path

Worst cases:
- tiny `N`
- huge vocab where `_lm_head_forward()` dominates

---

## Implementation phases

### Phase 1 — triangular forward, keep saved `P`

Change `tensor_attention()` in `src/attention.c`.

Current allocations:

```c
tensor *out = tensor_scratch(... [B,H,N,d]);
tensor *P   = tensor_scratch(... [B,H,N,N]);
float *scores_buf = workers * N * N;
```

New behavior:

1. Allocate `out` always.
2. Allocate `P` only when grad enabled and any of `q/k/v` needs grad.
3. Allocate per-worker tile buffers:
   - `scores_buf`: `workers * TB * N`
   - optional `p_buf`: `workers * TB * N` if not writing directly into packed temp
4. For each `(b,h)` and row tile:
   - compute `[M,S] = Q_tile @ K_prefix^T`
   - add additive mask if `mask != NULL`
   - apply causal mask inside tile (`j > r0 + mi`)
   - softmax each row over `0..global_i`
   - save `P` row prefix if training
   - compute `O_tile = P_tile @ V_prefix`

Pseudo-code:

```c
const int TB = 64; // tune 32/64/128
int needs_grad = dnn_grad_enabled() &&
    (tensor_requires_grad(q) || tensor_requires_grad(k) || tensor_requires_grad(v));

tensor *P = needs_grad ? tensor_scratch(scratch, 4, (int[]){B,H,N,N}, 0) : NULL;
float *scores_buf = _mem_pool_alloc(scratch, workers * TB * N * sizeof(float), NULL);
float *p_buf      = _mem_pool_alloc(scratch, workers * TB * N * sizeof(float), NULL);

#pragma omp parallel for collapse(2) if (B * H >= 2)
for b,h:
  for r0 in 0..N step TB:
    r1 = min(r0 + TB, N);
    M = r1 - r0;
    S = r1;

    scores = worker_scores;
    ptmp   = worker_p;

    cblas_sgemm(RowMajor, NoTrans, Trans,
                M, S, d, scale,
                q_slice + r0*d, d,
                k_slice, d,
                0.0f, scores, S);

    add_mask(scores, mask, b, h, r0, M, S);

    for mi in 0..M-1:
      i = r0 + mi;
      // softmax helper only reads visible prefix; future scores need not be -inf
      softmax only j=0..i;
      set ptmp[mi*S + j] = 0 for j=i+1..S-1;
      if (P) copy ptmp row prefix to P row;

    cblas_sgemm(RowMajor, NoTrans, NoTrans,
                M, d, S, 1.0f,
                ptmp, S,
                v_slice, d,
                0.0f, o_slice + r0*d, d);
```

Important: `P` full upper triangle can stay uninitialized if triangular backward never reads it. If keeping dense backward during transition, zero all upper entries.

Forward details that must be explicit in code:

```c
// Active dimensions per tile.
int M = r1 - r0;
int S = r1;

// scores/p buffers are allocated as TB*N, but BLAS sees stride S.
float *scores = worker_scores;
float *ptmp   = worker_p;

// After softmax row write, zero all future columns inside active prefix.
int visible = i + 1;
memset(ptmp + mi*S + visible, 0, (size_t)(S - visible) * sizeof(float));

// If training, save prefix into full P row with stride N.
if (P) {
    float *prow = p_slice + i * N;
    memcpy(prow, ptmp + mi*S, (size_t)S * sizeof(float));
    // Optional debug safety only:
    // memset(prow + S, 0, (size_t)(N - S) * sizeof(float));
}
```

Autograd allocation rule:

```c
int needs_grad = dnn_grad_enabled() &&
    (tensor_requires_grad(q) || tensor_requires_grad(k) || tensor_requires_grad(v));
```

Only create/save `P` and `grad_fn` when `needs_grad` is true. In no-grad/eval, `p_tile` is enough to compute `O_tile`.

Use `size_t` for allocation math:

```c
size_t tile_elems = (size_t)n_workers * DNN_ATTENTION_TILE_ROWS * N;
```

Do not use `int` byte counts for long-context configs.

### Phase 2 — triangular backward

Replace `attention_backward()` dense `N×N` buffers with row-blocked prefix buffers.

Per worker buffers:
- `p_tile`: `[TB,N]`
- `dP_tile`: `[TB,N]`
- `dS_tile`: `[TB,N]`

For each `(b,h)` and row tile `[r0,r1)`:

```c
M = r1 - r0;
S = r1;
pack p_tile from saved P rows, columns [0:S]

if (need_v)
    dV[0:S] += p_tile^T @ dO[r0:r1]

if (need_q || need_k) {
    dP_tile = dO[r0:r1] @ V[0:S]^T

    for each row mi:
        i = r0 + mi;
        dot = sum_{j=0..i} p_tile[mi,j] * dP_tile[mi,j]
        dS_tile[mi,j] = p_tile[mi,j] * (dP_tile[mi,j] - dot) * scale, j <= i
        dS_tile[mi,j] = 0, j > i && j < S

    if (need_q)
        dQ[r0:r1] += dS_tile @ K[0:S]

    if (need_k)
        dK[0:S] += dS_tile^T @ Q[r0:r1]
}
```

BLAS calls:

```c
// dV prefix update: [S,d] += [S,M] @ [M,d]
cblas_sgemm(RowMajor, Trans, NoTrans,
            S, d, M, 1.0f, p_tile, S, g_tile, d,
            1.0f, vg_prefix, d);

// dP tile: [M,S] = [M,d] @ [S,d]^T
cblas_sgemm(RowMajor, NoTrans, Trans,
            M, S, d, 1.0f, g_tile, d, v_prefix, d,
            0.0f, dP_tile, S);

// dQ tile: [M,d] += [M,S] @ [S,d]
cblas_sgemm(RowMajor, NoTrans, NoTrans,
            M, d, S, 1.0f, dS_tile, S, k_prefix, d,
            1.0f, qg_tile, d);

// dK prefix update: [S,d] += [S,M] @ [M,d]
cblas_sgemm(RowMajor, Trans, NoTrans,
            S, d, M, 1.0f, dS_tile, S, q_tile, d,
            1.0f, kg_prefix, d);
```

Threading rule: keep existing parallelism over `(B,H)`. Do **not** parallelize row tiles until race handling exists. `dK` and `dV` prefix updates overlap across row tiles, so multiple row-tile threads would race on same gradient rows.

Backward details:

```c
// Pack P rows from full stride N to compact stride S.
for (int mi = 0; mi < M; mi++) {
    int i = r0 + mi;
    memcpy(p_tile + mi*S, p_slice + i*N, (size_t)S * sizeof(float));
}
```

Gate work by needed gradients:

```c
if (vg) { /* dV */ }
if (qg || kg) { /* dP + dS */ }
if (qg) { /* dQ */ }
if (kg) { /* dK */ }
```

`dP_tile` is only needed for `qg || kg`. `dS_tile` is only needed for `qg || kg`. If only `v` needs grad, skip both.

Before any BLAS uses `dS_tile`, every element in active `[M,S]` must be written. For row `mi`, write `j <= i` from softmax-bwd and zero `j > i` through `S-1`.

Self-attention alias case (`q == k == v`) stays correct only if every gradient update is additive. Keep BLAS `beta=1.0f` for q/k/v grads and never overwrite existing grad buffer.

Mask note: Phase 1/2 backward does not need `mask`, because saved `P` already includes additive-mask effects. If Phase 4 recomputes `P`, recompute path must reapply additive mask before softmax.

### Phase 3 — no-grad fast path

When grad is disabled:
- do not allocate `P`
- do not write saved softmax matrix
- only keep tile `p_buf`

This directly benefits `decoder_lm_generate(... use_cache=0)` because `src/gpt.c` enters no-grad before generation. Cached generation mostly uses `src/transformer.c::transformer_block_forward_cached()` and is already prefix-shaped.

### Phase 4 — optional FlashAttention memory mode

After triangular compute lands, add memory-saving mode:
- save per-row `mx` and `sum_exp` instead of full `P`
- recompute `p_tile` in backward from `Q_tile @ K_prefix^T`

This changes saved memory from `B*H*N*N` floats to `B*H*N*2` floats, but backward recomputes QK. Treat as follow-up; do not mix with initial perf patch.

### Phase 5 — optional fully packed training `P`

Intermediate option before FlashAttention: save `P` in triangular packed form instead of full `[N,N]`.

Memory shape per `(B,H)`:

```c
N*(N+1)/2 floats
row i starts at i*(i+1)/2
```

This cuts saved `P` memory ~50% without recompute, but complicates indexing and tests. Do not do this in first patch unless full `P` memory blocks useful benchmarks.

---

## Autograd tape contract

Keep existing backward contract:

```c
fn->inputs[0] = q;
fn->inputs[1] = k;
fn->inputs[2] = v;
if (mask) fn->inputs[3] = mask;

fn->saved_tensors[0] = P;          // Phase 1/2: full [B,H,N,N]
fn->saved_tensors[1] = scale_saved;
fn->saved_tensors[2] = mask_flag;
```

`mask_flag` remains informational unless mask-gradient support is added. Backward should still ignore mask grad, matching current behavior.

If Phase 4 FlashAttention mode is later added, saved tensors change to stats instead of `P`; keep that under separate compile/runtime flag so Phase 1/2 can stay stable.

---

## Mask handling

Current API accepts additive `mask` broadcastable to `[B,H,N,N]`. Preserve it.

For tile `[r0,r1)`, only visit columns `0..S-1`:

```c
for mi in 0..M-1:
  i = r0 + mi;
  for j in 0..S-1:
    coord[4] = {b, h, i, j};
    scores[mi*S + j] += md[_bcast_off(mask, mask_ndim, coord)];
```

Then causal mask still applies:

```c
for j = i + 1; j < S; j++
    scores[mi*S + j] = -INFINITY;
```

Note: if `mask == NULL`, skip all broadcast-offset work.

Mask edge cases:
- Causal self-position means at least one visible token unless additive mask sets all visible logits to `-INFINITY`.
- If all visible logits are `-INFINITY`, current dense code would produce invalid softmax too; triangular path does not need new semantics.
- Additive mask should be applied before softmax and before writing `P`.
- Future positions (`j > i`) can be skipped instead of explicitly set to `-INFINITY`, as long as `p_tile`/saved `P` future entries are zero or never read.
- Fast path should branch outside loops: one `mask == NULL` implementation and one masked implementation, to avoid `_bcast_off()` cost in GPT common path.

---

## Softmax helpers to add

Avoid duplicating softmax logic. Add small static helpers in `src/attention.c`:

```c
static void attention_softmax_row_prefix(const float *scores,
                                         float *p,
                                         int visible_len);

static void attention_softmax_bwd_row_prefix(const float *p,
                                             const float *dp,
                                             float *ds,
                                             int visible_len,
                                             float scale);
```

Forward helper behavior:
- `visible_len = i + 1`
- compute online max + sum_exp over `scores[0:visible_len]`
- write `p[0:visible_len]`
- caller zeroes `p[visible_len:S]`

Backward helper behavior:
- compute `dot = sum p[j] * dp[j]` over visible prefix
- write `ds[j] = p[j] * (dp[j] - dot) * scale`
- caller zeroes `ds[visible_len:S]`

Use existing NEON patterns from current Step 3 and current causal softmax backward dot product.

---

## Tuning constants

Start with:

```c
#ifndef DNN_ATTENTION_TILE_ROWS
#define DNN_ATTENTION_TILE_ROWS 64
#endif
```

Benchmark `TB = 32, 64, 128`.

Tradeoff:
- smaller tile: less diagonal overcompute, worse BLAS efficiency
- larger tile: better BLAS efficiency, more diagonal overcompute

Likely winner on Apple Silicon: `64` or `128`.

Dispatch rule to benchmark:

```c
if (N <= DNN_ATTENTION_DENSE_THRESHOLD)
    use old dense path;
else
    use triangular path;
```

Start threshold at `0` while developing. Add threshold only if benchmark shows small-`N` regression.

Threading notes:
- Existing OpenMP over `(B,H)` can remain.
- More, smaller BLAS calls may amplify BLAS dispatch overhead.
- If `B*H` is small, compare OMP on/off. Existing `if (B * H >= 2)` may still be OK, but benchmark `>=4` or `>=8`.
- Avoid nested parallelism changes in first patch.

---

## Files to edit

### `src/attention.c`

Main work:
- add tile constant
- add row-prefix softmax helper
- change forward to row-blocked prefix matmuls
- change backward to row-blocked prefix matmuls
- allocate `P` only when needed
- shrink per-worker scratch from `N*N` to `TB*N`
- update `#if NO_CBLAS` fallback loops to match triangular dimensions (`M,S,d`) so non-BLAS builds still compile

Keep old dense implementation behind temporary debug flag if useful:

```c
#ifdef DNN_ATTENTION_DENSE_REF
// old implementation
#else
// triangular tiled implementation
#endif
```

### `include/attention.h`

Header is stale: it says 2D/3D/4D supported and mask is `[N,N]`. Current implementation asserts 4D contiguous inputs. Update doc to match reality or extend implementation deliberately. For this task, document current 4D path:

```c
Q,K,V: [B,H,N,d_head], contiguous
mask: NULL or broadcastable to [B,H,N,N]
causal mask is implicit
```

### `src/gpt.c`

No required API changes. Benefit flows through `decoder_lm_forward()` → `transformer_block_forward()` → `tensor_attention()`.

Small optional cleanup: no-cache generation already enters no-grad; after Phase 3 it avoids `P` allocation automatically.

### `bench/bench_transformer.c`

Verify bench still compiles. If stale field names break it, fix by collecting params through module API instead of direct old struct fields.

---

## Correctness tests

Run existing tests first:

```bash
make test_attention
make test_transformer
make test_decoder_lm_training
make test_generation
make test_generation_prefix
```

Add targeted attention regression if not already covered:

1. Compare triangular forward output against dense reference for:
   - `B=1,H=1,N=1,d=4`
   - `B=1,H=2,N=7,d=8`
   - `B=2,H=4,N=17,d=16`
   - `N` not divisible by tile size
   - `N < TB`, `N == TB`, `N == TB+1`
2. Compare gradients against dense reference / finite differences:
   - q grad
   - k grad
   - v grad
   - self-attention case `q == k == v`
   - only-q-grad, only-k-grad, only-v-grad if easy to isolate
3. Test with additive mask non-null:
   - exact `[N,N]` mask
   - broadcast `[1,1,N,N]` mask if tensor infra allows it
4. Test no-grad path output matches and does not create `grad_fn`.
5. Test deterministic equality with `B*H=1` and with parallel `(B,H)`.

Debug strategy:
- Keep dense implementation behind `DNN_ATTENTION_DENSE_REF` or helper function during development.
- In tests, run dense and triangular from same random tensors and compare outputs/grads.
- Remove debug flag later only after benchmark + tests are stable.

Numerical tolerance: expect small BLAS-order differences. Use existing tolerances unless failures are only ~1e-5.

---

## Benchmark plan

Add `bench/bench_attention.c` or extend existing benches.

Configs:

| B | H | N | d | Why |
|-:|-:|-:|-:|-----|
| 1 | 4 | 128  | 64  | small GPT debug |
| 2 | 8 | 512  | 64  | normal training |
| 1 | 12| 1024 | 64  | long context |
| 1 | 12| 2048 | 64  | target win |
| 1 | 8 | 4096 | 128 | stress |

Measure:
- forward no-grad
- forward+backward
- scratch peak if pool exposes it, or approximate bytes allocated
- speed with `TB=32/64/128`
- speed with dense threshold on/off

Use wall-clock timing (`clock_gettime(CLOCK_MONOTONIC, ...)`), not `clock()`, because OpenMP/BLAS CPU time can overcount multi-threaded work.

Report:

```text
N=2048,H=12,d=64
forward dense:      X ms
forward triangular: Y ms   speedup X/Y
step dense:         A ms
step triangular:    B ms   speedup A/B
```

Also run GPT-level bench:

```bash
make bench_transformer
```

Use at least one config where attention dominates and one where vocab dominates.

---

## Pitfalls

1. **BLAS leading dimensions**
   - `q_tile` rows are contiguous with stride `d`.
   - `k_prefix` rows are contiguous with stride `d`.
   - `p_tile` must be packed `[M,S]` contiguous for BLAS.
   - Full saved `P` rows have stride `N`; do not pass row prefixes as packed `[M,S]` unless copied.

2. **Gradient races**
   - Row tiles update overlapping `dK[0:S]` and `dV[0:S]`.
   - Parallelize `(B,H)` only for first implementation.
   - If later parallelizing tiles, use per-thread `dK/dV` accumulators or atomics/reduction; do not write shared grads directly.

3. **Self-attention aliasing**
   - Tests call `tensor_attention(ctx.scratch, x, x, x, NULL)`.
   - `_grad_ensure()` may return same grad buffer for q/k/v.
   - Accumulation must remain additive (`+=`, BLAS beta `1.0f`).

4. **Upper triangle initialization**
   - Triangular backward should never read `P[i,j]` for `j>i`.
   - If debug/dense fallback reads full `P`, upper triangle must be zeroed.

5. **Mask broadcast cost**
   - `_bcast_off()` inside nested tile loops is expensive.
   - Keep fast path for `mask == NULL` branch completely separate.

6. **Small `N` regressions**
   - For `N <= TB`, triangular method still computes one full tile (`N×N`) plus packing overhead.
   - Optional dispatch: use old dense path for `N <= 128` if benchmark says faster.

7. **Stride mismatch in saved `P`**
   - Saved full `P` uses row stride `N`.
   - Tile BLAS wants packed stride `S`.
   - Always pack in backward.

8. **`dS_tile` uninitialized tail**
   - BLAS consumes all active `[M,S]` values.
   - Zero `j > i` for every row before `dQ`/`dK` GEMMs.

9. **Threaded BLAS overhead**
   - Triangular path increases number of SGEMM calls by `ceil(N/TB)`.
   - Too-small `TB` can lose to call overhead even with fewer FLOPs.

10. **Pool pressure still includes `P` in training**
   - Phase 1/2 removes dense score/dP/dS scratch buffers but still saves full `P` for backward.
   - If scratch pool still overflows at long `N`, implement Phase 4 or Phase 5.

---

## Rollout checklist

1. Add dense reference guard or helper.
2. Implement triangular forward no-grad path first.
3. Compare forward outputs vs dense.
4. Enable training forward with saved full `P`.
5. Implement triangular backward.
6. Compare q/k/v grads vs dense/fd tests.
7. Run transformer/GPT tests.
8. Benchmark and tune `TB`/threshold.
9. Remove or hide debug ref before final handoff.

---

## Acceptance criteria

Patch complete when:

- `tensor_attention()` no longer allocates per-worker `N*N` scores.
- no-grad attention no longer allocates/saves full `P`.
- training backward uses triangular row tiles, not dense `N*N` dP/dS buffers.
- backward packs saved `P` prefixes correctly and never reads upper triangle.
- all attention/transformer/GPT tests pass.
- benchmark covers `TB=32/64/128` and reports chosen default.
- benchmark shows attention fwd+bwd speedup near 1.5–2× for `N>=1024`.
- GPT benchmark shows end-to-end gain and identifies remaining bottleneck if gain <50%.
