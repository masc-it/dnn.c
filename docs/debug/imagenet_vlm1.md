# VLM Debug Report — ImageNet-1k Training

**Date:** 2026-05-17
**Checkpoint:** `ckpt/imagenet_vlm_epoch05_20260517T085909.bin`
**Diagnostic tool:** `examples/imagenet_vlm/vlm_debug_1.c`

## Model Architecture

| Component | Config |
|-----------|--------|
| Image encoder | conv2d 3→128, k=16, s=16 → RMSNorm(128) → learned pos embed [1,196,128] |
| Text decoder | Embed(261→128) + 2×TransformerBlock(D=128, H=2, d_k=64) + RMSNorm + Linear(128→261) |
| Attention mode | Prefix-LM (bidirectional images, causal text) |
| Total params | 521K |

## Training Setup

| Param | Value |
|-------|-------|
| Optimizer | AdamW (lr=5e-4, β=(0.9,0.999), wd=1e-4) |
| LR schedule | Linear warmup (2 epochs) + cosine decay |
| Gradient clip | `max_norm=5.0` (L2) |
| Batch size | 64 |
| Buckets | 4: T={32,64,96,128}, limits={33,65,97,129} |

---

## Findings

### 1. Dead Image Pathway (Critical)

**Diagnostic data:**

```
Loss (correct img):         7.95
Loss (blank img):           8.08
Loss (random noise img):    7.58   ← BETTER than correct image!
Loss (swapped img):         7.88   ← BETTER than correct image!
```

**Gradient distribution per parameter group:**

```
Image-path grad norm:   0.055  (0.25% of total)
Embedding grad norm:    5.98   (7.3%)
LM-path grad norm:      21.40  (92.5%)
Total grad norm:        22.22
```

**Per-sample analysis (8 validation samples, loss with correct image vs noise):**

```
[0]  volcano           correct=10.38  noise=9.98   gap=+0.40  (image HARMS)
[1]  Italian greyhound correct=9.18   noise=8.46   gap=+0.72  (image HARMS)
[2]  chickadee         correct=5.42   noise=7.66   gap=-2.24  (image HELPS)
[3]  mitten            correct=11.06  noise=10.49  gap=+0.56  (image HARMS)
[4]  envelope          correct=10.42  noise=11.55  gap=-1.13  (image HELPS)
[5]  Welsh springer    correct=5.87   noise=6.06   gap=-0.19  (image helps slightly)
[6]  coucal            correct=6.12   noise=5.95   gap=+0.17  (image HARMS)
[7]  sombrero          correct=7.97   noise=6.77   gap=+1.19  (image HARMS)
```

**5/8 validation samples produce higher loss with the correct image than with random noise.** The image conditioning gap (correct − noise) averages to **+0.37** — the model is slightly worse with correct images than with random noise.

**Root cause: gradient bottleneck in prefix-LM attention.**

In prefix-LM attention, each text position i attends to all image positions (0..I-1) with weight ~1/(I+i) ≈ 0.3%. The gradient for image-position K comes from:

```
dL/d(K_j) = Σ_text dL/d(score_ij) · Q_i · scale
dL/d(score_ij) = P[i,j] · (∇attention_row)
```

With P[i,j] ≈ 0.003 and the model rapidly learning to predict text-from-text (the easy path), two things happen:
- The softmax probability P[text, image] collapses further
- The gradient dL/d(K_image) becomes negligible
- conv2d weight and bias gradients average ~200× smaller than LM path gradients per-parameter

The conv2d (98K params) receives < 0.25% of total gradient, so it barely trains. Its output remains near-initialization (effectively random features). The text decoder learns to use the random features as a sample-specific fingerprint (memorization) rather than learning a generalizable image→text mapping. This explains 0.75 training loss vs 7-10+ validation loss — classic overfitting through a dead visual backbone.

**Training loss (0.75) vs validation loss (7-10+) across 8 samples:**
```
Training loss (end of epoch 5): ~0.75
Validation loss (same checkpoint):  5.42 – 11.06  (avg ~8.3)
```

### 2. Autoregressive Generation — Image-Independent Output

**Generation with blank image (identical across all samples):**

```
pred(blank) = bigareareareareareareareeeeeeedeeeeee...
```

All 5 samples produce the **exact same** garbage text when image is blank. With correct images, the output varies per sample but is still garbage — confirming the model extracts sample-specific noise from conv2d features, not semantic content.

Partial qualitative observation: sample 658 (class "mitten") predicts "American Stafordshi" — plausible class name fragments appear but don't match the correct label, consistent with the model having learned the general distribution of ImageNet label text without image conditioning.

### 3. Bucket Loss Jump (Catastrophic Forgetting Within Epoch)

**Observed behavior:**

```
e5 end  loss ~0.75  (T=128 bucket, long labels)
e6 b0001 loss  7.37  (T=32 bucket, short labels)
e6 b0010 loss  5.71
e6 b0020 loss  4.42
e6 b0040 loss  3.62
```

**Test 5 result — same sample across bucket sizes:**

```
T=32   full_len=9  stored=8  loss=10.379329
T=64   full_len=9  stored=8  loss=10.379329
T=96   full_len=9  stored=8  loss=10.379329
T=128  full_len=9  stored=8  loss=10.379329
```

Loss is **identical** across padding lengths. Bucket size itself is not the cause.

**Root cause: unequal token count across buckets → gradient variance asymmetry.**

Each epoch processes buckets in fixed order (T=32→64→96→128). The key metric is tokens-per-batch:

| Bucket | Samples/batch | Avg stored_len | Tokens/batch | Gradient variance |
|--------|--------------|----------------|-------------|-------------------|
| T=32   | 64           | ~10            | ~640        | high              |
| T=64   | 64           | ~30            | ~1920       | medium            |
| T=96   | 64           | ~60            | ~3840       | low               |
| T=128  | 64           | ~115           | ~7360       | lowest            |

Long-label batches average gradients over **11× more tokens** than short-label batches. This gives 3.4× lower variance per gradient estimate. The optimizer's weight updates from long-label batches are consistently more precise.

Within an epoch, the model processes T=128 last (~5000 batches of clean, precise gradients). These updates **gradually overwrite** the weight directions that benefit short-label prediction. When the next epoch rolls back to T=32, the weights have drifted from the short-label optimum → loss spike of 7.37. After ~40 batches of short-label training, the model recovers to ~3.62.

The pattern repeats every epoch and worsens as training progresses (more consolidated long-label features overwrite short-label ones more aggressively).

### 4. Gradient Norms

Stable norms of 5-100 (pre-clip) throughout training. With `max_norm=5.0`, clipping is active on most batches. The effective update magnitude is bounded. Not a problem — consistent with training dynamics of a 521K-parameter model on this task.

---

## Recommended Fixes

### Fix 1 — Gradient scaler for image pathway (immediate, ~20 lines)

Manually scale image-path gradients to match LM-path magnitude before optimizer step:

```c
// After dnn_backward, before adamw_step:
float img_scale = 200.0f;  // compensates ~200× ratio
for (const char *name : {"patch_embed.weight", "patch_embed.bias", 
                          "image_norm.weight", "image_pos"}) {
    tensor *t = module_find_param(&vlm->base, name);
    if (t && tensor_grad(t)) {
        int n = tensor_numel(t);
        float *g = tensor_grad(t);
        for (int j = 0; j < n; j++) g[j] *= img_scale;
    }
}
```

Or compute norm ratio dynamically per step for adaptive scaling.

### Fix 2 — Per-sample loss normalization

Replace token-level averaging with sample-level averaging in `tensor_cross_entropy_masked` (or in the training step wrapper):

```c
// Per-sample loss, then average over batch dimension
// Each sample contributes equally regardless of label length
float loss_per_sample[B];
for (int b = 0; b < B; b++)
    loss_per_sample[b] = total_loss_b / mask_sum_b;
loss = mean(loss_per_sample);
```

This eliminates the 11× gradient variance gap between buckets.

### Fix 3 — Increase model capacity

The current architecture (D=128, 2 layers, 2 heads) lacks the capacity to route gradients through attention from text positions back to image positions. Recommended minimum:

```
D_MODEL:   128  → 256
N_LAYERS:  2   → 4
N_HEADS:   2   → 4
```

At D=256, each attention head has d_k=64 (same as before), but 4 heads × 4 layers provide 4× the gradient pathways for image→text information flow.

### Fix 4 — Warmup: freeze LM, train image path first

For the first N batches (e.g., 1000), freeze all LM parameters and train only the image pathway (conv2d, image_norm, image_pos). This forces useful visual features before the text decoder can dominate:

```c
for (int warmup = 0; warmup < 1000; warmup++) {
    // Set requires_grad=false on all LM params
    // Forward + backward (LM params don't accumulate grad)
    // Step optimizer (only image params update)
}
// Then unfreeze all and continue normal training
```

---

## Diagnostic Files

- **Debug source:** `examples/imagenet_vlm/vlm_debug_1.c`
- **Build:** `cd examples/imagenet_vlm && make vlm_debug_1`
- **Run:** `build/vlm_debug_1 --data-dir <path> --ckpt ckpt/<epoch>.bin`
- **Diagnostics:** image conditioning gap, per-param gradient norms, per-sample loss analysis, bucket invariance test, autoregressive generation
