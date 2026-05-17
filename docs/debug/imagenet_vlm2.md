# ImageNet VLM Debug Report

**Author:** AI debug agent (NVIDIA senior DL engineer review simulation)
**Date:** 2026-05-17
**Context:** Training decoder-only VLM (prefix-LM) on ImageNet-1k classification via label-text generation. Epoch 5, loss 5.65 → 0.75, but 0% val accuracy.

## Symptom Summary

| Symptom | Detail |
|---------|--------|
| Loss drops 5.65→0.75 | Good trend, model appears to converge |
| Grad norms stay 1-100 | Pre-clip avg ~7, spikes to 372. GRAD_CLIP=5.0 clips most steps |
| Predictions garbage | e.g. `true="stopwatch" VS pred="tailed frog, bell toad, ..."` |
| Bucket crossover spike | Loss jumps 0.75 → 7.36 at epoch boundary (T=128→T=32) |
| 0% val accuracy | Full-string autoregressive decode never matches |

## Root Cause: Gradient Imbalance — Image Encoder Starved by Uniform Attention

**No bug found in any backward path.** Every autograd function (conv2d, rms_norm, tensor_cat, transpose, reshape, contiguous copy, attention backward) is correctly implemented. The gradient path from CE loss → patch_embed is complete.

The issue is **gradient dilution via the attention mechanism**, not broken backward wiring.

### Mechanism

In PREFIX_LM attention, each text query attends to **all 196 image patch keys**. Early in training, attention weights are ~uniform over patches:

```
gradient_to_each_patch ≈ (1/196) × gradient_from_text_position
```

Each image patch receives **~0.5% of the gradient** that a text position receives. The conv2d (98K params: 128×3×16×16) competes with text components (embedding 33K params + transformer layers) getting full-strength gradients.

### Secondary Factor: image_pos Gradient Sink

`image_pos` (196×128 = 25K learnable params) sits on the same gradient path as conv2d via `tensor_add`. It has fewer params and a simpler function (just additive position bias), so it learns faster and **captures a disproportionate share of the image-position gradient**, further starving conv2d.

### How This Explains Each Symptom

#### Loss 5.65 → 0.75

Uniform over V=261 → CE ≈ ln(261) ≈ 5.56. The model learns a **text-only language model** of ImageNet label strings (byte-level patterns: comma-separated synonyms, common ASCII bytes, English word structure). The weakly trained image encoder provides marginal additional signal.

This is **not catastrophic forgetting** — it's the model converging to a local minimum where text patterns dominate because they're easier to learn.

#### Predictions Garbage but Text-Like

Outputs like "tailed frog, bell toad, ribbed toad" are **actual ImageNet synset names** (class 42: "tailed frog, bell toad, ribbed toad, tailed frog, Ascaphus truei"). The model has learned the label corpus distribution but cannot associate specific labels with specific images.

The repeated degeneracy patterns ("celular, ar, be, pur phobe...") occur when the model enters a fixed point: weak image signal → wrong byte → fed back as next input → attention collapses.

#### Grad Norms Stay High

A text-only LM of 1000×~50 byte sequences is inherently hard — the model never saturates. ImageNet labels have diverse byte patterns ("stopwatch, stop watch" vs "whiptail, whiptail lizard" vs "boathouse") with limited statistical redundancy. Gradients remain large.

#### Bucket Crossover Spike (0.75 → 7.36)

**This confirms image conditioning is weak, not absent.**

| Bucket | T | Text context | Loss | Why |
|--------|---|-------------|------|-----|
| 3 | 128 | ~100 bytes | 0.75 | Rich text context, text LM dominates |
| 0 | 32 | ~5-30 bytes | 7.36 | Sparse context → model relies MORE on weak image signal → confidently WRONG |

At T=128, the text LM has enough context to predict well. At T=32, short labels leave the model dependent on image features that aren't discriminative. A `-ln(0.0006)` per token means the model is assigning ~0.06% probability to the correct byte — **worse than uniform random** (0.38%) — because it's confidently predicting wrong tokens based on sparse text context.

The 7.36 is NOT forgetting (model weights don't change during evaluation). It's the model being confidently wrong with limited text context.

#### 0% Val Accuracy

Full-string autoregressive generation compounds the problem. First byte is wrong → all subsequent bytes are wrong → EOS never reached → model generates to max length → string never matches.

## Empirical Verification

The debug tool `vlm_debug_2.c` will confirm these predictions:

```
make vlm_debug_2 DATA_DIR=/path/to/imagenet-shards CKPT=ckpt/imagenet_vlm_epoch05_*.bin
```

Expected test results:

| Test | Expected Finding |
|------|------------------|
| TEST 1: Per-layer grad norms | `patch_embed.weight` gn 10-100× smaller than `embed.weight` gn |
| TEST 2: Real vs blank image | KL(real‖blank) < 0.01 nats. Argmax at 1st text position identical. |
| TEST 3: Bucket crossover | Bucket 3 (T=128) loss < 1.0. Bucket 0 (T=32) loss > 6.0. Same T=128 on bucket-0 data → similar loss to bucket-3. |
| TEST 4: Per-position entropy | First text position: HIGH entropy (~uniform). Later positions: LOW entropy (confident but wrong). |
| TEST 5: Autoregressive gen | First byte wrong → cascade → never hits EOS → max length garbage |
| TEST 6: Gradient flow | `patch_embed.weight` grad norm << `embed.weight` grad norm. `image_pos` grad norm relatively large. |

## Recommended Fixes

### 1. Gradient Rebalancing (Quickest Win)

Scale learning rates unequally:

```c
// Per-param LR groups in AdamW
float lr_vision = base_lr * 50.0f;   // conv2d, image_norm, image_pos
float lr_text   = base_lr * 1.0f;    // embed, transformer, lm_head
```

Implemented by creating two optimizer groups (or post-hoc scaling grads before step).

### 2. Reduce Patch Attention Dilution

Replace 196 direct image patch tokens with pooled representation:

```
[B, 196, D] → attention pooling → [B, K, D]  where K ≈ 4-16
```

This reduces gradient dilution from 1/196 to 1/K and gives each aggregated token a stronger gradient signal. A learned pooling (cross-attention with learned queries) or simple average + learned projection would work.

### 3. Vision Warmup

Pre-train `patch_embed + image_norm + image_pos` before VLM training:

- **Contrastive**: SimCLR-style on ImageNet images (class-agnostic, learns patch features)
- **Reconstruction**: Masked patch prediction (MAE-style)
- **Classification head**: Add a temporary 1000-way linear head on pooled image features, train for 1 epoch, remove head

Even 1 epoch of contrastive pre-training would push conv2d weights far from random and make the image signal discriminative.

### 4. Disable image_pos Initially

Remove `image_pos` for the first few epochs so ALL gradient through the image branch reaches conv2d. Add it back later for positional refinement.

### 5. Auxiliary Image Loss During VLM Training

Add a secondary loss that forces the image encoder to produce class-discriminative features:

```c
// Pool image features (mean over patches)
tensor *img_pooled = tensor_mean(scratch, img_embeds, 1);  // [B, D]
// Classification head (shared or separate)
tensor *aux_logits = linear_forward(scratch, aux_head, img_pooled);  // [B, 1000]
// Auxiliary CE loss
tensor *aux_loss = tensor_cross_entropy(scratch, aux_logits, class_labels, 1);
total_loss = ce_loss + 0.1 * aux_loss;
```

This ensures image features carry class information regardless of text encoder state.

## Code Artifacts

- **Debug tool:** `vlm_debug_2.c` — loads checkpoint, runs 6 diagnostic tests
- **Training entry:** `examples/imagenet_vlm/imagenet_vlm.c`
- **VLM model:** `src/vlm.c`, `include/vlm.h`
- **ImageNet dataloader:** `src/imagenet_vlm.c`, `include/imagenet_vlm.h`
- **Attention (prefix-LM):** `src/attention.c` (triangular tiled fwd/bwd with mode + seq_lens)
