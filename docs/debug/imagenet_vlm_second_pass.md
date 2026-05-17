# ImageNet VLM debug reports — second-pass assessment

Date: 2026-05-17  
Target checkpoint: `ckpt/imagenet_vlm_epoch05_20260517T085909.bin`  
Reviewed reports: `docs/debug/imagenet_vlm1.md`, `docs/debug/imagenet_vlm2.md`

## Scope

I re-read source paths used by training/eval:

- `examples/imagenet_vlm/imagenet_vlm.c`
- `src/vlm.c`
- `src/imagenet_vlm.c`
- `src/ops_activation.c`
- `examples/imagenet_vlm/vlm_debug_1.c`
- root `vlm_debug_2.c`

I ran checkpoint diagnostics against available local shards under `data/imagenet` and added a small candidate-scoring probe:

- probe source: `examples/imagenet_vlm/vlm_probe.c`
- binary used: `build/vlm_probe`

Important caveat: local `data/imagenet/labels.txt` is synthetic (`c000`, `cls654_short`, `label_381_of_imagenet_set`, ...), while saved checkpoint generations contain real ImageNet label fragments (`bighorn`, `trashbin`, etc.). Therefore local label accuracy/rank numbers are not valid ImageNet accuracy. They are still useful for image-dependence probes and checkpoint behavior sanity checks.

## Source facts that change interpretation

### Training log loss is cumulative epoch mean, not current batch loss

In `examples/imagenet_vlm/imagenet_vlm.c`:

- `epoch_loss += lv` at line 416
- printed `epoch_loss / batch_count` at line 429

So log comparison:

```text
e5 end loss ~0.75 at T=128
e6 batch 1 loss 7.36 at T=32
```

is not apples-to-apples. `e5 loss` is epoch cumulative mean after all previous buckets. `e6 batch 1` is near-current because only one batch exists in epoch mean. Need log current `lv`, EMA, bucket id, token count.

### Autoregressive shift/decode path looks structurally correct

Dataloader builds:

- input: `BOS, byte0, ..., EOS` (`src/imagenet_vlm.c:472-473`)
- target: `byte0, ..., EOS` (`src/imagenet_vlm.c:477`)
- mask includes target bytes/EOS positions (`src/imagenet_vlm.c:482`)

Generation uses last prompt token row to predict next byte:

- `last_tok = N_IMG_TOK + seq_len - 1` in `examples/imagenet_vlm/imagenet_vlm.c:113,243`

No obvious off-by-one generation bug found.

## Checkpoint-grounded data

### Real/blank/noise image losses + grad groups

Command pattern:

```bash
./build/vlm_debug_1 --data-dir data/imagenet --ckpt <ckpt>
```

First 5 checkpoints from same 20260517 run:

| ckpt | correct loss | blank loss | noise loss | swapped loss | image grad | embed grad | LM grad | total grad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch01 | 10.2890 | 10.4333 | 10.2203 | 10.3084 | 3.745e+00 | 8.946e+01 | 1.051e+02 | 1.381e+02 |
| epoch02 | 10.5603 | 12.2059 | 11.1415 | 10.5603 | 8.396e-22 | 1.995e+02 | 7.159e+01 | 2.120e+02 |
| epoch03 | 13.5928 | 14.6419 | 14.0395 | 13.5928 | 2.064e-02 | 1.276e+01 | 5.428e+01 | 5.576e+01 |
| epoch04 | 12.4676 | 13.6388 | 13.0838 | 12.4655 | 2.335e-01 | 2.511e+01 | 3.587e+01 | 4.378e+01 |
| epoch05 | 14.2776 | 14.4135 | 14.4678 | 14.2795 | 5.089e-02 | 1.124e+01 | 4.659e+01 | 4.792e+01 |

Readout:

- Image-path gradient at epoch05 is tiny relative to LM path.
- Correct-vs-swapped is basically identical at epoch05 (`14.2776` vs `14.2795`) on this probe.
- Blank/noise do not catastrophically change loss; model is weakly image-conditioned at best.

### Generation from epoch05

`vlm_debug_1` on epoch05 generated real ImageNet-like fragments even with synthetic local labels:

```text
true=cls654_short
pred=bigintin-b, sn, sn sn bin, trashbin, ...
pred(blank)=bigareareareareareareareeeeeeedeeeeee...
```

This supports: decoder learned ImageNet label text distribution / attractors. It does not support a decode indexing bug.

### Same sample, different padding lengths

Epoch05, same validation sample:

| T | full_len | stored | loss |
|---:|---:|---:|---:|
| 32 | 14 | 13 | 12.477915 |
| 64 | 14 | 13 | 12.477915 |
| 96 | 14 | 13 | 12.477915 |
| 128 | 14 | 13 | 12.477915 |

Padding length itself is not cause.

### Candidate-label scoring probe

Probe command:

```bash
./build/vlm_probe --data-dir data/imagenet \
  --ckpt ckpt/imagenet_vlm_epoch05_20260517T085909.bin --n 10
```

Epoch05 local-label candidate scoring:

```text
SUMMARY n=10 real_top1=0.0000 real_top5=0.0000
blank_top1=0.0000 blank_top5=0.0000
mean_rank_real=613.10 mean_rank_blank=623.60
mean_true_loss_real=14.1752 mean_true_loss_blank=14.3152
```

Per-sample behavior: real image mostly predicted same synthetic candidate (`822`); blank image predicted another fixed candidate (`142`). Since labels are synthetic while checkpoint text prior is real ImageNet, do not treat this as ImageNet accuracy. Treat it as another weak-image-control indicator.

Across same 20260517 checkpoints, `n=5`:

| ckpt | real top1 | real top5 | blank top1 | blank top5 | mean rank real | mean rank blank | mean true loss real | mean true loss blank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch01 | 0 | 0 | 0 | 0 | 598.6 | 599.6 | 10.0406 | 10.1988 |
| epoch02 | 0 | 0 | 0 | 0 | 471.8 | 489.4 | 10.2042 | 11.8378 |
| epoch03 | 0 | 0 | 0 | 0 | 494.4 | 476.2 | 13.1417 | 13.7908 |
| epoch04 | 0 | 0 | 0 | 0 | 501.8 | 471.2 | 12.7382 | 14.1302 |
| epoch05 | 0 | 0 | 0 | 0 | 566.8 | 567.2 | 13.7321 | 13.8618 |

Again: label mismatch caveat. Real-vs-blank ranks being very close is the useful part.

### Checkpoint weight movement

Parsed checkpoint tensors directly. Group weight norms:

| ckpt | vision wnorm | embed wnorm | LM wnorm |
|---|---:|---:|---:|
| epoch01 | 25.13 | 34.77 | 56.38 |
| epoch02 | 42.66 | 39.40 | 101.34 |
| epoch03 | 58.53 | 36.04 | 128.20 |
| epoch04 | 68.38 | 35.99 | 144.71 |
| epoch05 | 74.75 | 36.11 | 154.81 |

Delta from epoch01:

| ckpt | vision delta | vision rel | embed delta | embed rel | LM delta | LM rel |
|---|---:|---:|---:|---:|---:|---:|
| epoch02 | 25.84 | 1.03 | 26.02 | 0.75 | 68.47 | 1.21 |
| epoch03 | 44.10 | 1.75 | 29.98 | 0.86 | 100.22 | 1.78 |
| epoch04 | 54.89 | 2.18 | 31.91 | 0.92 | 118.94 | 2.11 |
| epoch05 | 61.73 | 2.46 | 32.69 | 0.94 | 130.27 | 2.31 |

This falsifies the strongest wording “vision path never trains”. Vision weights move a lot. Better diagnosis: vision path receives tiny current gradients and does not learn useful class-conditional features under this objective.

## Assessment of Pedro/Carmelo reports

### Report 1 (`imagenet_vlm1.md`)

Strong:

- Best empirical probes: real/blank/noise, per-sample correct-vs-noise, same-sample T invariance.
- Correctly identifies weak image conditioning.

Weak / overclaimed:

- “Dead image pathway” too strong. Weight deltas show vision params moved substantially across checkpoints.
- “Catastrophic forgetting from token-count variance” not proven. Epoch-boundary log comparison is contaminated by cumulative epoch mean logging.
- Proposed `200x` manual image gradient scaling is too blunt. Could destabilize optimizer; needs update-norm based param groups at minimum.

### Report 2 (`imagenet_vlm2.md`)

Strong:

- Best conceptual framing: model learns label-text distribution instead of image→label mapping.
- Correctly skeptical that generation itself is primary bug.
- Fixes with aux classifier / image loss are directionally better than blind scaling.

Weak / overclaimed:

- “No backward bug found” is not proven without finite-diff checks.
- Artifact issue: root `vlm_debug_2.c` calls `vision_lm_train_step_padded(..., NULL, GRAD_CLIP, ...)`; current `src/vlm.c` asserts `opt`, so that path is not runnable as written.
- Bucket explanation does not account for cumulative-loss logging and fixed class-by-label-length partition.

## Answers to original suspicions

### 1. Grad norm never decreased

Current answer: not main root cause by itself.

- Pre-clip grad norms in 1–100 range are plausible under clipped AdamW.
- But group distribution is problematic: epoch05 backward on probe has image grad `5.09e-02` vs LM grad `4.66e+01`.
- Vision weights did move across epochs, so issue is not zero training. Issue is weak/poor semantic supervision for vision under teacher-forced label CE.

Need add logs:

- unclipped grad norm
- clipped coefficient
- per-group grad norm
- per-group update norm / weight norm
- current batch loss `lv`, not only epoch mean

### 2. VLM generation bugged?

Current answer: unlikely as primary cause.

- Shift and AR decode indexing look correct.
- Generated strings are label-like ImageNet fragments, not random token-index corruption.
- Real/blank/candidate probes show image does not control label choice enough. First wrong token cascades into garbage.

Need add eval modes:

- teacher-forced label candidate scoring over 1000 labels
- first-token accuracy / CE
- greedy generation exact match
- real-vs-blank delta for all three

### 3. Bucketing catastrophic forgetting?

Current answer: unproven.

Evidence against direct padding bug:

- Same sample has identical loss for T=32/64/96/128.

Real risks still present:

- Fixed bucket order each epoch: T=32→64→96→128.
- Buckets partition classes by label text length, so this is a class curriculum/order effect.
- Token-averaged CE overweights long labels/classes because they contribute more supervised tokens.
- Logs currently hide current-bucket behavior because printed loss is cumulative epoch mean.

Need next run:

- randomize bucket order per epoch, or interleave buckets
- log per-bucket current loss and EMA
- normalize loss per sample, then average samples
- evaluate short/medium/long label classes separately

## Bottom-line diagnosis

Most likely failure mode: teacher-forced byte-label LM objective lets decoder reduce loss by modeling ImageNet label strings while vision features remain weak/non-discriminative. Greedy generation exposes this as label-prior attractors and first-token cascade. Bucketing may amplify bias through fixed class-length order and token weighting, but current reports do not prove catastrophic forgetting.

## Recommended next actions

1. Fix logging first: print `lv`, EMA, bucket id, mean stored_len, token count, clip coefficient.
2. Add proper eval: 1000-label candidate scoring with real ImageNet `labels.txt`, real vs blank vs shuffled images.
3. Add image-discriminative supervision: pooled image 1000-way aux head, weight ~0.1–1.0.
4. Change batching: random/interleaved buckets or token-budget batches; randomize bucket order.
5. Change loss aggregation: per-sample CE normalization before batch mean.
6. Avoid fixed `200x` scaling until update/weight norm logs show needed ratio.
