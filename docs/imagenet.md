# ImageNet-1k Training Pipeline

## Overview

Train a Vision Transformer (ViT) on ImageNet-1k using the dnn.c framework.

Three phases:
1. **Preprocess** (Python) — Download from HF, decode JPEG, resize, shard into raw uint8 binary
2. **Dataloader** (C) — Shard-aware reader with global shuffle, batch construction with on-the-fly uint8→float32 normalisation
3. **ViT model** (C) — Patch embedding, transformer encoder, classification head. Training loop with AdamW, LR schedule.

---

## 1. Data Format

### Shard binary format

Each shard file: `{split}-{shard:05d}-of-{total:05d}.bin`

```
[64B header]
  magic       = 0x494D474E    // "IMGN"
  version     = 1
  H, W, C     = image dimensions (uint32 each)
  num_samples = samples in this shard (uint32)
  shard_idx   = this shard index, 0-based (uint32)
  num_shards  = total shards for this split (uint32)
  reserved[32]= 0

[body] — num_samples × (label + pixels):
  [int32]    label         — class index 0–999
  [uint8 × H×W×C] pixels  — HWC row-major, NHWC layout
```

No per-sample alignment padding. Records are tightly packed.

**Magic constant rationale:** `0x494D474E` = ASCII "IMGN". Same pattern as tokenizer's `0x444E4E44` ("DNND").

### Why uint8, not float32

| Factor | uint8 | float32 |
|--------|-------|---------|
| Shard size (224²×3) | ~147 KB/sample | ~588 KB/sample |
| Total train (~1.28M) | ~192 GB | ~768 GB |
| Shard count (1GB ea) | ~192 | ~768 |
| Conversion cost | one NEON pass on batch copy | none |

uint8 saves 4× storage and I/O. Conversion to float32 [0,1] with mean/std normalisation folds into one NEON pass during batch construction.

### Why NHWC, not NCHW

- Natural output of JPEG decode → resize pipeline
- Pixel locality for uint8→float32 conversion (adjacent channels at same spatial position)
- ViT patch embedding (conv2d with kernel=stride=patch_size) typically expects NCHW; transposition happens once in the patch embedding forward, not on every batch copy
- Conversely, storing NCHW would require a channel-interleave transpose during preprocessing

**Decision:** Store NHWC. On first forward, transpose full batch NHWC->NCHW via `tensor_transpose` (one contiguous-to-contiguous pass, cheap with NEON). Then pass NCHW input to existing `tensor_conv2d`. This avoids a separate NHWC conv implementation. Standard ViT uses `Conv2d(in=3, out=D, kernel=P, stride=P)` on NCHW — we match that exactly after the transpose.

### Label encoding

Contiguous 0–999 class index. Mapping from HF synset WNID → int happens in preprocessing.

---

## 2. Shard Counts

Target ~1 GB per shard (balance of I/O granularity and file count).

### Training (1,281,167 images)

| Image size | Bytes/sample | Samples/shard | Shards |
|-----------|-------------|--------------|--------|
| 224×224×3 | 150,532 | ~7,116 | ~180 |
| 256×256×3 | 196,612 | ~5,453 | ~235 |

### Validation (50,000 images)

| Image size | Bytes/sample | Samples/shard | Shards |
|-----------|-------------|--------------|--------|
| 224×224×3 | 150,532 | ~7,116 | ~8 |
| 256×256×3 | 196,612 | ~5,453 | ~10 |

**Decision:** Preprocess at 224×224 (standard ViT input). Accept no random-crop augmentation in v1 — horizontal flip + color jitter only. Storing at 256+ for future random-crop additions is documented as a follow-up.

---

## 3. Preprocessing Pipeline (Python)

### Input

HuggingFace `datasets.load_dataset("ILSVRC/imagenet-1k", split="train")`.

Each sample: `{"image": PIL/JPEG, "label": int}`. Labels already map to 0–999.

### Steps

```
for each image in split:
    1. Decode JPEG → uint8 [H, W, 3] (PIL RGB)
    2. Resize short edge to 256 (bicubic)
    3. Center crop to 224×224
    4. Output: [label:int32][uint8×224×224×3]
    5. Pack into shards of ~1 GB

Last shard may be partial — header.num_samples records actual count.

Write shards to: data/imagenet/{split}-{shard:05d}-of-{total:05d}.bin
Write meta to:   data/imagenet/meta.txt  (1000 lines: synset_id class_name)
```

### Performance

- 1.28M images at ~150 KB/sample after processing
- Preprocessing ~6–8 hours on 8-core machine with PIL
- Use `from multiprocessing import Pool; Pool(...)` for parallelism
- Shuffle images before packing into shards (prevents class-order bias in contiguous shard reads)

### Output tree

```
data/imagenet/
  train-00001-of-00180.bin  ...  train-00180-of-00180.bin
  val-00001-of-00008.bin    ...  val-00008-of-00008.bin
  meta.txt
```

---

## 4. C DataLoader Design

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     imagenet_dataloader                       │
│                                                              │
│  order[] (int*, total_samples) — global epoch permutation    │
│  buffer (uint8*, ~1 GB) — current shard, malloc-owned        │
│                                                              │
│  Open shards via pattern: "data/imagenet/%s-%05d-of-%05d.bin"│
│  On shard boundary: fread next shard into buffer             │
│  Batch: read samples from buffer, convert in flight          │
└──────────────────────────────────────────────────────────────┘
```

### Key design decisions

#### 4a. Buffer outside pool system

The data pool is reset per batch (`mem_pool_reset(data)`). A shard buffer must persist across batches. Therefore the dataloader owns its buffer via `malloc`, not from any pool.

```
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
│ dataloader │──│ buffer   │──│ shard N  │──│ samples[] │
│ (malloc)   │  │ (~1 GB)  │  │ (fread)  │  │ (view)    │
└─────────┘   └──────────┘   └──────────┘   └───────────┘
                                                      │
                                                      ▼ batch construction
                                              ┌──────────────────┐
                                              │ scratch pool     │
                                              │ [bs, 3, H, W]    │
                                              │ float32, NCHW,   │
                                              │ mean/std norm'd  │
                                              └──────────────────┘
```

#### 4b. Global permutation, not per-shard

Single `int *order` of length `total_samples` for the split. Shuffled once per epoch (Fisher-Yates). Dataloader walks sequentially. When `order[pos]` references a sample in a different shard, load that shard.

Memory: 1.28M × 4 B = ~5 MB. Negligible.

#### 4c. No cross-shard batches

If the next `batch_size` positions in `order[]` cross a shard boundary, we advance `pos` to the start of the next shard instead. Wasted samples (<0.01% per epoch) are irrelevant for convergence.

**Simplified (v1 choice):** `order[]` is partitioned by shard. Walk shards in random order, samples within each shard randomly permuted. No boundary crossing possible.

> **Caveat:** Per-shard shuffle is not a true global permutation — samples from different shards never intermix within one epoch. If shards have class distribution skew (mitigated by pre-shuffling during preprocessing), gradients may be noisier. Impact small for 180+ shards; revisit if convergence degrades.

#### 4d. uint8→float32 conversion with mean/std

Folded into the per-batch copy. NEON vectorised on ARM.

```c
// Per-sample conversion: NHWC uint8 → NCHW float32 with mean/std
static void convert_sample(float *dst, const uint8_t *src,
                           int H, int W, int C,
                           const float *mean, const float *std) {
#if DNN_HAVE_NEON
    // NEON: load 8 pixels (24 bytes NHWC), deinterleave to 3 channels
    // convert to float32, (val/255 - mean)/std, interleave to NCHW.
    // mean/std arrays padded to 4 elements (lane 3 unused) via macros.
    float32x4_t vmean = vld1q_f32(mean);  // [m0, m1, m2, x]
    float32x4_t vstd  = vld1q_f32(std);   // [s0, s1, s2, x]
    float32x4_t vdiv  = vdupq_n_f32(1.0f / 255.0f);
    // Per-pixel: vld3_u8 deinterleaves NHWC->3xuint8x8,
    // vshll/vcvtq -> float32, vfma for norm, scatter to NCHW dst.
#else
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                float val = src[h * W * C + w * C + c] / 255.0f;
                dst[c * H * W + h * W + w] = (val - mean[c]) / std[c];
            }
        }
    }
#endif
}
```

#### 4e. Horizontal flip augmentation (v1)

Random horizontal flip: during batch copy, reverse width-direction reads with 50% probability per sample. Folds into the same uint8→float32 loop at zero extra memory.

```c
// xorshift64 from dl->rng_state, faster than libc rand()
unsigned long r = dl->rng_state;
r ^= r << 13; r ^= r >> 7; r ^= r << 17;
dl->rng_state = r;
int flip = (int)(r & 1);  // per sample
if (flip) {
    // read src[h, W-1-w, c] instead of src[h, w, c]
    w_src = W - 1 - w;
}
```

#### 4f. Memory use snapshot

| Allocation | Size | Lifetime |
|-----------|------|----------|
| `dataloader` struct | ~256 B | training duration |
| `buffer` (shard) | ~1 GB | training duration (reused, not reallocated) |
| `order[]` | ~5 MB | training duration (re-shuffled each epoch) |
| batch image tensor | `bs × 3 × H × W × 4 B` | one forward pass (scratch pool) |
| batch label tensor | `bs × 4 B` | one forward pass (scratch pool) |

---

## 5. DataLoader API

```c
#ifndef DNN_IMAGENET_H
#define DNN_IMAGENET_H

#include "tensor.h"

#define IMAGENET_MAGIC       0x494D474E
#define IMAGENET_HEADER_SIZE 64
#define IMAGENET_CLASSES     1000
#define IMAGENET_MEAN        {0.485f, 0.456f, 0.406f, 0.0f}
#define IMAGENET_STD         {0.229f, 0.224f, 0.225f, 0.0f}

/* ── Shard header (on-disk layout) ── */
typedef struct __attribute__((packed)) {
    int32_t magic;
    int32_t version;
    int32_t H;
    int32_t W;
    int32_t C;
    int32_t num_samples;
    int32_t shard_idx;
    int32_t num_shards;
    int8_t  reserved[32];
} imagenet_shard_header;

/* ── DataLoader ── */
typedef struct imagenet_dataloader {
    /* Shard iteration */
    char    pattern[256];      /* e.g. "data/imagenet/train-%05d-of-%05d.bin" */
    int     num_shards;
    int     current_shard;     /* index of shard currently in buffer */

    /* Buffer (malloc-owned, ~1 GB) */
    uint8_t *buffer;
    int      samples_in_shard;
    int      H, W, C;
    int      sample_bytes;     /* 4 + H*W*C */

    /* Epoch permutation */
    int     *order;            /* total_samples-length permutation */
    int      total_samples;
    int      pos;              /* current position in order[] */

    /* Fast per-sample RNG (xorshift64) */
    unsigned long rng_state;
} imagenet_dataloader;

/* ── Lifecycle ── */

/* Create dataloader for a split.
 *
 *   split     — "train" or "val"
 *   data_dir  — path to shard directory, e.g. "data/imagenet"
 *   shuffle   — non-zero to shuffle each epoch (usually train only)
 *
 * Opens all shards to count total samples, allocates order[] and buffer.
 * Returns NULL on error (missing shards, bad headers).
 */
imagenet_dataloader *imagenet_dataloader_create(const char *split,
                                                 const char *data_dir,
                                                 int shuffle);

/* Destroy dataloader, free buffer + order. */
void imagenet_dataloader_free(imagenet_dataloader *dl);

/* ── Epoch lifecycle ── */

/* Shuffle order[] for new epoch (Fisher-Yates). No-op if shuffle==0. */
void imagenet_dataloader_shuffle(imagenet_dataloader *dl);

/* ── Batch construction ── */

/* Fill batch tensors from current position.
 *
 *   img   — pre-allocated scratch tensor [bs, 3, H, W] (NCHW, float32)
 *           output pixels are mean/std normalised.
 *   lbl   — pre-allocated scratch tensor [bs] (int32 in float data region)
 *   bs    — batch size
 *
 * Returns number of samples actually written (< bs at epoch end, 0 if done).
 * Advances internal pos by the returned count.
 */
int imagenet_dataloader_next_batch(imagenet_dataloader *dl,
                                    tensor *img, tensor *lbl,
                                    int bs);

/* ── Accessors ── */
int imagenet_dataloader_total(imagenet_dataloader *dl);
int imagenet_dataloader_remaining(imagenet_dataloader *dl);

#endif /* DNN_IMAGENET_H */
```

### Implementation notes

**`next_batch` internals:**

```
1. if pos >= total_samples → return 0
2. Determine target shard: shard_of(pos)
3. If shard != current_shard → load_shard(shard)
4. For i = 0..bs-1:
     sample_idx_in_shard = order[pos + i] % samples_in_shard
     src = buffer + sample_idx_in_shard * sample_bytes
     label = *(int32_t*)src
     pixels = src + 4
     convert_sample(...) with optional horizontal flip
     lbl->data[i] = label
5. pos += bs
6. return min(bs, remaining)
```

**`load_shard`:**

```
1. FILE *f = fopen(pattern with shard_idx, "rb")
2. fread header, validate magic/version/HWC
3. Calculate data_bytes = num_samples * sample_bytes
4. Ensure buffer capacity (realloc if needed)
5. fread(buffer, data_bytes, f)
6. fclose(f)
7. current_shard = shard_idx
8. samples_in_shard = num_samples
```

**Shard buffer realloc:** Only happens if a later shard has more samples than expected (shouldn't happen with uniform shard packing). On first load, allocate exactly the needed size.

---

## 6. Training Loop Integration

```c
// ── Main training loop ──

srand(SEED);  // deterministic reproducibility

dnn_ctx ctx;
dnn_ctx_init(&ctx, PARAMS_POOL_SIZE, SCRATCH_POOL_SIZE, 16 << 20);

imagenet_dataloader *train_dl = imagenet_dataloader_create("train", "data/imagenet", 1);
imagenet_dataloader *val_dl   = imagenet_dataloader_create("val",   "data/imagenet", 0);
int total_train = imagenet_dataloader_total(train_dl);
int total_val   = imagenet_dataloader_total(val_dl);
int train_batches = (total_train + BATCH_SIZE - 1) / BATCH_SIZE;
int val_batches   = (total_val   + BATCH_SIZE - 1) / BATCH_SIZE;

vit_model *model = vit_create(ctx.params, ...);
int n_params;
tensor **params = module_parameters(&model->base, &n_params);
adamw_opt *opt = adamw_create(ctx.params, params, n_params, ...);

for (int epoch = 0; epoch < EPOCHS; epoch++) {
    imagenet_dataloader_shuffle(train_dl);
    double epoch_loss = 0.0;
    int batches_seen = 0;

    for (int b = 0; b < train_batches; b++) {
        tensor *img = tensor_scratch(ctx.scratch, 4,
                       (int[]){BATCH_SIZE, 3, IMG_H, IMG_W}, 0);
        tensor *lbl = tensor_scratch(ctx.scratch, 1,
                       (int[]){BATCH_SIZE}, 0);

        int got = imagenet_dataloader_next_batch(train_dl, img, lbl, BATCH_SIZE);
        if (got == 0) break;

        tensor *logits = vit_forward(ctx.scratch, model, img);
        tensor *loss   = tensor_cross_entropy(ctx.scratch, logits, lbl, 1);

        dnn_backward(ctx.scratch, loss);
        epoch_loss += ((float*)loss->data)[0];
        batches_seen++;

        adamw_step(opt);
        adamw_zero_grad(opt);

        mem_pool_reset(ctx.scratch);
    }

    printf("Epoch %d: train loss %.4f\n", epoch, epoch_loss / batches_seen);

    // ── Validation (no grad, no shuffle) ──
    double val_loss = 0.0;
    int val_seen = 0, correct = 0;
    for (int b = 0; b < val_batches; b++) {
        tensor *img = tensor_scratch(ctx.scratch, 4,
                       (int[]){BATCH_SIZE, 3, IMG_H, IMG_W}, 0);
        tensor *lbl = tensor_scratch(ctx.scratch, 1,
                       (int[]){BATCH_SIZE}, 0);

        int got = imagenet_dataloader_next_batch(val_dl, img, lbl, BATCH_SIZE);
        if (got == 0) break;

        tensor *logits = vit_forward(ctx.scratch, model, img);
        tensor *loss = tensor_cross_entropy(ctx.scratch, logits, lbl, 0);
        val_loss += ((float*)loss->data)[0];
        val_seen++;

        // Top-1 accuracy
        for (int i = 0; i < got; i++) {
            float *logit_i = (float*)logits->data + i * 1000;
            int pred = 0;
            for (int c = 1; c < 1000; c++)
                if (logit_i[c] > logit_i[pred]) pred = c;
            if (pred == (int)((float*)lbl->data)[i]) correct++;
        }

        mem_pool_reset(ctx.scratch);
    }
    printf("Epoch %d: val loss %.4f, top-1 %.2f%%\n",
           epoch, val_loss / val_seen, 100.0f * correct / total_val);
}

imagenet_dataloader_free(train_dl);
imagenet_dataloader_free(val_dl);
dnn_ctx_destroy(&ctx);
```

### Differences from MNIST training loop

| Aspect | MNIST | ImageNet |
|--------|-------|----------|
| Data location | loaded into data pool at start | dataloader external buffer |
| Data pool reset | never during training | optional (data pool is small, used only for model-internal data allocs) |
| Scratch reset | per batch | per batch (identical) |
| Shuffle | per epoch on `indices[tr_n]` | per epoch on `order[total]` via dataloader |

---

## 7. ViT Model Structure (for context)

Placeholder architecture spec — details will evolve in a separate model doc.

```
PatchEmbed(3 → D, kernel=P, stride=P) → [B, N, D]     where N = (H/P) × (W/P)
  + position embedding (learnable) + class token

TransformerEncoder × L:
  LayerNorm → MultiHeadSelfAttention(D, H) → residual
  LayerNorm → MLP(D, 4D) → residual

LayerNorm(global) → Linear(D, 1000)  (classification head)
```

Framework components that exist:
- `linear`, `layer_norm`, `tensor_attention` (scaled dot-product), `tensor_split_heads`/`tensor_merge_heads`
- `tensor_conv2d` (for patch embedding), `module` hierarchy
- `adamw`, `lr_scheduler`

Components to build:
- `vit_patch_embed` — conv2d with kernel=stride=patch_size, flatten to [B, N, D]
- `vit_pos_embed` — learnable position embedding + class token
- `vit_encoder_block` — pre-norm transformer block without causal mask (bidirectional)
- `vit_model` — compose all above + classification head
- `vit_forward` — full forward pass

---

## 8. Open Questions / Future Work

| Issue | Status |
|-------|--------|
| Random resized crop augmentation | Deferred. Requires resize op in C or store at 256+ and crop. |
| Color jitter augmentation | Deferred. Simple in batch-copy loop but adds params. |
| Mixup / CutMix | Deferred. Model-side, not data format. |
| Gradient accumulation | Trivial: don't step optimiser every batch. |
| Multi-GPU | Far future — framework has no GPU support. |
| FP16 storage in shards | Could halve shard count later. Requires fp16→fp32 conversion on batch load. |
| Memory-mapped shards | `mmap` instead of `fread` for zero-copy access. v1 uses `fread` for portability. |
| Distributed data loading | Out of scope — single node only. |
