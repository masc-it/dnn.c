# VLM Requirements 1 — Image Prefix + GPT Decoder

Goal: add proper high-level Vision-Language Model support while reusing current GPT/transformer stack.

Target architecture:

```text
image [B,C,H,W]
  -> patch conv2d(kernel=patch, stride=patch, out_channels=d_model)
  -> flatten patches to img_embeds [B,I,d_model]
  -> learned image positional embeddings (when use_image_pos=1)

text_ids [B,T]
  -> GPT token embedding
  -> txt_embeds [B,T,d_model]

cat(img_embeds, txt_embeds, dim=1)
  -> multimodal embeds [B,I+T,d_model]
  -> classic decoder-only GPT blocks
  -> norm + tied lm_head
  -> logits [B,I+T,vocab]
```

Expected deliverable:
- `include/vlm.h`
- `src/vlm.c`
- small GPT helper refactor in `include/gpt.h` / `src/gpt.c`
- `include/dnn.h` export update
- tests for forward/train/generate

Implementation target levels:
- **MVP**: image prefix + text forward, text loss, no-cache generation.
- **Proper caption VLM support**: cached generation, prefix-LM image attention, RoPE max length includes image tokens, module params/state paths work.
- **Flexible batching P1**: padded variable-length text batches with optimized attention lengths + masked CE.
- **Instruction/chat P1**: masked text loss for answer-only supervision.

---

## Closed decisions for handoff

These are fixed for this implementation. Do not leave as optional/TBD:

1. **Attention mode:** implement prefix-LM attention. Image patch prefix is bidirectional; text remains causal.
2. **Attention API:** add `tensor_attention_ex(..., attention_mode mode, int prefix_len, const int *seq_lens)` with `ATTENTION_CAUSAL` and `ATTENTION_PREFIX_LM`. `seq_lens` is nullable and can be ignored until P1 padding path. No `ATTENTION_FULL` in this task.
3. **Transformer API:** add `transformer_block_forward_ex(..., mode, prefix_len, seq_lens)` and GPT embed-forward `_ex` helpers. VLM must not duplicate transformer block internals.
4. **VLM forward:** use prefix-LM mode with `prefix_len = vlm->n_img_tokens`.
5. **Image positional embeddings:** implement learned `image_pos [1,I,D]` with `use_image_pos` flag. Tests should cover enabled path. No 2D row/col pos in this task.
6. **Modality embeddings:** do not implement. Leave for future doc/PR.
7. **Loss:** implement `vision_lm_train_step()` with full text CE for fixed-length caption/from-scratch training. `tensor_cross_entropy_masked()` is P1 for padded batches and instruction/chat tuning, not blocker for initial fixed-length caption pretraining.
8. **Generation:** implement both no-cache and cached VLM generation. Cached greedy output must match no-cache greedy output in tests.
9. **Sampling helpers:** expose GPT sampling helpers (`decoder_lm_argmax_token`, `decoder_lm_sample_with_temp`) instead of duplicating in VLM.
10. **Patch weight init:** use Xavier-style normal std `sqrt(2/(C*P*P + d_model))`; image pos uses std `0.02`; LM uses existing GPT init.
11. **2D RoPE:** do not implement. Use learned `image_pos [1,I,D]` plus existing 1D RoPE over combined sequence. Defer 2D image RoPE to future design.
12. **Saved attention `P`:** keep full `[B,H,N,N]` saved `P` initially. No packed-P or FlashAttention in this task.
13. **Image size:** fixed image size per `vision_lm` instance. Variable-size images out of scope.
14. **State/loading:** VLM checkpoints save full VLM module. GPT-only weight reuse can be done by loading into `&vlm->lm->base` separately.
15. **Padding/packing:** initial handoff assumes fixed-length or exact-length-bucketed text with no PAD. P1 adds padded variable-length batches with optimized per-sample effective lengths. Do not implement multi-example sequence packing yet.
16. **Cached VLM prefill:** cached generation prefill must also use prefix-LM attention for image rows. Current causal cached prefill is not sufficient.

---

## Current reusable pieces

Already available:

- `conv2d_create`, `conv2d_forward` for patch projection
- `embedding_create`, `embedding_forward` for text tokens
- `tensor_transpose`, `tensor_contiguous`, `tensor_reshape` for patch flattening
- `tensor_cat` for `[img_embeds, txt_embeds]`
- `transformer_block_forward` for full-sequence training
- `transformer_block_forward_cached` + `kv_cache` for generation
- `rms_norm_forward` / `layer_norm_forward`
- `tensor_matmul_add(... trans_b=1 ...)` for tied LM head
- `tensor_cross_entropy` for simple next-token LM loss
- `module` registration system for params + optimizer

Main gap: `src/gpt.c` only exposes token-ID forward (`decoder_lm_forward`). VLM needs embedding-level forward: start from `[B,S,D]` multimodal embeddings and reuse GPT blocks/norm/lm_head.

Secondary gaps:
- no masked cross-entropy / ignore-index loss yet; add as P1 before padded batches or instruction/chat tuning
- GPT sampling helpers are `static` in `src/gpt.c`; expose them in this task
- `include/dnn.h` currently does not export `gpt.h`; export it in this task
- no high-level image-prefix generation API; add it in this task

---

## Recommended GPT helper refactor

Add these public helpers to `include/gpt.h` and implement in `src/gpt.c`:

```c
/* Token embedding only: input_ids [B,T] -> embeds [B,T,d_model]. */
tensor *decoder_lm_token_embeds(struct mem_pool *scratch,
                                decoder_lm *lm,
                                const tensor *input_ids);

/* Forward from already-built embeddings [B,S,d_model].
 * Reuses lm->blocks, lm->norm, lm_head.
 * Returns logits [B,S,vocab_size]. */
tensor *decoder_lm_forward_embeds(struct mem_pool *scratch,
                                  decoder_lm *lm,
                                  const tensor *embeds);

/* Hidden forward only: embeds [B,S,D] -> final normalized h [B,S,D].
 * Useful for generation when caller only needs last-token logits. */
tensor *decoder_lm_hidden_from_embeds(struct mem_pool *scratch,
                                      decoder_lm *lm,
                                      const tensor *embeds);

/* LM head only: h @ embed->weight^T + lm_head bias.
 * Exposes current static _lm_head_forward behavior. */
tensor *decoder_lm_lm_head_forward(struct mem_pool *scratch,
                                   decoder_lm *lm,
                                   const tensor *h);
```

Then rewrite existing `decoder_lm_forward()` to reuse helpers:

```c
tensor *decoder_lm_forward(...) {
    tensor *h = decoder_lm_token_embeds(scratch, lm, input_ids);
    return decoder_lm_forward_embeds(scratch, lm, h);
}

tensor *decoder_lm_forward_embeds(...) {
    tensor *h = decoder_lm_hidden_from_embeds(scratch, lm, embeds);
    return decoder_lm_lm_head_forward(scratch, lm, h);
}
```

Why this matters:
- VLM can reuse all GPT internals without duplicating block/norm/lm_head logic.
- `lm_head` tied-weight implementation stays in one place.
- Future GPT changes automatically apply to VLM.

Do not duplicate GPT internals in `src/vlm.c`; expose helpers from `src/gpt.c`.

Expose sampling helpers too:

```c
int decoder_lm_argmax_token(const float *logits, int vocab_size);
int decoder_lm_sample_with_temp(const float *logits, int vocab_size, float temp);
```

---

## Public VLM API

Create `include/vlm.h`.

Header dependencies:

```c
#ifndef DNN_VLM_H
#define DNN_VLM_H

#include "tensor.h"
#include "module.h"
#include "nn.h"
#include "gpt.h"
#include "optim.h"
```

### Struct

```c
typedef struct vision_lm {
    module      base;             /* first field */

    decoder_lm *lm;               /* text decoder; owns token embed, blocks, norm, lm_head */
    conv2d     *patch_embed;      /* image patch projection: C -> d_model */

    tensor     *image_pos;        /* nullable [1, n_img_tokens, d_model], learned */

    int use_image_pos;

    int image_channels;
    int image_h;
    int image_w;
    int patch_size;
    int patch_h;
    int patch_w;
    int n_img_tokens;

    int d_model;
    int vocab_size;
} vision_lm;
```

Notes:
- `vision_lm` owns child `decoder_lm` created by `decoder_lm_create()`.
- `image_pos` must be a direct param registered on `vision_lm->base` when enabled.
- No modality embedding in this task.
- `patch_embed` must be registered as child module.
- `lm` must be registered as child module.
- `d_model == lm->d_model == patch_embed->out_channels`.
- State dict names must be stable: `patch_embed.weight`, `patch_embed.bias`, `image_pos`, `lm.embed.weight`, `lm.blocks.0...`.

### Constructor

```c
vision_lm *vision_lm_create(struct mem_pool *params_pool,
                            int vocab_size,
                            int d_model,
                            int n_layers,
                            int n_heads,
                            int d_k,
                            int intermediate_size,
                            int image_channels,
                            int image_h,
                            int image_w,
                            int patch_size,
                            int use_image_pos);
```

No extended constructor in this task. Keep API small; add `vision_lm_create_ex` only in a future PR if modality embeddings or variable image sizing are added.

Constructor behavior:

```c
assert(image_h % patch_size == 0);
assert(image_w % patch_size == 0);
assert(d_model == n_heads * d_k);

vlm = pool alloc;
module_init(&vlm->base, params_pool, "vision_lm");

vlm->patch_embed = conv2d_create(params_pool,
                                 image_channels,
                                 d_model,
                                 patch_size,
                                 patch_size,
                                 0);
module_add_child(&vlm->base, "patch_embed", &vlm->patch_embed->base);

vlm->lm = decoder_lm_create(params_pool, vocab_size, d_model,
                            n_layers, n_heads, d_k, intermediate_size);
module_add_child(&vlm->base, "lm", &vlm->lm->base);

if (use_image_pos) {
    vlm->image_pos = tensor_zeros(params_pool, 3,
        (int[]){1, n_img_tokens, d_model}, 1);
    module_param(&vlm->base, "image_pos", vlm->image_pos);
} else {
    vlm->image_pos = NULL;
}
```

Important: set all shape fields before allocating `image_pos`:

```c
vlm->patch_h = image_h / patch_size;
vlm->patch_w = image_w / patch_size;
vlm->n_img_tokens = vlm->patch_h * vlm->patch_w;
```

Parameter collection must then work with:

```c
int n_params;
tensor **params = module_parameters(&vlm->base, &n_params);
```

Add count helper:

```c
long long vision_lm_num_parameters(vision_lm *vlm);
```

Implementation:

```c
return module_num_parameters(&vlm->base);
```

### Weight init

```c
void vision_lm_init_weights(vision_lm *vlm);
```

Required behavior:
- call `decoder_lm_init_weights(vlm->lm)`
- initialize `patch_embed->weight` with Xavier-style normal std:

```text
std = sqrt(2 / (image_channels * patch_size * patch_size + d_model))
```

- zero `patch_embed->bias`
- initialize `image_pos` with normal `std=0.02` if present

Implementation detail: `src/gpt.c` has `_randn()` static. For this task, duplicate a small static Box-Muller helper in `src/vlm.c`.

### RoPE passthrough

```c
void vision_lm_enable_rope(struct mem_pool *params_pool,
                           vision_lm *vlm,
                           int max_text_len,
                           float base);
```

Implementation:

```c
decoder_lm_enable_rope(params_pool, vlm->lm,
                       vlm->n_img_tokens + max_text_len,
                       base);
```

Reason: combined sequence length includes image prefix tokens.

Guardrail: `vision_lm_generate()` must assert that enabled RoPE length covers `I + prompt_len + max_new_tokens` when RoPE is enabled. No dynamic RoPE growth in this task.

### Patch embedding forward

```c
tensor *vision_lm_image_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images);
```

Input:

```text
images: [B,C,H,W], float, contiguous
```

Output:

```text
img_embeds: [B,I,D]
I = (H/patch_size) * (W/patch_size)
```

Implementation:

```c
tensor *x = conv2d_forward(scratch, vlm->patch_embed, images);  // [B,D,Hp,Wp]

x = tensor_transpose(scratch, x, 1, 2);  // [B,Hp,D,Wp]
x = tensor_transpose(scratch, x, 2, 3);  // [B,Hp,Wp,D]
x = tensor_contiguous(scratch, x);

int B = images->shape[0];
tensor *img = tensor_reshape(scratch, x, 3,
    (int[]){B, vlm->n_img_tokens, vlm->d_model});

if (vlm->image_pos)
    img = tensor_add(scratch, img, vlm->image_pos);  // broadcast [1,I,D]

return img;
```

Flatten order must be row-major patches:

```text
patch index = row * patch_w + col
```

This order follows `[B,D,Hp,Wp] -> [B,Hp,Wp,D] -> [B,I,D]`.

`tensor_add` broadcast backward for `[B,I,D] + [1,I,D]` must be tested. If it fails, fix broadcast grad in core ops rather than adding VLM-local expansion.

Important assertions:
- `images->ndim == 4`
- `images->shape[1] == image_channels`
- `images->shape[2] == image_h`
- `images->shape[3] == image_w`
- `tensor_is_contiguous(images)`

Patch embedding must preserve autograd into:
- input image if `images->requires_grad` (rare but supported)
- `patch_embed->weight`
- `patch_embed->bias`
- `image_pos`

### Multimodal embedding builder

```c
tensor *vision_lm_build_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images,
                               const tensor *text_ids);
```

Inputs:

```text
images:   [B,C,H,W]
text_ids: [B,T], int tensor stored in tensor data region
```

Implementation:

```c
assert(images->shape[0] == text_ids->shape[0]);

img = vision_lm_image_embeds(scratch, vlm, images);       // [B,I,D]
txt = decoder_lm_token_embeds(scratch, vlm->lm, text_ids); // [B,T,D]

return tensor_cat(scratch, img, txt, 1);                  // [B,I+T,D]
```

### Forward

```c
tensor *vision_lm_forward(struct mem_pool *scratch,
                          vision_lm *vlm,
                          const tensor *images,
                          const tensor *text_ids);
```

Returns:

```text
logits [B, I+T, vocab_size]
```

Implementation:

```c
embeds = vision_lm_build_embeds(scratch, vlm, images, text_ids);
return decoder_lm_forward_embeds(scratch, vlm->lm, embeds);
```

### Text-only logits helper

```c
tensor *vision_lm_forward_text_logits(struct mem_pool *scratch,
                                      vision_lm *vlm,
                                      const tensor *images,
                                      const tensor *text_ids);
```

Returns only logits aligned to text input positions:

```text
[B,T,vocab_size]
```

Implementation:

```c
int T = text_ids->shape[1];
logits = vision_lm_forward(...);                         // [B,I+T,V]
return tensor_slice(scratch, logits, 1, vlm->n_img_tokens, T);
```

Alignment convention:
- `input_ids[b,t]` is token visible at position `I+t`.
- `logits_text[b,t]` predicts `target_ids[b,t]`.
- For next-token LM, caller should pass `input_ids = [BOS, y0, ..., y_{T-2}]`, `target_ids = [y0, ..., y_{T-1}]`.

---

## `include/vlm.h` prototype checklist

Final header must expose:

```c
vision_lm *vision_lm_create(struct mem_pool *params_pool,
                            int vocab_size, int d_model,
                            int n_layers, int n_heads, int d_k,
                            int intermediate_size,
                            int image_channels, int image_h, int image_w,
                            int patch_size, int use_image_pos);

void vision_lm_init_weights(vision_lm *vlm);
void vision_lm_enable_rope(struct mem_pool *params_pool,
                           vision_lm *vlm,
                           int max_text_len,
                           float base);
long long vision_lm_num_parameters(vision_lm *vlm);

tensor *vision_lm_image_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images);

tensor *vision_lm_build_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images,
                               const tensor *text_ids);

tensor *vision_lm_forward(struct mem_pool *scratch,
                          vision_lm *vlm,
                          const tensor *images,
                          const tensor *text_ids);

tensor *vision_lm_forward_text_logits(struct mem_pool *scratch,
                                      vision_lm *vlm,
                                      const tensor *images,
                                      const tensor *text_ids);

tensor *vision_lm_shift_targets(struct mem_pool *pool,
                                const tensor *text_ids);

tensor *vision_lm_train_step(struct mem_pool *scratch_pool,
                             vision_lm *vlm,
                             const tensor *images,
                             const tensor *input_ids,
                             const tensor *target_ids,
                             adamw_opt *opt,
                             float grad_clip,
                             float *grad_norm_out);

int *vision_lm_generate(struct mem_pool *scratch_pool,
                        struct mem_pool *data_pool,
                        vision_lm *vlm,
                        const tensor *image,
                        const tensor *prompt_ids,
                        int max_new_tokens,
                        float temperature,
                        int use_cache,
                        int *n_out);
```

Do not declare `vision_lm_train_step_masked()` until P1 masked CE implementation lands. No dangling declaration.

---

## Training API

### Target/input preparation

VLM train step expects already shifted text tensors. Add helper:

```c
tensor *vision_lm_shift_targets(struct mem_pool *pool,
                                const tensor *text_ids);
```

This reuses `decoder_lm_shift_targets(pool, text_ids)` to create `[B,T-1]` targets. Caller can create input view with:

```c
tensor *input_ids = tensor_slice(scratch, text_ids, 1, 0, T - 1);
tensor *target_ids = vision_lm_shift_targets(data_pool, text_ids);
```

Need caution: input view is scratch-owned; keep scratch alive through forward/backward.

### Simple full-text LM loss

```c
tensor *vision_lm_train_step(struct mem_pool *scratch_pool,
                             vision_lm *vlm,
                             const tensor *images,
                             const tensor *input_ids,
                             const tensor *target_ids,
                             adamw_opt *opt,
                             float grad_clip,
                             float *grad_norm_out);
```

Expected shapes:

```text
images:     [B,C,H,W]
input_ids:  [B,T]    e.g. [BOS, y0, y1, ..., y_{T-2}]
target_ids: [B,T]    e.g. [y0,  y1, y2, ..., y_{T-1}]
```

Flow:

```c
adamw_zero_grad(opt);

logits_text = vision_lm_forward_text_logits(scratch_pool, vlm, images, input_ids);
loss = tensor_cross_entropy(scratch_pool, logits_text, target_ids, 2);

dnn_backward(scratch_pool, loss);
clip/grad_norm;
adamw_step(opt);
return loss;
```

This trains every text position. Image-prefix positions are never targets.

Assertions:
- `input_ids->shape == target_ids->shape == [B,T]`
- `images->shape[0] == input_ids->shape[0]`
- `T >= 1`

Data pool note: `target_ids` is saved by cross-entropy backward. Do not reset data pool until after `dnn_backward()`.

### Byte-level ImageNet/class-label training policy

For ImageNet-style classification with byte-level tokenizer only, labels are text byte sequences. Do **not** add special class tokens. Use `vocab_size = TOKENIZER_VOCAB_SIZE` (261) for byte-level VLM.

Use tokenizer special IDs from `include/tokenizer.h`:

```c
TOKENIZER_BOS_ID = 257
TOKENIZER_EOS_ID = 258
TOKENIZER_PAD_ID = 259
```

For raw label bytes `b[0..M-1]`, append EOS to form target sequence `y = [b0..b{M-1}, EOS]` with length `L = M + 1`:

```text
input_ids:  [BOS, b0, b1, ..., b{M-1}]
target_ids: [b0,  b1, b2, ..., EOS]
T = L
```

Initial no-padding training uses exact-length buckets:

```text
bucket key = T = byte_len(label) + 1(EOS)
images     [B,C,H,W]
input_ids  [B,T]
target_ids [B,T]
```

This is not GPT stream packing. Do not concatenate multiple image/label examples into one row:

```text
BAD: [imgA][labelA][imgB][labelB]
```

Reason: one row has one image prefix semantics. Multi-example packing needs segment/block masks and is out of scope.

Classification evaluation must score candidate byte-label sequences, not rely only on greedy free-form generation:

```text
score(label | image) = sum_t log p(label_t | image, BOS, label_<t)
```

Initial simple eval can run 1000 class labels independently and choose max score. Optimized eval can later cache image prefix and use a trie over byte labels.

### Proper masked loss for instruction tuning

For real VLM instruction tuning, only answer tokens contribute loss. Current `tensor_cross_entropy()` has no ignore-index/loss-mask support.

Add general op in `src/ops_activation.c` + declaration in `include/ops.h`:

```c
tensor *tensor_cross_entropy_masked(struct mem_pool *scratch,
                                    const tensor *logits,   // [B,T,V]
                                    const tensor *target,   // [B,T] int
                                    const tensor *mask,     // [B,T] float 0/1
                                    int dim);
```

Do not add VLM-local CE duplicate.

Masked loss semantics:

```text
loss = sum(mask[b,t] * CE(logits[b,t], target[b,t])) / max(sum(mask), 1)
```

Backward:

```text
dlogits[b,t,c] = mask[b,t] / sum(mask) * (softmax[c] - one_hot[target])
```

P1 train step:

```c
tensor *vision_lm_train_step_masked(struct mem_pool *scratch_pool,
                                    vision_lm *vlm,
                                    const tensor *images,
                                    const tensor *input_ids,
                                    const tensor *target_ids,
                                    const tensor *loss_mask,
                                    adamw_opt *opt,
                                    float grad_clip,
                                    float *grad_norm_out);
```

Decision:
- `vision_lm_train_step()` with full text CE is required for fixed-length/from-scratch caption training.
- `tensor_cross_entropy_masked()` and padded train APIs are P1 for variable-length byte labels in mixed-length batches.
- `vision_lm_train_step_masked()` is P1 and required before instruction/chat fine-tuning.
- Do not block initial exact-length-bucketed caption/class-label pretraining support on masked CE.

---

## P1: padding support with optimized attention lengths

Current code assumes rectangular fixed-length `[B,T]` text. That is OK for initial VLM handoff if dataloader uses exact-length buckets (all samples in a batch have same byte-label length). For flexible byte-level labels and prompts, add P1 padding support.

### Why padding support matters

Byte-level labels are variable length:

```text
"tench"              -> 5 bytes + EOS
"golden retriever"   -> 16 bytes + EOS
"Australian terrier" -> 18 bytes + EOS
```

Exact-length bucketing works but can create many small buckets. Better batching uses length ranges:

```text
bucket 1: T=1..8
bucket 2: T=9..16
bucket 3: T=17..24
```

Within each batch:

```text
input_ids  [B,Tmax]
target_ids [B,Tmax]
loss_mask  [B,Tmax]  // 1 for real target positions, 0 for PAD
text_lens  [B]       // valid input length per sample
```

### Required P1 pieces

1. Masked CE:

```c
tensor *tensor_cross_entropy_masked(struct mem_pool *scratch,
                                    const tensor *logits,   // [B,Tmax,V]
                                    const tensor *target,   // [B,Tmax] int
                                    const tensor *mask,     // [B,Tmax] float 0/1
                                    int dim);
```

2. Attention effective lengths:

Add per-sample combined sequence lengths to extended attention:

```c
tensor *tensor_attention_ex(struct mem_pool *scratch,
                            tensor *q,
                            tensor *k,
                            tensor *v,
                            tensor *mask,
                            attention_mode mode,
                            int prefix_len,
                            const int *seq_lens);  // nullable [B], combined valid lengths
```

Semantics:
- `seq_lens == NULL`: current fixed rectangular behavior.
- `seq_lens[b]`: valid combined length for batch item `b`.
- VLM padded batch: `seq_lens[b] = vlm->n_img_tokens + text_lens[b]`.
- GPT text-only padded batch: `seq_lens[b] = text_lens[b]`.

3. Padded VLM train step:

```c
tensor *vision_lm_train_step_padded(struct mem_pool *scratch_pool,
                                    vision_lm *vlm,
                                    const tensor *images,      // [B,C,H,W]
                                    const tensor *input_ids,   // [B,Tmax]
                                    const tensor *target_ids,  // [B,Tmax]
                                    const tensor *loss_mask,   // [B,Tmax]
                                    const int *text_lens,      // [B]
                                    adamw_opt *opt,
                                    float grad_clip,
                                    float *grad_norm_out);
```

This is P1, not initial fixed-length handoff.

Padded implementation builds:

```c
combined_lens[b] = vlm->n_img_tokens + text_lens[b];
embeds = vision_lm_build_embeds(...); // [B,I+Tmax,D]
logits = decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                                      ATTENTION_PREFIX_LM,
                                      vlm->n_img_tokens,
                                      combined_lens);
```

### Optimized attention behavior

Do not implement padding as only a giant additive mask. That would still compute pad rows/cols. Use `seq_lens` to skip compute.

In `attention_forward_triangular()` / prefix-LM variant:

```c
int Nmax = q->shape[2];
int N_eff = seq_lens ? seq_lens[b] : Nmax;

for (int r0 = 0; r0 < N_eff; r0 += TB) {
    int r1 = min(r0 + TB, N_eff);
    int S = attention_tile_S(mode, r0, r1, prefix_len);
    if (S > N_eff) S = N_eff;
    ...
}

// padded query rows produce zero output
if (N_eff < Nmax)
    memset(o_slice + N_eff*d, 0, (size_t)(Nmax - N_eff) * d * sizeof(float));
```

Backward mirrors forward:

```c
int N_eff = seq_lens ? seq_lens[b] : Nmax;
for (r0 = 0; r0 < N_eff; r0 += TB) { ... }
// no dQ/dK/dV contribution from pad rows/cols
```

Loss mask ensures pad queries have no loss contribution. Attention `seq_lens` ensures real queries cannot attend pad keys and no FLOPs are spent on pad rows/cols.

### VLM padded semantics

For batch item `b`:

```text
combined sequence: [image I tokens][text Tmax tokens]
valid combined:    [image I tokens][text text_lens[b] tokens]
pad positions:     I + text_lens[b] .. I + Tmax - 1
```

Allowed attention with prefix-LM + padding:

```c
N_eff = I + text_lens[b];

if (qi >= N_eff)      no useful query; output zero
else if (qi < I)      kj < I
else                  kj < I || kj <= qi

kj must also be < N_eff
```

Implementation via `visible_len` and `S`:

```c
visible = attention_visible_len(mode, qi, prefix_len);
if (visible > N_eff) visible = N_eff;
S = attention_tile_S(mode, r0, r1, prefix_len);
if (S > N_eff) S = N_eff;
```

### Padded dataloader policy

Use bucketed padding, not arbitrary global padding:

- bucket samples by text length ranges
- choose `Tmax` per batch from bucket max
- keep `Tmax / avg(T)` close to 1
- set padded positions to `TOKENIZER_PAD_ID` (259); loss mask and `seq_lens` make PAD ignored

### What not to implement yet

Do not implement multi-example sequence packing such as:

```text
[imgA][labelA][imgB][labelB]
```

That requires segment IDs and block-diagonal prefix/causal masks:

```c
segment_id[b, pos]
```

It is more complex than padded batches and not needed once bucketed padding exists.

---

## Generation API

### Public function

```c
int *vision_lm_generate(struct mem_pool *scratch_pool,
                        struct mem_pool *data_pool,
                        vision_lm *vlm,
                        const tensor *image,
                        const tensor *prompt_ids,
                        int max_new_tokens,
                        float temperature,
                        int use_cache,
                        int *n_out);
```

Initial constraints:
- `image` shape `[1,C,H,W]` only
- `prompt_ids` shape `[1,T]`
- `max_new_tokens > 0`
- `temperature >= 0`; `0` means greedy argmax
- returns generated text token array allocated from `data_pool`
- output includes prompt text + generated text, not image tokens
- image prefix is implicit
- generation must run inside `dnn_no_grad_enter()` / `dnn_no_grad_exit()`

### No-cache generation

Simplest path:

```c
while cur_len < max_len:
    ids_tensor = [1,cur_len] from output ids
    logits = vision_lm_forward(scratch, vlm, image, ids_tensor)
    last_logits = logits[0, I + cur_len - 1, :]
    sample/argmax next token
    append
    mem_pool_reset(scratch)
```

This mirrors `decoder_lm_generate(... use_cache=0)` but calls VLM forward.

Memory detail:
- `ids_tensor` can live in `data_pool` each step.
- copy last logits into `data_pool` before `mem_pool_reset(scratch)`.
- take a `data_pool` mark inside each loop before allocating `ids_tensor` / `last_logit_buf`, then release it after sampling so output buffer survives and data growth stays bounded.

### Cached generation

Use existing `transformer_block_forward_cached()`.

Prefix prefill:

```c
img = vision_lm_image_embeds(scratch, vlm, image);       // [1,I,D]
txt = decoder_lm_token_embeds(scratch, lm, prompt_ids);  // [1,T,D]
h = tensor_cat(scratch, img, txt, 1);                    // [1,I+T,D]

for each layer l:
    h = transformer_block_forward_cached(scratch, lm->blocks[l], h, caches[l]);

h = rms_norm_forward(scratch, lm->norm, h);
logits = decoder_lm_lm_head_forward(scratch, lm, h);
sample from logits[0, I+T-1, :]
```

Current `transformer_block_forward_cached()` supports `N_new > 1`, but its inline attention is causal-only. For VLM cached generation, add a cached extended path so prefix prefill uses prefix-LM image attention:

```c
tensor *transformer_block_forward_cached_ex(struct mem_pool *scratch,
                                            transformer_block *block,
                                            const tensor *x,
                                            kv_cache *cache,
                                            attention_mode mode,
                                            int prefix_len);
```

Use it for initial image+prompt prefill only, when `cache->seq_len == 0`:

```c
h = transformer_block_forward_cached_ex(scratch, block, h, cache,
                                        ATTENTION_PREFIX_LM,
                                        vlm->n_img_tokens);
```

Implementation rule: `transformer_block_forward_cached_ex(... ATTENTION_PREFIX_LM ...)` must assert `cache->seq_len == 0` for this handoff. Its inline cached attention must use the same visible-length rule as `tensor_attention_ex`:

```c
visible = (global_q < prefix_len) ? prefix_len : global_q + 1;
```

After prefill, use normal causal cached one-token path.

Reason: plain cached causal prefill would make `img_0` unable to attend `img_1..img_{I-1}`, breaking bidirectional image-prefix semantics. Do not implement token-by-token image prefill for VLM.

Then per-token loop:

```c
single_id -> text embedding [1,1,D]
h = reshape to [1,1,D]
for each layer l:
    h = transformer_block_forward_cached(scratch, block[l], h, caches[l]);
h = norm(h)
logits = lm_head(h)
sample next
```

Cache sizing:

```c
max_seq = vlm->n_img_tokens + prompt_len + max_new_tokens;
```

Allocate caches from `data_pool`, not `scratch_pool`, same reason as `decoder_lm_generate()`.

Use a `data_pool` mark after output allocation and before cache allocation, then release it at end so output survives:

```c
int *output = _mem_pool_alloc(data_pool, max_len * sizeof(int), NULL);
size_t cache_mark = mem_pool_mark(data_pool);
// allocate caches + temp data after mark
...
mem_pool_release(data_pool, cache_mark);
```

RoPE works if `vision_lm_enable_rope()` used `max_seq >= I + prompt + max_new`.

Sampling helpers:
- use `decoder_lm_argmax_token()` / `decoder_lm_sample_with_temp()` exposed from GPT helpers

Keep EOS behavior same as GPT (`TOKENIZER_EOS_ID`).

Critical cached-vs-no-cache equivalence test:
- temperature `0`
- same weights/image/prompt
- cached output must equal no-cache output token-for-token

---

## Header include/export requirements

Add `include/vlm.h`.

Update `include/dnn.h`:

```c
#include "gpt.h"
#include "vlm.h"
```

Current `dnn.h` includes `transformer.h` but not `gpt.h`; VLM depends on `decoder_lm`, so include chain should be explicit. Put `gpt.h` before `vlm.h`.

Makefile already uses wildcard `src/*.c`, so adding `src/vlm.c` is enough for library build.

---

## Shape examples

For image `224×224`, patch `16`, `d_model=768`:

```text
I = (224/16)^2 = 196
image [B,3,224,224]
conv  [B,768,14,14]
flat  [B,196,768]
text  [B,T,768]
cat   [B,196+T,768]
```

For small test config:

```text
B=2, C=3, H=W=8, patch=4, D=16
I=4
T=5
combined length = 9
```

---

## Positional encoding policy

Required for this task:
- learned `image_pos [1,I,D]` support via `use_image_pos`
- existing GPT 1D RoPE over combined sequence positions when RoPE is enabled
- image tokens occupy positions `0..I-1`
- text tokens occupy positions `I..I+T-1`

Decision: **no 2D RoPE in this handoff**.

Why defer 2D RoPE:
- image Q/K would need row/col rotary rotations, different from text 1D RoPE
- mixed image/text sequence needs clear policy for image-image, text-text, and text-image dot products
- current `tensor_rope()` assumes one 1D position axis
- learned image position embeddings are enough for first VLM

Future work:
- 2D image RoPE or 2D learned row/col embedding
- modality/type embedding (`image_type`, `text_type`) so model can distinguish image vs text tokens beyond position

Deferred future params, not part of this task:

```c
tensor *image_row_pos;  // [Hp,D]
tensor *image_col_pos;  // [Wp,D]
tensor *modality_pos;   // [2,D]
```

Attention decision:
- VLM must use prefix-LM attention, not plain causal attention.
- Image tokens are bidirectional among themselves.
- Text tokens can attend every image patch and causal previous/current text tokens.

---

## Fundamental: bidirectional image-prefix attention

For proper VLM behavior, image patch tokens should be able to attend all other image patch tokens, while text remains causal. This is **prefix-LM attention**.

Combined sequence layout:

```text
[ img_0 ... img_{I-1} | txt_0 ... txt_{T-1} ]
N = I + T
```

Allowed attention:

```c
if (qi < I) {
    allowed = (kj < I);              // image query sees all image keys, no text keys
} else {
    allowed = (kj < I) || (kj <= qi); // text query sees all image + causal text
}
```

Equivalent row visible lengths for this layout:

```c
if (qi < I) visible_len = I;
else        visible_len = qi + 1;
```

Important: this **cannot be implemented with current additive `mask` alone**, because current `tensor_attention()` always applies implicit causal masking via row-prefix `visible = i + 1`. An additive mask can hide more entries but cannot unhide future image tokens.

### Current `src/attention.c` status

Current attention code has already moved to triangular row-prefix attention:

- `attention_forward_triangular(...)`
- `attention_backward_triangular(...)`
- row helper: `attention_softmax_row_prefix(scores, p, visible_len)`
- backward helper: `attention_softmax_bwd_row_prefix(p, dp, ds, visible_len, scale)`
- per tile uses `S = r1` and `visible = i + 1`

That means prefix-LM support is a small generalization:

```c
visible = attention_visible_len(mode, i, prefix_len, N);
S       = max visible_len over tile rows;
```

For causal mode today:

```c
visible = i + 1;
S = r1;
```

For VLM prefix-LM mode:

```c
visible = (i < prefix_len) ? prefix_len : i + 1;
S = (r1 <= prefix_len) ? prefix_len : r1;
```

This preserves rectangular BLAS tiles and avoids per-element `allowed()` checks in hot loops.

### Required API changes

Add attention mode enum to `include/attention.h`:

```c
typedef enum attention_mode {
    ATTENTION_CAUSAL = 0,      // current behavior: row i sees 0..i
    ATTENTION_PREFIX_LM = 1    // prefix_len tokens are bidirectional prefix
} attention_mode;
```

Add extended attention API:

```c
tensor *tensor_attention_ex(struct mem_pool *scratch,
                            tensor *q,
                            tensor *k,
                            tensor *v,
                            tensor *mask,
                            attention_mode mode,
                            int prefix_len,
                            const int *seq_lens);  // nullable [B], P1 padding support
```

Keep existing API as compatibility wrapper:

```c
tensor *tensor_attention(struct mem_pool *scratch,
                         tensor *q, tensor *k, tensor *v, tensor *mask) {
    return tensor_attention_ex(scratch, q, k, v, mask,
                               ATTENTION_CAUSAL, 0, NULL);
}
```

Validation:

```c
if (mode == ATTENTION_PREFIX_LM)
    assert(prefix_len > 0 && prefix_len <= N);
if (mode == ATTENTION_CAUSAL)
    assert(prefix_len == 0);
```

### Required transformer API changes

Current `transformer_block_forward()` calls causal `tensor_attention()`. VLM needs prefix mode.

Add:

```c
tensor *transformer_block_forward_ex(struct mem_pool *scratch,
                                     transformer_block *block,
                                     const tensor *x,
                                     attention_mode mode,
                                     int prefix_len,
                                     const int *seq_lens);  // nullable [B]
```

Then keep old wrapper:

```c
tensor *transformer_block_forward(struct mem_pool *scratch,
                                  transformer_block *block,
                                  const tensor *x) {
    return transformer_block_forward_ex(scratch, block, x,
                                        ATTENTION_CAUSAL, 0, NULL);
}
```

Add cached extended wrapper too:

```c
tensor *transformer_block_forward_cached_ex(struct mem_pool *scratch,
                                            transformer_block *block,
                                            const tensor *x,
                                            kv_cache *cache,
                                            attention_mode mode,
                                            int prefix_len);
```

`transformer_block_forward_cached()` remains wrapper with `ATTENTION_CAUSAL, 0`.

Inside block forward, replace:

```c
tensor *attn_out = tensor_attention(scratch, Qh, Kh, Vh, NULL);
```

with:

```c
tensor *attn_out = tensor_attention_ex(scratch, Qh, Kh, Vh, NULL,
                                       mode, prefix_len, seq_lens);
```

Add extended GPT helpers:

```c
tensor *decoder_lm_hidden_from_embeds_ex(struct mem_pool *scratch,
                                         decoder_lm *lm,
                                         const tensor *embeds,
                                         attention_mode mode,
                                         int prefix_len,
                                         const int *seq_lens);  // nullable [B]

tensor *decoder_lm_forward_embeds_ex(struct mem_pool *scratch,
                                     decoder_lm *lm,
                                     const tensor *embeds,
                                     attention_mode mode,
                                     int prefix_len,
                                     const int *seq_lens);  // nullable [B]
```

VLM forward must call:

```c
decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                             ATTENTION_PREFIX_LM,
                             vlm->n_img_tokens,
                             NULL);
```

Do not duplicate block loop in `src/vlm.c`; use GPT `_ex` helpers.

### Prefix-LM forward optimization

Do **not** implement prefix-LM with generic per-`(i,j)` branch checks. Use row visible lengths.

Modify triangular forward:

```c
static inline int attention_visible_len(attention_mode mode, int i,
                                        int prefix_len) {
    switch (mode) {
    case ATTENTION_CAUSAL:
        return i + 1;
    case ATTENTION_PREFIX_LM:
        return (i < prefix_len) ? prefix_len : i + 1;
    default:
        assert(0 && "unknown attention mode");
        return i + 1;
    }
}

static inline int attention_tile_S(attention_mode mode, int r0, int r1,
                                   int prefix_len) {
    (void)r0;
    switch (mode) {
    case ATTENTION_CAUSAL:
        return r1;
    case ATTENTION_PREFIX_LM:
        return (r1 <= prefix_len) ? prefix_len : r1;
    default:
        assert(0 && "unknown attention mode");
        return r1;
    }
}
```

Then replace current hardcoded values in `attention_forward_triangular()`:

```c
int S = r1;
...
int visible = i + 1;
```

with:

```c
int S = attention_tile_S(mode, r0, r1, prefix_len);
...
int visible = attention_visible_len(mode, i, prefix_len);
```

Everything else remains the same:

- compute `scores = Q_tile @ K[0:S]^T`
- softmax first `visible` entries
- zero `p_tile[visible:S]`
- save prefix into full `P` row
- compute `O_tile = p_tile @ V[0:S]`

For image-only tiles (`r1 <= I`):

```text
S = I
visible = I for every row
```

So BLAS computes dense image-image attention once per image-row tile.

For text tiles:

```text
S = r1
visible = i + 1
```

Because image tokens are before text, normal row prefix already includes all image keys plus causal text keys.

### Prefix-LM backward optimization

Mirror forward in `attention_backward_triangular()`:

Replace hardcoded:

```c
int S = r1;
int visible = i + 1;
```

with mode-aware values:

```c
int S = attention_tile_S(mode, r0, r1, prefix_len);
int visible = attention_visible_len(mode, i, prefix_len);
```

Then existing BLAS calls still work:

```c
dV[0:S] += P_tile^T @ dO_tile
dP_tile = dO_tile @ V[0:S]^T
dS_tile = softmax_bwd over visible prefix, zero visible:S
dQ_tile += dS_tile @ K[0:S]
dK[0:S] += dS_tile^T @ Q_tile
```

Need to pass `mode` and `prefix_len` through autograd saved tensors:

```c
fn->n_saved = 5;
fn->saved_tensors[0] = P;
fn->saved_tensors[1] = scale_saved;
fn->saved_tensors[2] = mask_flag;
fn->saved_tensors[3] = mode_saved;
fn->saved_tensors[4] = prefix_len_saved;
```

Backward reads these and calls `attention_backward_triangular(... mode, prefix_len)`.

### Perf model

Prefix-LM allowed pair count:

```text
image -> image: I²
text  -> image: T*I
text  -> text:  T(T+1)/2

total_prefix = I² + T*I + T(T+1)/2
```

Plain causal over combined sequence:

```text
total_causal = (I+T)(I+T+1)/2
             ≈ I²/2 + I*T + T²/2
```

Extra prefix-LM cost over causal:

```text
≈ I²/2
```

Example `I=196`, `T=128`:

```text
causal pairs:    ~52k
prefix-LM pairs: ~72k
attention ratio: ~1.37x
```

End-to-end hit is usually smaller because FFN and LM head also cost time.

### Optimization priorities

1. **Keep region-specialized row-prefix logic**
   - image rows: visible `I`
   - text rows: visible `I + t + 1`
   - no `if (allowed)` inside inner softmax loop

2. **Keep triangular tiling**
   - image block uses rectangular `[M,I]` GEMMs
   - text block uses rectangular prefix `[M,r1]` GEMMs
   - worker scratch remains `TB*N`, not `N*N`

3. **Reduce image token count when possible**
   - attention has `I²`
   - `224/16 => I=196` OK
   - `336/14 => I=576` much heavier
   - later add image token pooling / patch merger if high-res needed

4. **Saved `P` memory**
   - first implementation can keep full `[B,H,N,N]` `P`
   - prefix-LM packed `P` can save memory later:
     ```text
     I² + T*I + T(T+1)/2 floats per head/batch
     ```
   - Flash/recompute mode later if long image/text sequences overflow scratch

5. **Cached generation remains efficient**
   - prefill image+prompt with `transformer_block_forward_cached()` using `N_new=I+T`
   - after prefill, each new text token attends all cached image+text tokens
   - per-token cost increases only by image prefix length in cache

### Tests for bidirectional image attention

Add attention-level tests:

1. `prefix_len=I`, `N=I+T`, compare against explicit dense mask reference.
2. Image query row can attend later image key:
   - construct values so `img_0` output changes when `img_{I-1}` changes.
3. Image query row cannot attend text key:
   - changing text value should not affect image outputs.
4. Text query can attend all image keys and only previous text keys.
5. Backward finite-diff for prefix-LM mode.
6. Existing causal tests still pass through wrapper.

Add VLM-level tests:

1. With prefix-LM mode, image patch order interactions affect image hidden states.
2. Cached/no-cache generation still matches for text generation.
3. RoPE max length uses `I+T+new`, not just text length.

### Benchmark additions

Benchmark attention modes:

| I | T | N | mode | expected |
|-:|-:|-:|------|----------|
| 196 | 128 | 324 | causal | baseline |
| 196 | 128 | 324 | prefix-LM | ~1.3-1.4x attention time |
| 576 | 128 | 704 | causal | baseline |
| 576 | 128 | 704 | prefix-LM | image `I²` dominates |

Report:
- forward no-grad
- forward+backward
- scratch peak
- VLM train step with causal vs prefix-LM

---

## State/loading considerations

Because `vision_lm` embeds a full `decoder_lm` child, model save paths should look like:

```text
patch_embed.weight
patch_embed.bias
image_pos
lm.embed.weight
lm.blocks.0.qkv_proj.weight
...
lm.norm.weight
lm.lm_head.bias
```

Tied LM head weight remains `lm.embed.weight`; do not register duplicate `lm_head.weight`.

GPT-only weight reuse decision: load GPT checkpoints into the child LM directly, not through the VLM root:

```c
module_load(&vlm->lm->base, path, strict);
```

Do not add checkpoint key-prefix rewriting in this task.

---

## Limitations / caveats

1. **Conv patch embedding is not ViT-level preprocessing**
   - User must pass normalized float images in BCHW.
   - No resize/crop/mean/std pipeline in core library.

2. **Fixed image size first**
   - Constructor stores `image_h/image_w` and asserts input matches.
   - Variable image sizes require dynamic `n_img_tokens`, dynamic image pos, and variable RoPE max length.

3. **Prefix-LM decoder, not ViT encoder**
   - Text can attend all image tokens.
   - Image tokens attend all image tokens via prefix-LM attention.
   - This is still a decoder-only prefix architecture, not a standalone bidirectional ViT encoder.

4. **No cross-attention**
   - Architecture is prefix-LM, not Flamingo-style cross-attention.

5. **Masked loss is P1 for padding/instruction tuning**
   - Simple full-text LM training is enough for exact-length-bucketed caption/from-scratch training.
   - `tensor_cross_entropy_masked` is required for padded variable-length batches.
   - `vision_lm_train_step_masked` is required before instruction/chat fine-tuning, but not blocker for initial fixed-length VLM support.

6. **Generation only batch=1 first**
   - Matches current GPT generation limitation.

7. **Image prefix recompute in no-cache generation**
   - No-cache path recomputes image embeddings every step.
   - Cached path avoids repeated attention over image prefix after prefill.

8. **No image preprocessing**
   - Pixel normalization and channel order are caller responsibility.
   - Core expects float BCHW.

9. **No variable patch size per sample**
   - Batch must have fixed `H/W` and same patch grid.

10. **No multi-example sequence packing**
   - P1 padding support handles variable text lengths.
   - Packing multiple image/text examples into one row needs segment masks and is out of scope.

---

## Tests to add

Create required `test/test_vlm.c`. PyTorch `test/ref_vlm.py` is not required for this handoff.

### Unit tests

1. Create tiny VLM:

```text
vocab=16, d_model=8, n_layers=1, n_heads=2, d_k=4, intermediate=16
image: C=3,H=8,W=8,patch=4 => I=4
text T=5
```

2. `vision_lm_image_embeds()` shape is `[B,4,8]`.
3. `vision_lm_build_embeds()` shape is `[B,9,8]`.
4. `vision_lm_forward()` shape is `[B,9,16]`.
5. `vision_lm_forward_text_logits()` shape is `[B,5,16]`.
6. `vision_lm_train_step()` returns finite scalar loss and changes at least one param.
7. Gradients exist for:
   - patch conv weight/bias
   - image_pos if enabled
   - token embedding
   - at least one transformer param
8. `module_parameters(&vlm->base)` includes patch, image_pos, and LM params.
9. `module_num_parameters(&vlm->base)` equals manual count for tiny config.
10. Generation no-cache returns valid token IDs and preserves prompt prefix.
11. Cached generation matches no-cache for greedy `temperature=0` on tiny deterministic model.
12. RoPE-enabled VLM cached/no-cache generation matches for short prompt.
13. Constructor asserts invalid image size not divisible by patch size.
14. P1 padded test: mixed text lengths in one batch match exact-length unpadded losses for same examples.
15. P1 masked CE test: `vision_lm_train_step_masked()` ignores mask=0 positions once masked CE lands.

### Reference checks inside C tests

C tests must explicitly verify:
- Conv2d patch flatten order `[B,D,Hp,Wp] -> [B,Hp,Wp,D] -> [B,I,D]`
- Concat image/text sequence order
- Text-logit slice starts at `vlm->n_img_tokens`
- Prefix-LM attention permits image-to-later-image and blocks image-to-text

---

## Benchmarks to add

Add `bench/bench_vlm.c` as part of this task:

Configs:

| image | patch | I | text T | D | layers | purpose |
|------:|------:|--:|-------:|--:|-------:|---------|
| 64x64 | 16 | 16 | 32 | 128 | 2 | smoke |
| 224x224 | 16 | 196 | 64 | 256 | 4 | realistic small |
| 224x224 | 14 | 256 | 128 | 512 | 8 | stress |

Measure:
- patch embedding time
- multimodal forward time
- train step time
- generation no-cache vs cached

---

## Implementation order

1. Add attention/transformer prefix-LM support:
   - `attention_mode`
   - `tensor_attention_ex(... mode, prefix_len, seq_lens)`
   - `transformer_block_forward_ex(... mode, prefix_len, seq_lens)`
   - `transformer_block_forward_cached_ex(... mode, prefix_len)` for prefix prefill
2. Add GPT helpers:
   - `decoder_lm_token_embeds`
   - `decoder_lm_hidden_from_embeds`
   - `decoder_lm_forward_embeds`
   - `decoder_lm_lm_head_forward`
   - `_ex` embed forward helpers with `mode/prefix_len/seq_lens`
   - sampling helpers
3. Add `include/vlm.h` with struct + declarations.
4. Add `src/vlm.c`:
   - constructor
   - parameter count
   - init weights
   - RoPE passthrough
   - image embeds
   - build embeds
   - forward
   - text logits
   - train step
   - generation no-cache
5. Add cached generation with prefix-LM prefill.
6. Add tests.
7. Update `include/dnn.h`.
8. Run full tests.
9. P1 follow-up: complete optimized padded train path (`seq_lens` attention + masked CE) before mixed-length batching.
10. P1 follow-up: add masked VLM train step before instruction tuning.

---

## Acceptance criteria

VLM support is complete when:

- `vision_lm_create()` builds patch conv + decoder LM and registers all params.
- `tensor_attention_ex(... ATTENTION_PREFIX_LM, prefix_len=I, ...)` supports bidirectional image-prefix attention.
- `transformer_block_forward_ex()` and GPT `_ex` helpers route VLM forward through prefix-LM attention.
- `vision_lm_forward()` returns `[B,I+T,V]` logits.
- `vision_lm_forward_text_logits()` returns `[B,T,V]` logits aligned to text positions.
- `vision_lm_train_step()` trains text tokens conditioned on image prefix.
- `vision_lm_generate()` supports greedy no-cache generation for `[1,C,H,W] + [1,T]`.
- cached generation uses prefix-LM prefill and matches no-cache under greedy sampling on tests.
- RoPE max sequence accounts for image tokens.
- `module_parameters()` sees all VLM params.
- `module_save/load` works for VLM-created checkpoints.
- tests pass for forward, backward, train, and generation.

P1 acceptance before flexible mixed-length batching:
- `tensor_cross_entropy_masked()` exists.
- `tensor_attention_ex()` implements non-NULL `seq_lens` path and skips pad rows/cols.
- `vision_lm_train_step_padded()` accepts `text_lens` + `loss_mask`.
- padded batch test matches exact-length unpadded losses on same samples.

P1 acceptance before instruction/chat tuning:
- `vision_lm_train_step_masked()` supports answer-only loss masking.
- masked train test verifies mask=0 positions produce no gradient contribution.
