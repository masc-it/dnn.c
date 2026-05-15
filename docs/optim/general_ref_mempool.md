# Explicit Pool Migration: `dnn_ctx`

Kill global pool state. Thread a `mem_pool *scratch` through every op that
allocates intermediates, and a `mem_pool *params` through every *create fn.
Bundle the three pools into a lightweight `dnn_ctx` convenience struct.

Total estimated churn: ~500 line changes across ~45 files.

---

## 1. New file: `include/context.h`

```c
#ifndef DNN_CONTEXT_H
#define DNN_CONTEXT_H

#include "pool.h"

typedef struct dnn_ctx {
    mem_pool *params;   // weights, optimizer state, model structs — never reset
    mem_pool *scratch;  // activations, grad_fn, intermediates — reset per step
    mem_pool *data;     // raw data, batch tensors — reset per batch/epoch
} dnn_ctx;

// Convenience init — allocates all 3 pools from separate budgets.
// Returns 0 on success, -1 if any malloc fails.
int dnn_ctx_init(dnn_ctx *ctx, size_t params_sz, size_t scratch_sz, size_t data_sz);

// Reset scratch + data (called each step/batch). Params unchanged.
void dnn_ctx_reset_step(dnn_ctx *ctx);

// Destroy all 3 pools.
void dnn_ctx_destroy(dnn_ctx *ctx);

#endif
```

---

## 2. Delete global state: `include/pool_int.h` + `src/pool_int.h`

Remove file `src/pool_int.h` entirely. Delete these three declarations:

```
mem_pool *_mem_pool_params(void);    // pool_int.h:11  — DELETE
mem_pool *_mem_pool_scratch(void);   // pool_int.h:12  — DELETE
mem_pool *_mem_pool_data(void);      // pool_int.h:13  — DELETE
```

Remove the `tensor_int.h` internal header and `_tensor_scratch_create`.

---

## 3. Rewrite `src/pool.c`

**Before** (`pool.c`):
- Global `static mem_pool *pool_params/scratch/data` — DELETE
- `mem_pool_set_defaults()` — DELETE
- `mem_params_alloc(bytes, src)`, `mem_scratch_alloc(bytes, src)`, `mem_data_alloc(bytes, src)` — DELETE
- `_mem_pool_params()`, `_mem_pool_scratch()`, `_mem_pool_data()` — DELETE

**After** (`pool.c`):
- Only `mem_pool_create`, `mem_pool_destroy`, `mem_pool_reset`, `mem_pool_mark`, `mem_pool_release`, `_mem_pool_alloc`, `_mem_pool_alloc_nz` survive.

`include/pool.h` changes:
- Delete: `mem_pool_set_defaults`, `mem_params_alloc`, `mem_scratch_alloc`, `mem_data_alloc`
- Keep: `mem_pool_create`, `mem_pool_destroy`, `mem_pool_reset`, `mem_pool_mark`, `mem_pool_release`

---

## 4. `dnn_ctx` implementation: new `src/context.c`

```c
#include "context.h"
#include "pool_int.h"    // for _mem_pool_alloc
#include <stdlib.h>
#include <assert.h>

int dnn_ctx_init(dnn_ctx *ctx, size_t params_sz, size_t scratch_sz, size_t data_sz) {
    mem_pool *p = malloc(sizeof(mem_pool));
    mem_pool *s = malloc(sizeof(mem_pool));
    mem_pool *d = malloc(sizeof(mem_pool));
    if (!p || !s || !d) { free(p); free(s); free(d); return -1; }
    *p = mem_pool_create(params_sz);
    *s = mem_pool_create(scratch_sz);
    *d = mem_pool_create(data_sz);
    if (!p->buffer || !s->buffer || !d->buffer) {
        mem_pool_destroy(p); free(p);
        mem_pool_destroy(s); free(s);
        mem_pool_destroy(d); free(d);
        return -1;
    }
    ctx->params  = p;
    ctx->scratch = s;
    ctx->data    = d;
    return 0;
}

void dnn_ctx_reset_step(dnn_ctx *ctx) {
    mem_pool_reset(ctx->scratch);
    mem_pool_reset(ctx->data);
}

void dnn_ctx_destroy(dnn_ctx *ctx) {
    if (ctx->params)  { mem_pool_destroy(ctx->params);  free(ctx->params);  }
    if (ctx->scratch) { mem_pool_destroy(ctx->scratch); free(ctx->scratch); }
    if (ctx->data)    { mem_pool_destroy(ctx->data);    free(ctx->data);    }
    ctx->params = ctx->scratch = ctx->data = NULL;
}
```

---

## 5. Rule of thumb for signature changes

| Call site pattern | New param | Used for |
|---|---|---|
| Every op: `tensor_add`, `tensor_mul`, `tensor_relu`, ... | `mem_pool *scratch` first arg | Allocate output tensor, grad_fn, saved tensors, temp offset arrays |
| Every *create: `linear_create`, `sgd_create`, `kv_cache_create`, ... | `mem_pool *params` first arg | Allocate the struct, weight/bias tensors, optimizer state |
| Every forward/layer: `transformer_block_forward`, `linear_forward` | `mem_pool *scratch` first arg | Intermediate activations |
| Tensor construction: `tensor_zeros`, `tensor_randn`, `tensor_uniform` | `mem_pool *pool` first arg | Pick params or data pool explicitly |
| `tensor_zeros_data` | `mem_pool *data` first arg (rename to `tensor_zeros_data`) | Raw data tensors |
| `_tensor_scratch_create` | Delete. Replace with `tensor_scratch(scratch, ndim, shape)` | Scratch intermediates |

---

## 6. Signature migration: all public headers

### `include/tensor.h`

```c
// Before                                    // After
tensor *tensor_zeros(int ndim, const int *shape, int requires_grad);
// →  tensor *tensor_zeros(mem_pool *pool, int ndim, const int *shape, int requires_grad);

tensor *tensor_zeros_data(int ndim, const int *shape);
// →  tensor *tensor_zeros_data(mem_pool *pool, int ndim, const int *shape);

tensor *tensor_randn(int ndim, const int *shape, int requires_grad);
// →  tensor *tensor_randn(mem_pool *pool, int ndim, const int *shape, int requires_grad);

tensor *tensor_uniform(int ndim, const int *shape, int requires_grad, float bound);
// →  tensor *tensor_uniform(mem_pool *pool, int ndim, const int *shape, int requires_grad, float bound);

// New convenience (requires_grad=0 implicit):
tensor *tensor_scratch(mem_pool *pool, int ndim, const int *shape);

// View ops (tensor_slice, tensor_transpose, etc.) allocate a tiny view struct
// on scratch.  They need a scratch pool too.
// Before                                    // After
tensor *tensor_slice(tensor *t, int dim, int start, int len);
// →  tensor *tensor_slice(mem_pool *scratch, tensor *t, int dim, int start, int len);

tensor *tensor_transpose(tensor *t, int d1, int d2);
// →  tensor *tensor_transpose(mem_pool *scratch, tensor *t, int d1, int d2);

tensor *tensor_reshape(tensor *t, int ndim, const int *shape);
// →  tensor *tensor_reshape(mem_pool *scratch, tensor *t, int ndim, const int *shape);

tensor *tensor_flatten(tensor *t);
// →  tensor *tensor_flatten(mem_pool *scratch, tensor *t);

tensor *tensor_contiguous(tensor *t);
// →  tensor *tensor_contiguous(mem_pool *scratch, tensor *t);

// tensor_root, tensor_print, accessors — no pool needed, no change
```

### `include/ops.h`

Every op gets `mem_pool *scratch` as first arg:

```c
// Before                                    // After
tensor *tensor_add(const tensor *a, const tensor *b);
// →  tensor *tensor_add(mem_pool *scratch, const tensor *a, const tensor *b);

tensor *tensor_sub(...);     // same pattern
tensor *tensor_mul(...);     // same
tensor *tensor_div(...);     // same
tensor *tensor_matmul(...);  // same
tensor *tensor_relu(...);    // same
tensor *tensor_sigmoid(...); // same
tensor *tensor_tanh(...);    // same
tensor *tensor_silu(...);    // same
tensor *tensor_swiglu(...);  // same
tensor *tensor_softmax(...); // same
tensor *tensor_causal_softmax(...); // same
tensor *tensor_attention(...);      // same
tensor *tensor_dropout(...);        // same
tensor *tensor_sum(...);     // same
tensor *tensor_mean(...);    // same
tensor *tensor_cross_entropy(...);  // same
tensor *tensor_embedding(...);      // same
tensor *tensor_pow(...);     // same
tensor *tensor_neg(...);     // same
tensor *tensor_triu(...);    // same
tensor *tensor_avg_pool2d(...);     // same
tensor *tensor_cat(...);     // same
```

### `include/autograd.h`

```c
// Before                                    // After
void dnn_backward(tensor *loss);
// →  void dnn_backward(mem_pool *scratch, tensor *loss);
//   (backward allocates topo sort arrays on scratch)
```

### `include/nn.h`

```c
// Before                                    // After
linear  *linear_create(int in_features, int out_features);
// →  linear  *linear_create(mem_pool *params, int in_features, int out_features);

tensor *linear_forward(linear *l, const tensor *x);
// →  tensor *linear_forward(mem_pool *scratch, linear *l, const tensor *x);

swiglu_ffn *swiglu_ffn_create(int d_model, int intermediate_size);
// →  swiglu_ffn *swiglu_ffn_create(mem_pool *params, int d_model, int intermediate_size);

tensor *swiglu_ffn_forward(swiglu_ffn *ffn, const tensor *x);
// →  tensor *swiglu_ffn_forward(mem_pool *scratch, swiglu_ffn *ffn, const tensor *x);

// linear_num_parameters, swiglu_ffn_num_parameters — no pool, no change
```

### `include/norm.h`

```c
tensor *tensor_layer_norm(const tensor *x, const tensor *weight,
                          const tensor *bias, float eps);
// →  tensor *tensor_layer_norm(mem_pool *scratch, const tensor *x,
//                               const tensor *weight, const tensor *bias, float eps);
```

### `include/conv.h`

```c
tensor *tensor_conv2d(tensor *input, tensor *weight, tensor *bias,
                       int stride, int pad);
// →  tensor *tensor_conv2d(mem_pool *scratch,
//                           tensor *input, tensor *weight, tensor *bias,
//                           int stride, int pad);
```

### `include/attention.h`

```c
tensor *tensor_attention(tensor *Q, tensor *K, tensor *V, tensor *mask);
// →  tensor *tensor_attention(mem_pool *scratch,
//                              tensor *Q, tensor *K, tensor *V, tensor *mask);
```

### `include/multihead.h`

```c
tensor *tensor_split_heads(tensor *t, int H);
// →  tensor *tensor_split_heads(mem_pool *scratch, tensor *t, int H);

tensor *tensor_merge_heads(tensor *t);
// →  tensor *tensor_merge_heads(mem_pool *scratch, tensor *t);
```

### `include/rope.h`

```c
tensor *tensor_rope(tensor *x, const tensor *freqs_cos, const tensor *freqs_sin);
// →  tensor *tensor_rope(mem_pool *scratch,
//                         tensor *x, const tensor *freqs_cos, const tensor *freqs_sin);
//   (rope allocates a lightweight view tensor on scratch)

void tensor_rope_freqs_init(tensor **freqs_cos, tensor **freqs_sin,
                             int max_seq_len, int d_k, float base);
// →  void tensor_rope_freqs_init(mem_pool *params,
//                                 tensor **freqs_cos, tensor **freqs_sin,
//                                 int max_seq_len, int d_k, float base);
```

### `include/optim.h`

```c
sgd_opt    *sgd_create(tensor **params, int n_params, float lr, float momentum);
// →  sgd_opt *sgd_create(mem_pool *params_pool,
//                         tensor **params, int n_params, float lr, float momentum);

adamw_opt  *adamw_create(tensor **params, int n_params, float lr,
                          float beta1, float beta2, float eps, float weight_decay);
// →  adamw_opt *adamw_create(mem_pool *params_pool,
//                             tensor **params, int n_params, float lr,
//                             float beta1, float beta2, float eps, float weight_decay);

lr_scheduler *lr_scheduler_create(adamw_opt *opt, int schedule, ...);
// →  lr_scheduler *lr_scheduler_create(mem_pool *params_pool,
//                                       adamw_opt *opt, int schedule, ...);

// sgd_step, adamw_step, clip_grad_norm, clip_grad_value — no pool needed (no allocs)
// sgd_zero_grad, adamw_zero_grad — no pool needed
// sgd_free, adamw_free — empty (pool owns memory), keep as-is or delete
// lr_scheduler_step, lr_scheduler_get_lr, lr_scheduler_reset, lr_scheduler_destroy — no pool
```

### `include/transformer.h`

```c
kv_cache *kv_cache_create(int B, int H, int max_seq, int d_k);
// →  kv_cache *kv_cache_create(mem_pool *params_pool,
//                               int B, int H, int max_seq, int d_k);

// kv_cache_append, kv_cache_get_K, kv_cache_get_V — no pool (uses internal buffers)

transformer_block *transformer_block_create(int d_model, int n_heads, int d_k,
                                             int intermediate_size);
// →  transformer_block *transformer_block_create(mem_pool *params_pool,
//                                                 int d_model, int n_heads, int d_k,
//                                                 int intermediate_size);

tensor *transformer_block_forward(transformer_block *block, const tensor *x);
// →  tensor *transformer_block_forward(mem_pool *scratch,
//                                       transformer_block *block, const tensor *x);

decoder_lm *decoder_lm_create(int vocab_size, int d_model,
                               int n_layers, int n_heads, int d_k,
                               int intermediate_size);
// →  decoder_lm *decoder_lm_create(mem_pool *params_pool,
//                                   int vocab_size, int d_model, ...);

tensor *decoder_lm_forward(decoder_lm *lm, const tensor *input_ids);
// →  tensor *decoder_lm_forward(mem_pool *scratch,
//                                decoder_lm *lm, const tensor *input_ids);

tensor *decoder_lm_train_step(decoder_lm *lm, const tensor *input_ids,
                               adamw_opt *opt, float grad_clip,
                               float *grad_norm_out);
// →  tensor *decoder_lm_train_step(mem_pool *scratch_pool,
//                                   mem_pool *data_pool,
//                                   decoder_lm *lm, const tensor *input_ids,
//                                   adamw_opt *opt, float grad_clip,
//                                   float *grad_norm_out);
//   (allocates target tensor in data pool, intermediates in scratch)

tensor *transformer_block_forward_cached(transformer_block *block,
                                          const tensor *x, kv_cache *cache);
// →  tensor *transformer_block_forward_cached(mem_pool *scratch,
//                                              transformer_block *block,
//                                              const tensor *x, kv_cache *cache);

int *decoder_lm_generate(decoder_lm *lm, const tensor *prompt_ids,
                          int max_new_tokens, float temperature,
                          int use_cache, int *n_out);
// →  int *decoder_lm_generate(mem_pool *scratch_pool,
//                              mem_pool *data_pool,
//                              decoder_lm *lm, const tensor *prompt_ids,
//                              int max_new_tokens, float temperature,
//                              int use_cache, int *n_out);
//   (allocates output IDs in data pool, temp tensors in scratch)

void decoder_lm_enable_rope(decoder_lm *lm, int max_seq_len, float base);
// →  void decoder_lm_enable_rope(mem_pool *params_pool,
//                                 decoder_lm *lm, int max_seq_len, float base);

void decoder_lm_init_weights(decoder_lm *lm);
//   no allocs, no pool needed

// *_num_parameters — no pool, no change
```

### `include/tokenizer.h`

```c
int *tokenizer_encode(tokenizer *tok, const char *text, int *len);
// →  int *tokenizer_encode(mem_pool *data_pool, tokenizer *tok, const char *text, int *len);

tensor *tokenizer_text_to_tensor(tokenizer *tok, const char *text);
// →  tensor *tokenizer_text_to_tensor(mem_pool *data_pool, tokenizer *tok, const char *text);
```

### `include/mnist.h`

```c
mnist_model       *mnist_model_create(void);
// →  mnist_model *mnist_model_create(mem_pool *params_pool);

mnist_model_cnn   *mnist_model_create_cnn(void);
// →  mnist_model_cnn *mnist_model_create_cnn(mem_pool *params_pool);

mnist_model_cnn_pool *mnist_model_create_cnn_pool(void);
// →  mnist_model_cnn_pool *mnist_model_create_cnn_pool(mem_pool *params_pool);

tensor *mnist_model_forward(mnist_model *m, const tensor *x);
// →  tensor *mnist_model_forward(mem_pool *scratch, mnist_model *m, const tensor *x);

tensor *mnist_model_forward_cnn(mnist_model_cnn *m, const tensor *x);
// →  tensor *mnist_model_forward_cnn(mem_pool *scratch, mnist_model_cnn *m, const tensor *x);

tensor *mnist_model_forward_cnn_pool(mnist_model_cnn_pool *m, const tensor *x);
// →  tensor *mnist_model_forward_cnn_pool(mem_pool *scratch, mnist_model_cnn_pool *m, const tensor *x);

void mnist_train(mnist_model *m, tensor *train_images, tensor *train_labels,
                 int epochs, int batch_size, float lr, int val_n, int patience);
// →  void mnist_train(dnn_ctx *ctx, mnist_model *m,
//                     tensor *train_images, tensor *train_labels,
//                     int epochs, int batch_size, float lr, int val_n, int patience);

void mnist_train_cnn(mnist_model_cnn *m, ...);
// →  void mnist_train_cnn(dnn_ctx *ctx, mnist_model_cnn *m, ...);

void mnist_train_cnn_pool(mnist_model_cnn_pool *m, ...);
// →  void mnist_train_cnn_pool(dnn_ctx *ctx, mnist_model_cnn_pool *m, ...);

float mnist_eval_generic(void *model, ...);
//   no allocs (runs in no-grad), but needs scratch for forward:
// →  float mnist_eval_generic(mem_pool *scratch, void *model, ...);

// Inline wrappers mnist_eval, mnist_eval_cnn, mnist_eval_cnn_pool
// →  pass scratch through
```

---

## 7. Source file internal call sites

### `src/tensor.c`

| Line | Before | After |
|---|---|---|
| 42 | `_tensor_scratch_create(t->ndim, t->shape, t->requires_grad)` | `tensor_scratch(scratch, t->ndim, t->shape)` |
| 74-76 | `_tensor_scratch_create`: body uses `_mem_pool_scratch()` | Delete fn entirely. Callers use `tensor_scratch(scratch, ...)` |
| 89 | `tensor_zeros`: body calls `_mem_pool_params()` | `tensor_zeros(mem_pool *pool, ndim, shape, requires_grad)` — use pool param |
| 93 | `tensor_zeros_data`: body calls `_mem_pool_data()` | `tensor_zeros_data(mem_pool *pool, ndim, shape)` — use pool param |
| 97 | `tensor_randn`: body calls `_mem_pool_params()` | `tensor_randn(mem_pool *pool, ndim, shape, requires_grad)` — use pool param |
| 113 | `tensor_uniform`: body calls `_mem_pool_params()` | `tensor_uniform(mem_pool *pool, ndim, shape, requires_grad, bound)` — use pool param |
| 229 | `mem_scratch_alloc(sizeof(tensor), t)` | `_mem_pool_alloc(scratch, sizeof(tensor), t)` |
| 243 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL)` |
| 246-248 | `mem_scratch_alloc(...)` ×3 | `_mem_pool_alloc(scratch, ...)` ×3 |
| 262 | `mem_scratch_alloc(sizeof(tensor), t)` | `_mem_pool_alloc(scratch, sizeof(tensor), t)` |
| 301 | `mem_scratch_alloc(sizeof(tensor), t)` | `_mem_pool_alloc(scratch, sizeof(tensor), t)` |
| 317 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL)` |
| 366 | `mem_params_alloc(...)` | `_mem_pool_alloc(params, ...)` |

### `src/autograd.c`

| Line | Before | After |
|---|---|---|
| 29 | `mem_scratch_alloc(sizeof(grad_fn), NULL)` | `_mem_pool_alloc(scratch, sizeof(grad_fn), NULL)` |
| 41 | `mem_params_alloc(n * sizeof(float), NULL)` | `_mem_pool_alloc(params, n * sizeof(float), NULL)` |
| 43 | `mem_scratch_alloc(n * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, n * sizeof(float), NULL)` |
| 106 | `mem_scratch_alloc(256 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 256 * sizeof(tensor*), NULL)` |
| 111 | `mem_scratch_alloc(n_nodes * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, n_nodes * sizeof(tensor*), NULL)` |
| 114 | `mem_scratch_alloc(256 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 256 * sizeof(tensor*), NULL)` |
| 122 | `mem_params_alloc(tensor_numel(loss) * sizeof(float), NULL)` | `_mem_pool_alloc(params, ...)` |
| 124 | `mem_scratch_alloc(tensor_numel(loss) * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, ...)` |

### `src/ops_elem.c`

Every `_tensor_scratch_create` → `tensor_scratch(scratch, ...)`.
Every `mem_scratch_alloc` → `_mem_pool_alloc(scratch, ...)`.

| Pattern | Count | Lines |
|---|---|---|
| `_tensor_scratch_create(ndim, shape, 0)` | 7 | 73, 183, 289, 401, 470, 532, 788 |
| `mem_scratch_alloc(n * sizeof(int), NULL)` | 8 | 40,41, 150,151, 256,257, 368,369 |
| `mem_scratch_alloc(n * sizeof(tensor*), NULL)` (grad_fn) | 8 | 107, 214, 320, 432, 496, 558, 800 |
| `mem_scratch_alloc(sizeof(float), NULL)` (saved) | 2 | 500, 805 |
| `mem_scratch_alloc(sizeof(int), NULL)` (saved) | 1 | 500, 805 |
| `_tensor_scratch_create(2, (int[]){N, N}, 0)` (triu) | 1 | 586 |

### `src/ops_activation.c`

| Pattern | Count | Lines |
|---|---|---|
| `_tensor_scratch_create` | 8 | 40, 97, 170, 291, 505, 697, 1047, 1120 |
| `mem_scratch_alloc` (grad_fn/saved) | ~20 | 61, 117, 120, 191, 239, 240, 324, 400, 401, 446, 447, 562, 565, 567, 837, 840, 934, 935, 987, 1055, 1058, 1062, 1065, 1126, 1151, 1155, 1158, 1162 |

### `src/ops_matrix.c`

| Line | Before | After |
|---|---|---|
| 323 | `_tensor_scratch_create(2, (int[]){M, N}, 0)` | `tensor_scratch(scratch, 2, (int[]){M, N})` |
| 365 | `mem_scratch_alloc(2 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL)` |
| 391 | `_tensor_scratch_create(out_ndim, out_shape, 0)` | `tensor_scratch(scratch, out_ndim, out_shape)` |
| 453 | `mem_scratch_alloc(2 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL)` |

### `src/ops_reduce.c`

| Line | Before | After |
|---|---|---|
| 69 | `_tensor_scratch_create(ndim, shape_out, 0)` | `tensor_scratch(scratch, ndim, shape_out)` |
| 125 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL)` |
| 128 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL)` |
| 129 | `mem_scratch_alloc(sizeof(int), NULL)` | `_mem_pool_alloc(scratch, sizeof(int), NULL)` |
| 146 | `_tensor_scratch_create(1, (int[]){1}, 0)` | `tensor_scratch(scratch, 1, (int[]){1})` |

### `src/ops_pool.c`

| Line | Before | After |
|---|---|---|
| 263 | `_tensor_scratch_create(4, (int[]){N, C, H_out, W_out}, 0)` | `tensor_scratch(scratch, 4, ...)` |
| 365 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 369-373 | `mem_scratch_alloc(...)` ×3 | `_mem_pool_alloc(scratch, ...)` ×3 |

### `src/attention.c`

| Line | Before | After |
|---|---|---|
| 70 | `mem_scratch_alloc((size_t)N * N * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 71 | `mem_scratch_alloc((size_t)N * N * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 238 | `_tensor_scratch_create(4, (int[]){B, H, N, d}, 0)` | `tensor_scratch(scratch, 4, ...)` |
| 242 | `_tensor_scratch_create(4, (int[]){B, H, N, N}, 0)` | `tensor_scratch(scratch, 4, ...)` |
| 250 | `mem_scratch_alloc((size_t)N * N * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 388 | `mem_scratch_alloc((mask ? 4 : 3) * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 395-402 | `mem_scratch_alloc(...)` ×4 | `_mem_pool_alloc(scratch, ...)` ×4 |

### `src/conv.c`

| Line | Before | After |
|---|---|---|
| 402 | `mem_pool_mark(_mem_pool_scratch())` | `mem_pool_mark(scratch)` |
| 403 | `mem_scratch_alloc((size_t)K * M * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 434 | `mem_pool_release(_mem_pool_scratch(), _dcm)` | `mem_pool_release(scratch, _dcm)` |
| 493 | `mem_scratch_alloc(...)` | `_mem_pool_alloc(scratch, ...)` |
| 628 | `_tensor_scratch_create(4, ...)` | `tensor_scratch(scratch, 4, ...)` |
| 646 | `mem_scratch_alloc(...)` | `_mem_pool_alloc(scratch, ...)` |
| 656 | `mem_scratch_alloc(...)` | `_mem_pool_alloc(scratch, ...)` |
| 722 | `mem_scratch_alloc(2 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 727-730 | `mem_scratch_alloc(...)` ×3 | `_mem_pool_alloc(scratch, ...)` ×3 |
| 746 | `mem_pool_mark(_mem_pool_scratch())` | `mem_pool_mark(scratch)` |
| 747 | `mem_scratch_alloc(...)` | `_mem_pool_alloc(scratch, ...)` |
| 802 | `mem_scratch_alloc(2 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 807-810 | `mem_scratch_alloc(...)` ×3 | `_mem_pool_alloc(scratch, ...)` ×3 |
| 828 | `mem_pool_release(_mem_pool_scratch(), _fcm)` | `mem_pool_release(scratch, _fcm)` |

### `src/embedding.c`

| Line | Before | After |
|---|---|---|
| 76 | `_tensor_scratch_create(2, (int[]){N, d_model}, 0)` | `tensor_scratch(scratch, 2, ...)` |
| 101 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 104 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |

### `src/multihead.c`

| Line | Before | After |
|---|---|---|
| 120 | `_tensor_scratch_create(4, (int[]){B, H, N, d_k}, 0)` | `tensor_scratch(scratch, 4, ...)` |
| 133 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 137-138 | `mem_scratch_alloc(...)` ×2 | `_mem_pool_alloc(scratch, ...)` ×2 |
| 202 | `_tensor_scratch_create(3, (int[]){B, N, H * d_k}, 0)` | `tensor_scratch(scratch, 3, ...)` |
| 215 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |

### `src/nn.c`

| Line | Before | After |
|---|---|---|
| 12 | `mem_params_alloc(sizeof(linear), NULL)` | `_mem_pool_alloc(params, sizeof(linear), NULL)` |
| 16 | `tensor_uniform(2, (int[]){...}, 1, bound)` | `tensor_uniform(params, 2, (int[]){...}, 1, bound)` |
| 38 | `mem_params_alloc(sizeof(swiglu_ffn), NULL)` | `_mem_pool_alloc(params, sizeof(swiglu_ffn), NULL)` |

### `src/norm.c`

| Line | Before | After |
|---|---|---|
| 191 | `mem_scratch_alloc(n * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, n * sizeof(float), NULL)` |
| 192 | `mem_scratch_alloc(n * sizeof(float), NULL)` | `_mem_pool_alloc(scratch, n * sizeof(float), NULL)` |
| 221 | `_tensor_scratch_create(ndim, x->shape, 0)` | `tensor_scratch(scratch, ndim, x->shape)` |
| 263 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 267 | `mem_scratch_alloc(4 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |

### `src/optim.c`

| Line | Before | After |
|---|---|---|
| 9 | `mem_params_alloc(sizeof(sgd_opt), NULL)` | `_mem_pool_alloc(params, sizeof(sgd_opt), NULL)` |
| 16 | `mem_params_alloc(n_params * sizeof(tensor*), NULL)` | `_mem_pool_alloc(params, ...)` |
| 19 | `tensor_zeros(p->ndim, p->shape, 0)` | `tensor_zeros(params, p->ndim, p->shape, 0)` |
| 70 | `mem_params_alloc(sizeof(adamw_opt), NULL)` | `_mem_pool_alloc(params, ...)` |
| 79-80 | `mem_params_alloc(...)` ×2 | `_mem_pool_alloc(params, ...)` ×2 |
| 84-85 | `tensor_zeros(p->ndim, p->shape, 0)` ×2 | `tensor_zeros(params, p->ndim, p->shape, 0)` ×2 |
| 154 | `mem_params_alloc(sizeof(lr_scheduler), NULL)` | `_mem_pool_alloc(params, ...)` |

### `src/rope.c`

| Line | Before | After |
|---|---|---|
| 120 | `mem_scratch_alloc(sizeof(tensor), x)` | `_mem_pool_alloc(scratch, sizeof(tensor), x)` |
| 133 | `mem_scratch_alloc(1 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 136 | `mem_scratch_alloc(2 * sizeof(tensor*), NULL)` | `_mem_pool_alloc(scratch, ...)` |
| 157 | `tensor_zeros(2, shape, 0)` | `tensor_zeros(params, 2, shape, 0)` |
| 158 | `tensor_zeros(2, shape, 0)` | `tensor_zeros(params, 2, shape, 0)` |

### `src/tokenizer.c`

| Line | Before | After |
|---|---|---|
| 103 | `mem_data_alloc((size_t)max_ids * sizeof(int), NULL)` | `_mem_pool_alloc(data, ...)` |
| 165-168 | `tensor_zeros_data(2, shape)` | `tensor_zeros_data(data, 2, shape)` |

### `src/transformer.c`

| Line | Before | After |
|---|---|---|
| 32 | `mem_params_alloc(sizeof(kv_cache), NULL)` | `_mem_pool_alloc(params, ...)` |
| 34-35 | `tensor_zeros(4, shape, 0)` ×2 | `tensor_zeros(params, 4, shape, 0)` ×2 |
| 122 | `mem_params_alloc(sizeof(transformer_block), NULL)` | `_mem_pool_alloc(params, ...)` |
| 133-139 | `tensor_zeros(1, (int[]){d_model}, 1)` ×4 | `tensor_zeros(params, 1, (int[]){d_model}, 1)` ×4 |
| 214 | `mem_params_alloc(sizeof(decoder_lm), NULL)` | `_mem_pool_alloc(params, ...)` |
| 221 | `tensor_uniform(2, (int[]){...}, 1, bound)` | `tensor_uniform(params, 2, ..., 1, bound)` |
| 224 | `mem_params_alloc(n_layers * sizeof(...), NULL)` | `_mem_pool_alloc(params, ...)` |
| 231-232 | `tensor_zeros(1, (int[]){d_model}, 1)` ×2 | `tensor_zeros(params, 1, (int[]){d_model}, 1)` ×2 |
| 237 | `mem_params_alloc(sizeof(linear), NULL)` | `_mem_pool_alloc(params, ...)` |
| 243 | `mem_params_alloc(sizeof(tensor), tmp)` | `_mem_pool_alloc(params, ...)` |
| 246 | `tensor_zeros(1, (int[]){vocab_size}, 1)` | `tensor_zeros(params, 1, (int[]){vocab_size}, 1)` |
| 303 | `tensor_zeros_data(2, (int[]){B, N - 1})` | `tensor_zeros_data(data, 2, ...)` |
| 394 | `_tensor_scratch_create(4, (int[]){B, H, N_new, d_k}, 0)` | `tensor_scratch(scratch, 4, ...)` |
| 406 | `mem_scratch_alloc(...)` | `_mem_pool_alloc(scratch, ...)` |
| 611 | `mem_data_alloc((size_t)max_len * sizeof(int), NULL)` | `_mem_pool_alloc(data, ...)` |
| 623 | `mem_pool_mark(_mem_pool_params())` | `mem_pool_mark(params)` |
| 628 | `mem_params_alloc(...)` | `_mem_pool_alloc(params, ...)` |
| 635 | `tensor_zeros_data(1, (int[]){1})` | `tensor_zeros_data(data, 1, ...)` |
| 657 | `mem_data_alloc(...)` | `_mem_pool_alloc(data, ...)` |
| 659 | `mem_pool_reset(_mem_pool_scratch())` | `mem_pool_reset(scratch)` |
| 673 | `mem_pool_reset(_mem_pool_scratch())` | `mem_pool_reset(scratch)` |
| 693 | `mem_data_alloc(...)` | `_mem_pool_alloc(data, ...)` |
| 695 | `mem_pool_reset(_mem_pool_scratch())` | `mem_pool_reset(scratch)` |
| 709 | `mem_pool_release(_mem_pool_params(), cache_mark)` | `mem_pool_release(params, cache_mark)` |
| 716 | `tensor_zeros_data(2, (int[]){1, cur_len})` | `tensor_zeros_data(data, 2, ...)` |
| 725 | `mem_data_alloc(...)` | `_mem_pool_alloc(data, ...)` |
| 729 | `mem_pool_reset(_mem_pool_scratch())` | `mem_pool_reset(scratch)` |

### `src/mnist.c`

| Line | Before | After |
|---|---|---|
| 114 | `tensor_zeros_data(2, (int[]){N, img_size})` | `tensor_zeros_data(data, 2, ...)` |
| 137 | `tensor_zeros_data(1, (int[]){N})` | `tensor_zeros_data(data, 1, ...)` |
| 150 | `mem_params_alloc(sizeof(mnist_model), NULL)` | `_mem_pool_alloc(params, ...)` |
| 171 | `mem_params_alloc(sizeof(mnist_model_cnn), NULL)` | `_mem_pool_alloc(params, ...)` |
| 175-186 | `tensor_uniform(...)`, `tensor_zeros(...)` ×6 | `tensor_uniform(params, ..., requires_grad, bound)`, `tensor_zeros(params, ..., requires_grad)` |
| 220-232 | (same pattern for cnn_pool) | same |
| 324-325 | `_tensor_scratch_create(2, ...)` ×2 | `tensor_scratch(scratch, ...)` ×2 |
| 353 | `mem_pool_reset(_mem_pool_scratch())` | `mem_pool_reset(scratch)` |
| 378-379 | `_tensor_scratch_create(2, ...)` ×2 | `tensor_scratch(scratch, ...)` ×2 |
| 402 | `mem_pool_reset(_mem_pool_scratch())` | `mem_pool_reset(scratch)` |
| 535 | `mem_pool_reset(_mem_pool_scratch())` | `mem_pool_reset(scratch)` |

---

## 8. Entry point files

### `main.c` (current pattern)

```c
mem_pool params  = mem_pool_create(10 * 1024 * 1024);
mem_pool scratch = mem_pool_create(192 * 1024 * 1024);
mem_pool data    = mem_pool_create(210 * 1024 * 1024);
mem_pool_set_defaults(&params, &scratch, &data);
```

**After:**

```c
dnn_ctx ctx;
dnn_ctx_init(&ctx, 10*1024*1024, 192*1024*1024, 210*1024*1024);

// pass ctx.params to model_create, ctx.scratch to forwards, ctx.data to loaders

mnist_model_cnn *m_cnn = mnist_model_create_cnn(ctx.params);
// internally: tensor_uniform(ctx.params, ..., 1, bound)

// eval:
float acc = mnist_eval_cnn(ctx.scratch, m_cnn, test_images, test_labels);

// cleanup:
dnn_ctx_destroy(&ctx);
```

### `main_lm.c`

```c
// Before:
mem_pool params  = mem_pool_create(64 * 1024 * 1024);
mem_pool scratch = mem_pool_create(512 * 1024 * 1024);
mem_pool data    = mem_pool_create(10 * 1024 * 1024);
mem_pool_set_defaults(&params, &scratch, &data);

// After:
dnn_ctx ctx;
dnn_ctx_init(&ctx, 64*1024*1024, 512*1024*1024, 10*1024*1024);
```

### `main_prep_data.c`

```c
// same pattern: dnn_ctx_init, use ctx.data for data loading
```

---

## 9. Test files

Every test currently does:

```c
mem_pool params  = mem_pool_create(SZ);
mem_pool scratch = mem_pool_create(SZ);
mem_pool data    = mem_pool_create(SZ);
mem_pool_set_defaults(&params, &scratch, &data);
// ... run tests ...
mem_pool_destroy(&params);
mem_pool_destroy(&scratch);
mem_pool_destroy(&data);
```

**After** (every test file — ~40 instances across test/*.c):

```c
dnn_ctx ctx;
dnn_ctx_init(&ctx, PARAMS_SZ, SCRATCH_SZ, DATA_SZ);
// ... run tests, passing ctx.params / ctx.scratch / ctx.data ...
dnn_ctx_destroy(&ctx);
```

Test files that currently use `_mem_pool_scratch()` / `_mem_pool_params()` directly
to call `mem_pool_reset`:

| File | Line(s) | Replace with |
|---|---|---|
| `test/test_cnn_stress.c` | 94, 132, 154, 169-170, 254, 267-268, 296, 312 | `mem_pool_reset(ctx.scratch)` / `mem_pool_reset(ctx.params)` |
| `test/test_rope.c` | 185 | `_mem_pool_alloc(ctx.scratch, ...)` |
| `test/test_decoder_lm_training.c` | 57 | `_mem_pool_alloc(ctx.params, ...)` |
| `test/test_generation_prefix.c` | 71 | `_mem_pool_alloc(ctx.params, ...)` |
| `test/test_generation.c` | 72 | `_mem_pool_alloc(ctx.params, ...)` |
| `test/test_lr_scheduler.c` | 24 | `_mem_pool_alloc(ctx.params, ...)` |
| `test/test_pool.c` | 9, 18, 23 | `_mem_pool_alloc(...)` with explicit pool |

---

## 10. Bench files

| File | Line(s) | Change |
|---|---|---|
| `bench/bench_ops.c` | 469, 486 | `mem_pool_reset(_mem_pool_scratch())` → `mem_pool_reset(scratch)` |
| `bench/bench_transformer.c` | 99 | `mem_params_alloc(...)` → `_mem_pool_alloc(params, ...)` |

All `tensor_randn` / `tensor_zeros` calls in benches need pool pointer.

---

## 11. `dnn.h` umbrella header

`include/dnn.h` currently includes all public headers. Add `context.h`:

```c
#ifndef DNN_H
#define DNN_H

#include "context.h"    // NEW
#include "pool.h"
#include "tensor.h"
#include "ops.h"
#include "autograd.h"
#include "nn.h"
#include "norm.h"
#include "conv.h"
#include "optim.h"
#include "attention.h"
#include "multihead.h"
#include "rope.h"
#include "tokenizer.h"
#include "transformer.h"

#endif
```

---

## 12. Migration order (safe sequence)

1. Add `include/context.h` + `src/context.c` (no-op until anything uses it)
2. Add `tensor_scratch(pool, ndim, shape)` to `tensor.h`/`tensor.c` alongside old fns
3. One-by-one, migrate each source file's internal allocs. Do `src/ops_elem.c` first as proof of concept — it's the simplest mechanical change.
4. Once all `src/*.c` are migrated, delete the globals from `pool.c`, delete `pool_int.h`.
5. Migrate all test files.
6. Migrate benches.
7. Delete `_tensor_scratch_create` and `tensor_int.h`.
8. Delete `mem_pool_set_defaults`, `mem_params_alloc`, `mem_scratch_alloc`, `mem_data_alloc` from `pool.h`.
9. Final: delete `dnn_grad_enabled` → thread-local no-grad flag? (separate concern, not part of this migration)
