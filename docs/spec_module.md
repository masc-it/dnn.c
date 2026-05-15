# Module Abstraction — Design Proposal

## Problem

Every model/layer struct collects params manually in many places:

- **Training call sites**: build `tensor *params_arr[] = { fc1->weight, fc1->bias, ... }` and pass `n_params`
- **`*_num_parameters()`**: walk same fields again, summing tensor sizes
- **`adamw_zero_grad()` / `sgd_zero_grad()`**: iterate optimizer's manual array
- **Save/load**: not implemented yet, but would need same list plus stable names

Fragile: new param means updating N spots. Blocks generic tooling.

## Goal

Each layer/model declares params and child modules once, in constructor. Everything else derives from that.

Similar to PyTorch `nn.Module`: define a module, compose submodules and parameters, and `.parameters()` works. In C we cannot intercept `__setattr__`, so registration is explicit.

Constraints:

- params pool owns module structs, registration entries, flat caches
- no pre-sized capacity guess
- no realloc
- child names exist now so state-dict paths can exist later
- tied/shared params do not double-update optimizer state

## Design

### Module header

Every module struct embeds this as its **first field**. Offset 0 makes cast between `module*` and concrete type safe.

Use one ordered registration list. Each entry is either a direct param or a child module. This preserves constructor registration order and keeps state-dict order deterministic.

```c
typedef enum module_item_kind {
    MODULE_ITEM_PARAM = 1,
    MODULE_ITEM_CHILD = 2,
} module_item_kind;

/* One registration entry — small alloc from params pool. */
typedef struct module_item {
    struct module_item *next;
    module_item_kind    kind;
    const char         *name;   /* required, stable pointer: "weight", "fc1", "blocks.0" */
    union {
        tensor        *param;
        struct module *child;
    } as;
} module_item;

typedef struct module {
    struct mem_pool *pool;       /* params pool; owns entries + flat cache */
    const char      *type_name;  /* e.g. "linear", "mnist_cnn" */
    struct module   *parent;     /* set by module_add_child, NULL for root */

    module_item     *items_head;
    module_item     *items_tail;
    int              n_items;
    int              n_direct_params;
    int              n_children;

    /* Lazy cache: unique recursive params, built once by module_parameters(). */
    tensor          **flat_params;
    int              n_flat;
} module;
```

### Registration helpers

```c
/* Initialize header. params pool owns the containing module struct. */
void module_init(module *m, struct mem_pool *params, const char *type_name);

/* Register direct parameter tensor. t must be stable in params pool. */
void module_param(module *m, const char *name, tensor *t);

/* Register child submodule. Sets child->parent = m. */
void module_add_child(module *m, const char *name, module *child);
```

Rules:

- `name` required and must outlive module (string literal or pool-owned generated string).
- `module_add_child()` asserts `child->parent == NULL`.
- Registration after introspection is invalid. Helpers assert current module and ancestors have no `flat_params` cache.
- Module tree is immutable after construction / first `module_parameters()`.

`module_init()` stores pool so `module_parameters()` can allocate its cache without extra args.

### Introspection functions

```c
/* Unique recursive params, depth-first in registration order.
   Builds flat cache once in root->pool; later calls are O(1). */
tensor **module_parameters(module *m, int *n_out);

/* Total scalar elements across unique recursive params. */
long long module_num_parameters(module *m);

/* Zero grads for all unique recursive params. */
void module_zero_grad(module *m);
```

`module_parameters()` deduplicates by exact `tensor*` identity. This protects true tied weights from double optimizer state/update.

If two different `tensor` structs alias same storage, dedup will **not** catch that. Prefer true same `tensor*` for tied params, or add explicit alias handling later.

### Named paths for future state_dict

State-dict naming comes from child edge names + param names:

- root direct param: `embedding_table`
- child param: `fc1.weight`
- indexed child: `blocks.0.q_proj.weight`

No need to materialize this in first implementation, but registration API must capture child names now. Later save/load can stream traversal and build paths on scratch/data pool.

## Examples

### Linear

```c
typedef struct linear {
    module  base;             /* first field */
    tensor *weight;
    tensor *bias;
    int     in_features;
    int     out_features;
} linear;

linear *linear_create(struct mem_pool *params, int in_features, int out_features) {
    linear *l = _mem_pool_alloc(params, sizeof(linear), NULL);
    module_init(&l->base, params, "linear");

    l->in_features  = in_features;
    l->out_features = out_features;

    float bound = 1.0f / sqrtf((float)in_features);
    l->weight = tensor_uniform(params, 2, (int[]){in_features, out_features}, 1, bound);
    l->bias   = tensor_uniform(params, 1, (int[]){out_features}, 1, bound);

    module_param(&l->base, "weight", l->weight);
    module_param(&l->base, "bias",   l->bias);
    return l;
}
```

### Composite with direct params + children

```c
/* current mnist_model_cnn — conv params direct, linear layers as children */
typedef struct mnist_model_cnn {
    module  base;                 /* first field */
    tensor *conv1_w, *conv1_b;
    tensor *conv2_w, *conv2_b;
    tensor *conv3_w, *conv3_b;
    linear *fc1;
    linear *fc2;
} mnist_model_cnn;

mnist_model_cnn *mnist_model_create_cnn(struct mem_pool *params) {
    mnist_model_cnn *m = _mem_pool_alloc(params, sizeof(mnist_model_cnn), NULL);
    module_init(&m->base, params, "mnist_cnn");

    m->conv1_w = tensor_...;
    m->conv1_b = tensor_...;
    module_param(&m->base, "conv1_w", m->conv1_w);
    module_param(&m->base, "conv1_b", m->conv1_b);
    /* conv2/conv3 same pattern */

    m->fc1 = linear_create(params, 3136, 128);
    module_add_child(&m->base, "fc1", &m->fc1->base);

    m->fc2 = linear_create(params, 128, 10);
    module_add_child(&m->base, "fc2", &m->fc2->base);

    return m;
}
```

### Training call site

Before:

```c
tensor *params_arr[] = { m->fc1->weight, m->fc1->bias, m->fc2->weight, m->fc2->bias };
mnist_train_impl(..., params_arr, 4, ..., m, mnist_model_forward_cnn);
```

After:

```c
int n_params;
tensor **params = module_parameters(&m->base, &n_params);
mnist_train_impl(..., params, n_params, ..., m, mnist_model_forward_cnn);
```

Optimizer creation becomes:

```c
int n_params;
tensor **params = module_parameters(&lm->base, &n_params);
adamw_opt *opt = adamw_create(ctx.params, params, n_params, LR,
                              0.9f, 0.999f, 1e-8f, 0.01f);
```

## Implementation notes

### Lazy flat array

`module_parameters()` can build cache in two passes:

1. Count max recursive entries including duplicates.
2. Allocate `max_count * sizeof(tensor*)` from `m->pool`.
3. Walk again, append only if `tensor*` not already present.
4. Set `m->flat_params` and `m->n_flat` to unique count.

Wasting a few pointer slots for duplicates is fine in params pool. Avoids realloc.

Pseudo-code:

```c
walk(m, array, &idx):
    for each item in m->items:
        if item.kind == MODULE_ITEM_PARAM:
            if (!contains(array, idx, item.as.param))
                array[idx++] = item.as.param;
        else:
            walk(item.as.child, array, &idx);
```

### Parameter count

```c
long long module_num_parameters(module *m) {
    int n;
    tensor **params = module_parameters(m, &n);
    long long total = 0;
    for (int i = 0; i < n; i++)
        total += tensor_numel(params[i]);
    return total;
}
```

### Tensor grad utility dependency

`module_zero_grad()` should not reach into tensor internals directly. It needs public tensor-level helpers:

```c
float *tensor_grad(const tensor *t);   /* return grad buffer or NULL */
void   tensor_zero_grad(tensor *t);    /* if grad exists, memset to zero */
```

Current codebase already exposes `tensor_grad()`. If that changes or is missing in target branch, add it with `module.c`. `tensor_zero_grad()` does not exist yet; add it to `tensor.h` / `tensor.c` as small wrapper around `tensor_grad()` + `memset`.

### Zero grad

```c
void module_zero_grad(module *m) {
    int n;
    tensor **params = module_parameters(m, &n);
    for (int i = 0; i < n; i++)
        tensor_zero_grad(params[i]);
}
```

Optimizers can keep their existing `*_zero_grad()` APIs. They become correct automatically once created from `module_parameters()`.

### Streaming iterator

No forward-iterator API for params in v1. Optimizers and train loops need a full `tensor **` plus count, so `module_parameters()` is enough. Add streaming traversal later only if state-dict save/load needs it to avoid building temp path arrays.

### Sequential helper

Defer. Module abstraction only needs param discovery.

A later `sequential` needs forward dispatch (`module_forward` function pointer or explicit switch). Current repo has different forward signatures, so vtable now adds noise without benefit.

## Design alternatives considered

### Pre-sized arrays (`module_init_sized`)

Rejected. Requires knowing param/child count before construction. Annoying for composites and indexed blocks.

### Separate registry object

Rejected. Adds lifecycle management and dangling registry risk. Header-first-field ties registration to module struct.

### Separate param list + child list

Rejected for this design. It works for `.parameters()`, but loses constructor registration order and makes future state-dict ordering less explicit. Unified ordered `module_item` list costs same allocations and keeps order.

### Global registry via pool

Rejected. Loses hierarchy and names. Cannot distinguish `blocks.0.q_proj.weight` from `blocks.1.q_proj.weight`.

### vtable / polymorphic forward

Rejected for now. Forward signatures differ. Need param discovery, not forward polymorphism.

## Migration plan

1. Add `include/module.h` and `src/module.c`.
2. Add unit tests for:
   - linear param count/order
   - nested child params
   - duplicate `tensor*` dedup
   - registration-after-cache assert behavior
   - `module_zero_grad()` clears existing grads
3. Refactor `linear`: embed `module base`, register `weight`/`bias`.
4. Refactor `swiglu_ffn`: embed `module base`, register `gate_proj`, `up_proj`, `down_proj` as children.
5. Refactor `transformer_block`: embed `module base`, direct norm params, child projection/FFN modules.
6. Refactor `decoder_lm`: embed `module base`, direct embedding/final norm params, `blocks.N`, `lm_head` child.
7. Refactor MNIST models: embed `module base`, direct conv params, child FC layers.
8. Replace manual training param arrays with `module_parameters()`.
9. Replace `*_num_parameters()` callers with `module_num_parameters()`.
10. Keep old `*_num_parameters()` wrappers temporarily if useful, implemented as `return module_num_parameters(&x->base);`.
