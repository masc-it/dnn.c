# State Dict & Summary — Design Spec

## Problem

Two missing pieces after module registration tree is in place:

1. **No way to persist weights.** Fine-tuning or resuming requires re-training.
2. **No way to inspect model structure.** Debugging means reading source or printing tensor shapes by hand. No `print(model)` equivalent.

Both are buildable from existing module registration data: child names, param names, and `type_name`.

## Shared assumptions

V1 state-dict support only handles registered params that are real, contiguous, pool-owned tensors.

`module_param()` should enforce this invariant:

```c
assert(t->pool == m->pool && "param must live in module params pool");
assert(t->parent == NULL && "registered params must not be views");
assert(tensor_is_contiguous(t) && "registered params must be contiguous");
```

Reason: save/load uses `tensor_data_ptr()` and raw contiguous `float32` bytes. Non-contiguous params/views are out of scope for v1.

Weight tying rule for v1:

- register one real tensor as param
- do not register transpose/slice views as params
- if a forward path needs transpose-tied weight, create/use view during forward, not as persistent registered param

Also keep concepts separate:

- `module_parameters()` returns **unique** recursive params for optimizers.
- state dict traversal walks **named registered param paths** and may include duplicate tensor pointers if user registers same tensor under multiple names.

## Module Summary

### Goal

`module_summary(m)` prints module tree to stdout: type name, child structure, param names, shapes, and counts. Like `print(model)` in PyTorch.

### API

```c
/* Print module tree to stdout.
 *
 *   m      — root module
 *   indent — initial indent level (0 for root)
 *   detail — if non-zero, print direct param tensor names + shapes
 */
void module_summary(module *m, int indent, int detail);
```

### Output format (detail=1)

```
decoder_lm [<total> params]
  embedding_table: [50000, 256] (12800000)
  norm_weight: [256] (256)
  norm_bias: [256] (256)
  blocks.0:
    transformer_block [<block total> params]
      q_proj:
        linear [65792 params]
          weight: [256, 256] (65536)
          bias: [256] (256)
      k_proj:
        linear [65792 params]
          weight: [256, 256] (65536)
          bias: [256] (256)
      attn_norm_weight: [256] (256)
      attn_norm_bias: [256] (256)
      ffn:
        swiglu_ffn [<ffn total> params]
          gate_proj:
            linear ...
  lm_head:
    linear [50000 params]
      bias: [50000] (50000)
```

### Output format (detail=0)

```
decoder_lm [<total> params]
  blocks.0: transformer_block [<block total> params]
  blocks.1: transformer_block [<block total> params]
  lm_head: linear [50000 params]
```

### Implementation sketch

```c
static void print_indent(int indent) {
    for (int i = 0; i < indent; i++) printf("  ");
}

static void print_shape(tensor *t) {
    printf("[");
    for (int d = 0; d < t->ndim; d++)
        printf("%s%d", d ? ", " : "", t->shape[d]);
    printf("]");
}

void module_summary(module *m, int indent, int detail) {
    print_indent(indent);
    printf("%s [%lld params]\n", m->type_name, module_num_parameters(m));

    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_PARAM && detail) {
            tensor *t = item->as.param;
            print_indent(indent + 1);
            printf("%s: ", item->name);
            print_shape(t);
            printf(" (%d)\n", tensor_numel(t));
        } else if (item->kind == MODULE_ITEM_CHILD) {
            module *child = item->as.child;
            if (detail) {
                print_indent(indent + 1);
                printf("%s:\n", item->name);
                module_summary(child, indent + 2, detail);
            } else {
                print_indent(indent + 1);
                printf("%s: %s [%lld params]\n",
                       item->name, child->type_name, module_num_parameters(child));
            }
        }
    }
}
```

Notes:

- `module_num_parameters()` uses optimizer-style unique `tensor*` params.
- If same tensor is shared across sibling subtrees, root count can be less than sum of printed child counts. Accept for v1.
- Stream-to-`FILE*` can be added later as `module_summary_file(module*, FILE*, int, int)`.

---

## State Dict

### Goal

```c
/* Save all registered named params to binary file. */
void module_save(module *m, const char *path);

/* Load weights into already-constructed model.
 * strict != 0:
 *   - unknown file key asserts
 *   - missing live key asserts
 * strict == 0:
 *   - unknown file key skipped
 *   - missing live key left unchanged
 * Shape mismatch for a matched key always asserts.
 */
void module_load(module *m, const char *path, int strict);
```

### Design

Two-phase behavior, like PyTorch `state_dict()` / `load_state_dict()`:

1. **Save**: walk module tree depth-first by registration order. For each registered param path, write `(full_name, shape, raw_float_data)`.
2. **Load**: read file entries, find live param by full name, verify exact shape, then copy raw data.

Important: save entry count comes from named state traversal, **not** `module_parameters()`. `module_parameters()` dedups for optimizer use; state dict is path-based.

### Binary format v1

Little-endian, no padding. Magic is written as bytes, not as host-endian int.

```
+0:  magic        — 4 bytes: 'D', 'N', 'N', 'M'
+4:  version      — uint32, 1
+8:  header_size  — uint32, 16 (byte offset of first entry)
+12: n_entries    — uint32

[n_entries entries:]
    name_len   — uint32, includes null terminator
    name       — name_len bytes, UTF-8 C string
    ndim       — uint32
    shape      — ndim * int32
    numel      — uint32
    data       — numel * 4 bytes, raw float32, contiguous row-major
```

Validation:

- `magic == "DNNM"`
- `version == 1`
- `header_size == 16`
- `0 < name_len < MODULE_STATE_MAX_NAME` (4096)
- `0 < ndim <= DNN_MAX_DIMS`
- all shape dims > 0
- product(shape) == `numel`
- matched live tensor has same `ndim` and same shape

V1 may assert little-endian host. Portable byte-swapping helpers can come later.

### Rationale for flat path format

- Simple read/write: one recursive save walk, one load loop.
- Load does not reconstruct tree; it matches strings against live model.
- Strict flag controls compatibility:
  - strict load catches architecture mismatch
  - non-strict load supports partial checkpoints / fine-tuning
- File is sequential and can later support selective loading.

### Save implementation sketch

```c
#define MODULE_STATE_MAX_NAME 4096

static int state_count_walk(module *m) {
    int n = 0;
    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_PARAM)
            n++;
        else if (item->kind == MODULE_ITEM_CHILD)
            n += state_count_walk(item->as.child);
    }
    return n;
}

void module_save(module *m, const char *path) {
    FILE *f = fopen(path, "wb");
    assert(f);

    char magic[4] = {'D', 'N', 'N', 'M'};
    uint32_t version = 1;
    uint32_t header_size = 16;
    uint32_t n_entries = (uint32_t)state_count_walk(m);

    fwrite(magic, 1, 4, f);
    fwrite(&version, 4, 1, f);
    fwrite(&header_size, 4, 1, f);
    fwrite(&n_entries, 4, 1, f);

    char path_buf[MODULE_STATE_MAX_NAME];
    path_buf[0] = '\0';
    save_walk(m, f, path_buf, 0);

    fclose(f);
}
```

Path writing:

```c
static void write_entry(FILE *f, const char *name, tensor *t) {
    assert(t->parent == NULL);
    assert(tensor_is_contiguous(t));

    uint32_t name_len = (uint32_t)strlen(name) + 1;
    uint32_t ndim = (uint32_t)t->ndim;
    uint32_t numel = (uint32_t)tensor_numel(t);

    fwrite(&name_len, 4, 1, f);
    fwrite(name, 1, name_len, f);
    fwrite(&ndim, 4, 1, f);
    for (int d = 0; d < t->ndim; d++) {
        int32_t dim = (int32_t)t->shape[d];
        fwrite(&dim, 4, 1, f);
    }
    fwrite(&numel, 4, 1, f);
    fwrite(tensor_data_ptr(t), sizeof(float), numel, f);
}

static void save_walk(module *m, FILE *f, char *path, int pos) {
    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_PARAM) {
            int n = snprintf(path + pos, MODULE_STATE_MAX_NAME - pos,
                             "%s", item->name);
            assert(n > 0 && pos + n < MODULE_STATE_MAX_NAME);
            write_entry(f, path, item->as.param);
            path[pos] = '\0';
        } else if (item->kind == MODULE_ITEM_CHILD) {
            int n = snprintf(path + pos, MODULE_STATE_MAX_NAME - pos,
                             "%s.", item->name);
            assert(n > 0 && pos + n < MODULE_STATE_MAX_NAME);
            save_walk(item->as.child, f, path, pos + n);
            path[pos] = '\0';
        }
    }
}
```

### Load implementation sketch

Strict mode needs a seen set to detect missing live params. Use heap `calloc`/`free` for this small temporary bitset; it is not model-owned and should not consume params pool.

```c
typedef struct module_find_result {
    tensor *t;
    int     index;  /* path index in state traversal order, or -1 */
} module_find_result;

static module_find_result module_find_param_index(module *m, const char *name);

void module_load(module *m, const char *path, int strict) {
    FILE *f = fopen(path, "rb");
    assert(f);

    char magic[4];
    uint32_t version, header_size, n_entries;
    fread(magic, 1, 4, f);
    assert(memcmp(magic, "DNNM", 4) == 0 && "bad magic");
    fread(&version, 4, 1, f);
    fread(&header_size, 4, 1, f);
    fread(&n_entries, 4, 1, f);
    assert(version == 1);
    assert(header_size == 16);

    int n_live = state_count_walk(m);
    unsigned char *seen = strict ? calloc((size_t)n_live, 1) : NULL;

    for (uint32_t i = 0; i < n_entries; i++) {
        char name[MODULE_STATE_MAX_NAME];
        uint32_t name_len, ndim, numel;
        int32_t shape[DNN_MAX_DIMS];

        fread(&name_len, 4, 1, f);
        assert(name_len > 0 && name_len < MODULE_STATE_MAX_NAME);
        fread(name, 1, name_len, f);
        assert(name[name_len - 1] == '\0');

        fread(&ndim, 4, 1, f);
        assert(ndim > 0 && ndim <= DNN_MAX_DIMS);
        for (uint32_t d = 0; d < ndim; d++) {
            fread(&shape[d], 4, 1, f);
            assert(shape[d] > 0);
        }
        fread(&numel, 4, 1, f);
        assert(numel == product(shape, ndim));

        module_find_result r = module_find_param_index(m, name);
        if (!r.t) {
            if (strict) assert(!"module_load: unknown param");
            fseek(f, (long)numel * (long)sizeof(float), SEEK_CUR);
            continue;
        }

        assert((uint32_t)r.t->ndim == ndim && "module_load: ndim mismatch");
        for (uint32_t d = 0; d < ndim; d++)
            assert(r.t->shape[d] == shape[d] && "module_load: shape mismatch");

        fread(tensor_data_ptr(r.t), sizeof(float), numel, f);
        if (seen) seen[r.index] = 1;
    }

    if (strict) {
        for (int i = 0; i < n_live; i++)
            assert(seen[i] && "module_load: missing live param in checkpoint");
        free(seen);
    }

    fclose(f);
}
```

### Helper: find by full path

```c
/* Find param by full state path. Linear scan is fine for <10K entries. */
tensor *module_find_param(module *m, const char *name);
```

For strict missing checks, internal implementation should also support returning traversal index:

```c
static module_find_result module_find_param_index(module *m, const char *name);
```

Implementation: recursive walk with same path-building logic as save, compare full path to `name`, increment index for every registered param path.

### Strict behavior

`strict != 0`:

- file param not found in live model: assert
- live model param not present in file: assert
- matched param shape mismatch: assert

`strict == 0`:

- file param not found in live model: skip bytes
- live model param not present in file: leave unchanged
- matched param shape mismatch: assert

This supports loading partial checkpoints while still refusing shape-corrupt matches.

### Weight tying / aliases

V1 writes every registered param path. If same `tensor*` is registered twice, file gets two entries. On load, both entries may copy into same tensor. This is harmless if bytes match, but wasteful.

For current tied LM-head case, do not register a persistent transpose/view as a param. Register `embedding_table` once; create/use transpose view in forward as needed.

Future improvement: explicit alias metadata or storage dedup. Not needed for v1.

### Memory ownership

- **Save**: reads param data, writes file. Stack path buffer only.
- **Load**: copies into existing tensors in params pool. Uses stack name/shape buffers plus optional heap `seen` bitset for strict mode.

---

## Future extensions

Both features use only registration tree. They can be implemented and tested independently.

Downstream additions:

- checkpoint resume — save every N epochs, load with `strict=1`
- fine-tuning — load base with `strict=0`, leave new head unchanged
- inference export — load into eval-only binary
- model surgery — load compatible subtrees, swap modules, save modified
- `FILE*` variants — `module_save_file`, `module_load_file`, `module_summary_file`
