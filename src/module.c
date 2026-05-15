#include "module.h"
#include "pool.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

/* ── Helpers ── */

/* Linear scan for dedup. n is typically < 10K. */
static int contains_tensor(tensor **arr, int n, tensor *t) {
    for (int i = 0; i < n; i++)
        if (arr[i] == t) return 1;
    return 0;
}

/* Recursive count of direct params (upper bound for allocation). */
static int count_params_recursive(module *m) {
    int count = m->n_direct_params;
    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_CHILD && item->as.child)
            count += count_params_recursive(item->as.child);
    }
    return count;
}

/* Recursive fill with dedup. Returns next index to write. */
static int fill_params_recursive(module *m, tensor **arr, int idx) {
    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_PARAM) {
            if (!contains_tensor(arr, idx, item->as.param))
                arr[idx++] = item->as.param;
        } else {
            idx = fill_params_recursive(item->as.child, arr, idx);
        }
    }
    return idx;
}

/* Check that m and all ancestors have no flat cache (registration invariant). */
static void assert_no_cache_in_ancestors(module *m) {
    (void)m;
#ifndef NDEBUG
    while (m) {
        assert(m->flat_params == NULL
               && "module: registration invalid after introspection");
        m = m->parent;
    }
#endif
}

/* ── Lifecycle ── */

void module_init(module *m, struct mem_pool *pool, const char *type_name) {
    assert(m);
    assert(pool);
    assert(type_name);
    m->pool      = pool;
    m->type_name = type_name;
    m->parent    = NULL;
    m->items_head  = NULL;
    m->items_tail  = NULL;
    m->n_items     = 0;
    m->n_direct_params = 0;
    m->n_children      = 0;
    m->flat_params = NULL;
    m->n_flat      = 0;
}

void module_param(module *m, const char *name, tensor *t) {
    assert(m);
    assert(name);
    assert(t);
    assert(t->pool == m->pool && "param must live in module params pool");
    assert(t->parent == NULL && "registered params must not be views");
    assert(tensor_is_contiguous(t) && "registered params must be contiguous");
    assert_no_cache_in_ancestors(m);

    module_item *item = _mem_pool_alloc(m->pool, sizeof(module_item), NULL);
    item->next    = NULL;
    item->kind    = MODULE_ITEM_PARAM;
    item->name    = name;
    item->as.param = t;

    if (m->items_tail)
        m->items_tail->next = item;
    else
        m->items_head = item;
    m->items_tail = item;
    m->n_items++;
    m->n_direct_params++;
}

void module_add_child(module *m, const char *name, module *child) {
    assert(m);
    assert(name);
    assert(child);
    assert(child != m && "module cannot be child of itself");
    assert(child->pool == m->pool && "child must use same params pool");
    assert(child->parent == NULL && "child already has a parent");
    assert_no_cache_in_ancestors(m);

    module_item *item = _mem_pool_alloc(m->pool, sizeof(module_item), NULL);
    item->next    = NULL;
    item->kind    = MODULE_ITEM_CHILD;
    item->name    = name;
    item->as.child = child;

    if (m->items_tail)
        m->items_tail->next = item;
    else
        m->items_head = item;
    m->items_tail = item;
    m->n_items++;
    m->n_children++;

    child->parent = m;
}

/* ── Introspection ── */

tensor **module_parameters(module *m, int *n_out) {
    assert(m);
    assert(n_out);

    /* Return cached if available. */
    if (m->flat_params) {
        *n_out = m->n_flat;
        return m->flat_params;
    }

    /* First call — count max, allocate, fill with dedup. */
    int max = count_params_recursive(m);
    tensor **arr = _mem_pool_alloc(m->pool, (size_t)max * sizeof(tensor *), NULL);
    int n = fill_params_recursive(m, arr, 0);

    m->flat_params = arr;
    m->n_flat      = n;
    *n_out = n;
    return arr;
}

long long module_num_parameters(module *m) {
    int n;
    tensor **params = module_parameters(m, &n);
    long long total = 0;
    for (int i = 0; i < n; i++)
        total += tensor_numel(params[i]);
    return total;
}

/* ── State dict ── */

typedef struct {
    tensor *t;
    int     index;
} module_find_result;

static module_find_result _find_param_index_walk(module *m, const char *target,
                                                   char *path, int pos, int *idx);

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

static void write_entry(FILE *f, const char *name, tensor *t) {
    uint32_t name_len = (uint32_t)strlen(name) + 1;
    uint32_t ndim     = (uint32_t)t->ndim;
    uint32_t numel    = (uint32_t)tensor_numel(t);

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

void module_save(module *m, const char *path) {
    assert(m);
    assert(path);

    FILE *f = fopen(path, "wb");
    assert(f && "module_save: cannot open file");

    char    magic[4]     = {'D', 'N', 'N', 'M'};
    uint32_t version     = 1;
    uint32_t header_size = 16;
    uint32_t n_entries   = (uint32_t)state_count_walk(m);

    fwrite(magic, 1, 4, f);
    fwrite(&version, 4, 1, f);
    fwrite(&header_size, 4, 1, f);
    fwrite(&n_entries, 4, 1, f);

    char path_buf[MODULE_STATE_MAX_NAME];
    path_buf[0] = '\0';
    save_walk(m, f, path_buf, 0);

    fclose(f);
}

static uint32_t product(int32_t *shape, int ndim) {
    uint32_t p = 1;
    for (int d = 0; d < ndim; d++) p *= (uint32_t)shape[d];
    return p;
}

static module_find_result module_find_param_index(module *m, const char *name) {
    char path[4096];
    path[0] = '\0';
    int idx = 0;
    return _find_param_index_walk(m, name, path, 0, &idx);
}

static module_find_result _find_param_index_walk(module *m, const char *target,
                                                   char *path, int pos, int *idx) {
    module_find_result r = {NULL, -1};
    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_PARAM) {
            int n = snprintf(path + pos, 4096 - pos, "%s", item->name);
            assert(n > 0 && pos + n < 4096);
            if (strcmp(path, target) == 0) {
                r.t     = item->as.param;
                r.index = *idx;
                path[pos] = '\0';
                return r;
            }
            (*idx)++;
            path[pos] = '\0';
        } else if (item->kind == MODULE_ITEM_CHILD) {
            int n = snprintf(path + pos, 4096 - pos, "%s.", item->name);
            assert(n > 0 && pos + n < 4096);
            r = _find_param_index_walk(item->as.child, target, path, pos + n, idx);
            if (r.t) return r;
            path[pos] = '\0';
        }
    }
    return r;
}

void module_load(module *m, const char *path, int strict) {
    assert(m);
    assert(path);

    FILE *f = fopen(path, "rb");
    assert(f && "module_load: cannot open file");

    char     magic[4];
    uint32_t version, header_size, n_entries;
    fread(magic, 1, 4, f);
    assert(memcmp(magic, "DNNM", 4) == 0 && "module_load: bad magic");
    fread(&version, 4, 1, f);
    fread(&header_size, 4, 1, f);
    fread(&n_entries, 4, 1, f);
    assert(version == 1 && "module_load: unsupported version");
    assert(header_size == 16 && "module_load: unexpected header size");

    int n_live = state_count_walk(m);
    unsigned char *seen = strict ? calloc((size_t)n_live, 1) : NULL;

    for (uint32_t i = 0; i < n_entries; i++) {
        char     name[MODULE_STATE_MAX_NAME];
        uint32_t name_len, ndim, numel;
        int32_t  shape[DNN_MAX_DIMS];

        size_t nr = fread(&name_len, 4, 1, f);
        assert(nr == 1 && "module_load: truncated file");
        assert(name_len > 0 && name_len < MODULE_STATE_MAX_NAME);

        nr = fread(name, 1, name_len, f);
        assert(nr == name_len && "module_load: truncated file");
        assert(name[name_len - 1] == '\0');

        nr = fread(&ndim, 4, 1, f);
        assert(nr == 1 && "module_load: truncated file");
        assert(ndim > 0 && ndim <= DNN_MAX_DIMS);

        nr = fread(shape, 4, ndim, f);
        assert(nr == ndim && "module_load: truncated file");
        for (uint32_t d = 0; d < ndim; d++) assert(shape[d] > 0);

        nr = fread(&numel, 4, 1, f);
        assert(nr == 1 && "module_load: truncated file");
        assert(numel == product(shape, (int)ndim));

        module_find_result r = module_find_param_index(m, name);
        if (!r.t) {
            assert(!strict && "module_load: unknown param");
            fseek(f, (long)numel * (long)sizeof(float), SEEK_CUR);
            continue;
        }

        assert((uint32_t)r.t->ndim == ndim && "module_load: ndim mismatch");
        for (uint32_t d = 0; d < ndim; d++)
            assert(r.t->shape[d] == shape[d] && "module_load: shape mismatch");

        nr = fread(tensor_data_ptr(r.t), sizeof(float), numel, f);
        assert(nr == numel && "module_load: truncated file");
        if (seen) seen[r.index] = 1;
    }

    if (strict) {
        for (int i = 0; i < n_live; i++)
            assert(seen[i] && "module_load: missing live param in checkpoint");
        free(seen);
    }

    fclose(f);
}

void module_zero_grad(module *m) {
    int n;
    tensor **params = module_parameters(m, &n);
    for (int i = 0; i < n; i++)
        tensor_zero_grad(params[i]);
}

/* ── Inspection ── */

static tensor *_find_param_walk(module *m, const char *target, char *path, int pos);

static void print_indent(int indent) {
    for (int i = 0; i < indent; i++) printf("  ");
}

static void print_shape(tensor *t) {
    printf("[");
    for (int d = 0; d < t->ndim; d++)
        printf("%s%d", d ? ", " : "", t->shape[d]);
    printf("]");
}

tensor *module_find_param(module *m, const char *name) {
    assert(m);
    assert(name);
    char path[4096];
    path[0] = '\0';
    return _find_param_walk(m, name, path, 0);
}

static tensor *_find_param_walk(module *m, const char *target, char *path, int pos) {
    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_PARAM) {
            int n = snprintf(path + pos, 4096 - pos, "%s", item->name);
            assert(n > 0 && pos + n < 4096);
            if (strcmp(path, target) == 0) {
                path[pos] = '\0';
                return item->as.param;
            }
            path[pos] = '\0';
        } else if (item->kind == MODULE_ITEM_CHILD) {
            int n = snprintf(path + pos, 4096 - pos, "%s.", item->name);
            assert(n > 0 && pos + n < 4096);
            tensor *r = _find_param_walk(item->as.child, target, path, pos + n);
            if (r) return r;
            path[pos] = '\0';
        }
    }
    return NULL;
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
                       item->name, child->type_name,
                       module_num_parameters(child));
            }
        }
    }
}
