#include "module.h"
#include "pool.h"
#include <string.h>
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

void module_zero_grad(module *m) {
    int n;
    tensor **params = module_parameters(m, &n);
    for (int i = 0; i < n; i++)
        tensor_zero_grad(params[i]);
}
