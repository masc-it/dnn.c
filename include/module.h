#ifndef DNN_MODULE_H
#define DNN_MODULE_H

#include "tensor.h"

/* ── Module abstraction (loosely PyTorch nn.Module) ──
 *
 * Every model/layer struct embeds `module base` as first field.
 * Constructor registers direct params and child modules via
 * module_param() / module_add_child().  Recursive parameter
 * discovery (module_parameters()) derives from that registration.
 *
 * No type enum — type_name is a free string set by constructor.
 * No unified forward — typed forward signatures stay per-model.
 */

typedef enum module_item_kind {
    MODULE_ITEM_PARAM = 1,
    MODULE_ITEM_CHILD = 2,
} module_item_kind;

/* One registration entry — small alloc from params pool. */
typedef struct module_item {
    struct module_item *next;
    module_item_kind    kind;
    const char         *name;   /* stable pointer: "weight", "fc1", ... */
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
    tensor         **flat_params;
    int              n_flat;
} module;

/* ── Lifecycle ── */

/* Initialize header. pool must be the params pool (never reset). */
void module_init(module *m, struct mem_pool *pool, const char *type_name);

/* Register a direct parameter tensor. t must be stable in params pool.
   name is required, must outlive module (string literal or pool copy).
   Asserts that neither m nor any ancestor has been introspected. */
void module_param(module *m, const char *name, tensor *t);

/* Register a child submodule. Sets child->parent = m.
   Asserts child->parent == NULL. */
void module_add_child(module *m, const char *name, module *child);

/* ── Introspection ── */

/* All unique recursive params, depth-first in registration order.
   Builds flat cache once; subsequent calls are O(1).
   n_out receives the count. */
tensor **module_parameters(module *m, int *n_out);

/* Total scalar elements across all unique recursive params. */
long long module_num_parameters(module *m);

/* Zero grad buffers for all unique recursive params. */
void module_zero_grad(module *m);

/* ── Inspection ── */

/* Print module tree to stdout.
 *
 *   m      — root module
 *   indent — initial indent level (0 for root)
 *   detail — if non-zero, print direct param tensor names + shapes
 */
void module_summary(module *m, int indent, int detail);

/* Find a parameter tensor by its full state path, e.g. "blocks.0.q_proj.weight".
 * Returns NULL if not found. */
tensor *module_find_param(module *m, const char *name);

/* ── State dict save/load ── */

#define MODULE_STATE_MAX_NAME 4096

/* Save all registered named params to a binary file.
 * Format: magic(4) + version(4) + header_size(4) + n_entries(4) +
 *   entries[n_entries] of {name_len(4), name, ndim(4), shape[], numel(4), data}.
 * All ints are uint32 little-endian. */
void module_save(module *m, const char *path);

/* Load weights into an already-constructed model.
 *
 *   strict != 0:
 *     - unknown file key asserts
 *     - missing live key asserts
 *   strict == 0:
 *     - unknown file key skipped
 *     - missing live key left unchanged
 *   Shape mismatch for a matched key always asserts. */
void module_load(module *m, const char *path, int strict);


#endif /* DNN_MODULE_H */
