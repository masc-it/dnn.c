#!/usr/bin/env python3
"""
Fix grad_fn infrastructure after pool migration Phase 1+2.

1. Update autograd.c to use pool-aware allocs
2. Update _grad_fn_create call sites to pass pool param
3. Update backward functions to use fn->pool instead of scratch
4. Update function definitions missing pool params
"""

import sys
from pathlib import Path

REPO = Path(__file__).parent

def read(p):
    return (REPO / p).read_text()

def write(p, content):
    (REPO / p).write_text(content)

def edit_all(p, old, new):
    content = read(p)
    if old not in content:
        return False
    n = content.count(old)
    write(p, content.replace(old, new))
    print(f"  {p}: {n}x '{old[:50]}...' -> '{new[:50]}...'")
    return True

def edit_one(p, old, new):
    content = read(p)
    if old not in content:
        print(f"  NOT FOUND in {p}: '{old[:60]}...'")
        return False
    n = content.count(old)
    if n > 1:
        print(f"  MULTIPLE ({n}) in {p}: '{old[:60]}...'")
        return False
    write(p, content.replace(old, new))
    print(f"  {p}: '{old[:50]}...' -> '{new[:50]}...'")
    return True

# ============================================================
# Step 1: Rewrite autograd.c 
# ============================================================

AUTOGRAD_C_NEW = '''#include "autograd.h"
#include "pool.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>


/* ── Grad mode ── */

static _Thread_local int _grad_enabled = 1;

int dnn_grad_enabled(void) {
    return _grad_enabled;
}

dnn_grad_ctx dnn_no_grad_enter(void) {
    dnn_grad_ctx ctx = { ._prev = _grad_enabled };
    _grad_enabled = 0;
    return ctx;
}

void dnn_no_grad_exit(dnn_grad_ctx ctx) {
    _grad_enabled = ctx._prev;
}

/* ── grad_fn lifecycle ── */

grad_fn *_grad_fn_create(struct mem_pool *pool) {
    grad_fn *fn = _mem_pool_alloc(pool, sizeof(grad_fn), NULL);
    fn->pool = pool;
    return fn;
}

/* ── Gradient buffer management ── */

float *_grad_ensure(tensor *t) {
    tensor *base = t;
    while (base->parent) base = base->parent;
    if (!base->grad) {
        int n = tensor_numel(base);
        base->grad = _mem_pool_alloc(base->pool, n * sizeof(float), NULL);
    }
    return base->grad;
}

/* ── Backward ── */

static int _in_list(tensor *t, tensor **list, int n) {
    for (int i = 0; i < n; i++) if (list[i] == t) return 1;
    return 0;
}

/* Count reachable grad_fn nodes via DFS. Uses scratch-allocated seen array. */
static int _count_reachable(tensor *t, tensor **seen, int *n_seen) {
    if (!t->grad_fn) return 0;
    if (_in_list(t, seen, *n_seen)) return 0;
    seen[(*n_seen)++] = t;
    int count = 1;
    for (int i = 0; i < t->grad_fn->n_inputs; i++)
        count += _count_reachable(t->grad_fn->inputs[i], seen, n_seen);
    for (int i = 0; i < t->grad_fn->n_inputs; i++) {
        tensor *p = t->grad_fn->inputs[i];
        while (p->parent) {
            p = p->parent;
            if (p->grad_fn && !_in_list(p, seen, *n_seen))
                count += _count_reachable(p, seen, n_seen);
        }
    }
    return count;
}

static void _build_topo_from(tensor *t, tensor **topo, int *n,
                              tensor **seen, int *n_seen) {
    if (!t->grad_fn) return;
    if (_in_list(t, seen, *n_seen)) return;
    seen[(*n_seen)++] = t;

    for (int i = 0; i < t->grad_fn->n_inputs; i++)
        _build_topo_from(t->grad_fn->inputs[i], topo, n, seen, n_seen);

    for (int i = 0; i < t->grad_fn->n_inputs; i++) {
        tensor *p = t->grad_fn->inputs[i];
        while (p->parent) {
            p = p->parent;
            if (p->grad_fn && !_in_list(p, seen, *n_seen))
                _build_topo_from(p, topo, n, seen, n_seen);
        }
    }

    topo[(*n)++] = t;
}

void dnn_backward(struct mem_pool *scratch, tensor *loss) {
    assert(loss);

    /* First pass: count reachable grad_fn nodes */
    tensor **tmp = _mem_pool_alloc(scratch, 256 * sizeof(tensor*), NULL);
    int n_tmp = 0;
    int n_nodes = _count_reachable(loss, tmp, &n_tmp);

    tensor **topo = _mem_pool_alloc(scratch, n_nodes * sizeof(tensor*), NULL);
    tensor **seen = _mem_pool_alloc(scratch, 256 * sizeof(tensor*), NULL);
    int n_seen = 0, n = 0;
    _build_topo_from(loss, topo, &n, seen, &n_seen);
    assert(n == n_nodes && "dnn_backward: topo count mismatch");

    /* allocate and set loss gradient to all-ones */
    if (!loss->grad) {
        loss->grad = _mem_pool_alloc(loss->pool, tensor_numel(loss) * sizeof(float), NULL);
    }
    int numel = tensor_numel(loss);
    for (int i = 0; i < numel; i++) loss->grad[i] = 1.0f;

    /* reverse topological order */
    for (int i = n - 1; i >= 0; i--) {
        tensor *t = topo[i];
        grad_fn *fn = t->grad_fn;
        if (!fn) continue;
        if (!t->grad) continue;

        tensor gv;
        memset(&gv, 0, sizeof(gv));
        gv.data   = (void*)t->grad;
        gv.ndim   = t->ndim;
        memcpy(gv.shape, t->shape, t->ndim * sizeof(int));
        int stride = 1;
        for (int d = t->ndim - 1; d >= 0; d--) {
            gv.strides[d] = stride;
            stride *= t->shape[d];
        }
        gv.offset = 0;
        gv.contiguous = 1;

        fn->backward(fn, &gv);
    }
}
'''

# ============================================================
# Step 2: Update _grad_fn_create() call sites → _grad_fn_create(scratch)
# ============================================================

GRAD_FN_CALL_FILES = [
    "src/ops_elem.c",
    "src/ops_activation.c",
    "src/ops_matrix.c",
    "src/ops_reduce.c",
    "src/ops_pool.c",
    "src/attention.c",
    "src/conv.c",
    "src/embedding.c",
    "src/multihead.c",
    "src/nn.c",
    "src/norm.c",
    "src/rope.c",
    "src/tensor.c",
    "src/transformer.c",
]

def fix_grad_fn_calls():
    """Replace _grad_fn_create() with _grad_fn_create(scratch) in all source files."""
    for f in GRAD_FN_CALL_FILES:
        path = f
        edit_all(path, "_grad_fn_create()", "_grad_fn_create(scratch)")
    print("grad_fn_create calls updated")

# ============================================================
# Step 3: Fix backward functions — replace scratch/fn->pool references
# ============================================================

def fix_backward_fns():
    """
    In backward functions, replace 'scratch' with 'fn->pool'.
    Backward functions are called via fn->backward(fn, grad_output),
    and their temp allocs use the pool stored in fn->pool.
    """
    backward_files = [
        "src/ops_elem.c",
        "src/ops_activation.c",
        "src/ops_matrix.c",
        "src/ops_reduce.c",
        "src/ops_pool.c",
        "src/attention.c",
        "src/conv.c",
        "src/embedding.c",
        "src/multihead.c",
        "src/nn.c",
        "src/norm.c",
        "src/rope.c",
        "src/tensor.c",
        "src/transformer.c",
    ]
    for f in backward_files:
        # In backward functions, we need to use fn->pool since they don't have
        # their own scratch param. But we've already replaced mem_scratch_alloc
        # with _mem_pool_alloc(scratch, ...) in Phase 2.
        # In backward functions, replace scratch with fn->pool.
        content = read(f)
        
        # Only replace inside backward functions (static void ..._backward)
        # We'll do a cautious approach: replace scratch with fn->pool
        # but only where it follows _mem_pool_alloc(
        # Actually, backward functions are the only place 'scratch' appears
        # that isn't a function parameter (forward ops have scratch as param).
        # But wait — forward ops also need a 'scratch' parameter from their caller.
        # 
        # The trick: in backward functions, _mem_pool_alloc(scratch,...) needs 
        # to become _mem_pool_alloc(fn->pool,...). But in forward functions,
        # _mem_pool_alloc(scratch,...) is correct because scratch is a param.
        #
        # Since both appear in the same file, we need to be selective.
        # Backward functions are static and have '_backward' in their name.
        # Forward functions are the public API.
        #
        # Let's try: only replace scratch with fn->pool inside backward function bodies.
        # A simpler heuristic: replace 'scratch' only in lines that start with
        # whitespace containing '_mem_pool_alloc(scratch' and are inside a backward fn.
        
        # Actually, the cleanest approach: for each backward function's body,
        # find the section between its opening { and the next function definition.
        # Within that section, replace scratch with fn->pool.
        
        # For now, let's just do a blanket replacement of scratch -> fn->pool
        # in files that ONLY have backward functions (like tensor.c which has
        # static slice/reshape backward fns). For mixed files, we'll be selective.
        
        lines = content.split('\n')
        new_lines = []
        in_backward = False
        brace_depth = 0
        
        for line in lines:
            # Detect start of backward function definition
            if not in_backward and 'static void' in line and '_backward' in line:
                in_backward = True
                brace_depth = 0
            
            if in_backward:
                if '{' in line:
                    brace_depth += line.count('{')
                if '}' in line:
                    brace_depth -= line.count('}')
                    if brace_depth <= 0:
                        in_backward = False
                
                # Replace scratch with fn->pool in this line
                if 'scratch' in line and ('_mem_pool_alloc' in line or 'mem_pool_reset' in line or 'mem_pool_release' in line or 'mem_pool_mark' in line):
                    line = line.replace('scratch', 'fn->pool')
            
            new_lines.append(line)
        
        new_content = '\n'.join(new_lines)
        if new_content != content:
            write(f, new_content)
            print(f"  {f}: backward fn scratch -> fn->pool")
        else:
            print(f"  {f}: no backward changes needed")

# Actually, the approach above is too fragile. Let me instead just replace
# scratch with fn->pool in ALL source files for the backward functions.
# The forward functions get scratch as a parameter, so 'scratch' is valid there.
# But backward functions' scratch references are wrong — they need fn->pool.
# 
# Let me check: which files have both forward AND backward functions?
# The key issue is in files like ops_elem.c where both tensor_add (forward, 
# has scratch param) and add_backward (no scratch param) exist in the same file.
# 
# In the forward functions, scratch IS a valid parameter.
# In backward functions, scratch is NOT declared.
#
# Since both use _mem_pool_alloc(scratch, ...), we need to differentiate.
# 
# Simplest approach: in each file, 'scratch' is valid only if the enclosing
# function has it as a parameter. For backward functions, we need fn->pool.
# 
# Let me try a simpler heuristic: only functions defined as 'static ... backward'
# need scratch→fn->pool. I'll handle this by updating the backward function
# signatures or by doing the transformation per-function.

# Actually the simplest approach: after adding scratch as param to all ops,
# static backward functions still don't have scratch. Their _mem_pool_alloc(scratch,...)
# calls are errors. We need those to use fn->pool.
#
# But forward ops DO have scratch as param, so their _mem_pool_alloc(scratch,...) is fine.
#
# The backward functions are all named with the pattern: *_backward
# Their call to _mem_pool_alloc(scratch,...) needs scratch → fn->pool
#
# The easiest way: just replace 'scratch' with 'fn->pool' in every file,
# and then add 'mem_pool *scratch' as a local var at the top of forward functions
# that re-binds it. No, that's crazy.
#
# OK, let me just use the line-by-line approach above but make it work properly.

def fix_backward_fns_v2():
    """Replace scratch with fn->pool inside backward function bodies only."""
    bw_files = [
        "src/ops_elem.c",
        "src/ops_activation.c",
        "src/ops_matrix.c",
        "src/ops_reduce.c",
        "src/ops_pool.c",
        "src/attention.c",
        "src/conv.c",
        "src/embedding.c",
        "src/multihead.c",
        "src/nn.c",
        "src/norm.c",
        "src/rope.c",
        "src/tensor.c",
        "src/transformer.c",
    ]
    for f in bw_files:
        content = read(f)
        lines = content.split('\n')
        result = []
        in_backward = False
        brace_depth = 0
        
        for line in lines:
            # Detect start of backward function
            stripped = line.strip()
            if not in_backward and stripped.startswith('static void ') and '_backward' in stripped:
                in_backward = True
                brace_depth = stripped.count('{') - stripped.count('}')
                result.append(line)
                continue
            
            if in_backward:
                brace_depth += line.count('{') - line.count('}')
                # Replace scratch with fn->pool in allocs
                if 'scratch' in line:
                    line = line.replace('scratch', 'fn->pool')
                result.append(line)
                if brace_depth <= 0:
                    in_backward = False
            else:
                result.append(line)
        
        new_content = '\n'.join(result)
        if new_content != content:
            write(f, new_content)
            print(f"  {f}: backward functions updated")
        else:
            print(f"  {f}: no backward changes")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, help="1=autograd.c, 2=grad_fn calls, 3=backward fns")
    args = parser.parse_args()
    
    if args.step == 1:
        write("src/autograd.c", AUTOGRAD_C_NEW)
        print("Step 1: autograd.c rewritten")
    elif args.step == 2:
        fix_grad_fn_calls()
        print("Step 2 done")
    elif args.step == 3:
        fix_backward_fns_v2()
        print("Step 3 done")
