# dnn.c — Design Spec

1. Memory Pool
2. Tensor
3. Ops
4. Autograd

## 1. Memory Pool

Foundation of all memory management. Three global bump allocators — no heap allocs during training.

| Pool | Reset? | Contents |
|------|--------|----------|
| `pool_params` | Never | Weights, biases, grads, optimizer state, tensor structs |
| `pool_scratch` | Each step | Activations, saved tensors for backward, grad_fn nodes |
| `pool_data` | Per batch | Raw file bytes, decoded images, preprocessing temps, batch float tensors |

### Rules

- **All pooled.** tensor structs, data buffers, grad buffers, grad_fn structs, saved tensors, batch data — every allocation goes through one of the three pools. No raw `malloc`/`calloc`.
- **No `free` on pooled memory.** Pool owns everything. `mem_pool_destroy` does one `free` for the entire buffer.
- **Scratch invalidates on reset.** Saved tensors must be consumed before `pool_reset(pool_scratch)`. Guaranteed: backward runs, optimizer reads grads from params pool, then reset.
- **Data pool resets per batch.** Dataloader owns this lifecycle. Model never touches it directly.
- **OOM is fatal.** v1 uses 1GB pools. MNIST CNN touches <1MB. If hit, increase pool size.


## 2. Tensor

The foundational computation node.

## 3. Ops

All output tensors allocate from `pool_scratch`. All input tensors are assumed to be pooled (params, scratch, or data). Free-list ops (in-place relu) reuse the input buffer.

Each op that requires gradient registers a `grad_fn` on its output. The `grad_fn` is allocated from `pool_scratch` and holds:
- pointers to input tensors (for gradient propagation)
- saved tensors (e.g., relu mask, matmul input) allocated from `pool_scratch`


## 4. Autograd

### grad_fn struct

```c
struct grad_fn {
    void     (*backward)(grad_fn *fn, tensor *grad_output);
    tensor  **inputs;           // inputs to the forward op
    int       n_inputs;
    tensor  **saved_tensors;    // saved for backward
    int       n_saved;
};
```

Allocated from `pool_scratch`. Created during forward pass, consumed during backward pass, invalidated on next `pool_reset(pool_scratch)`.


### Backward

```c
void dnn_backward(tensor *loss);
```

Topological sort of the computation graph. Walks `grad_fn` chain from loss backward. Calls each node's `backward` function. Gradients accumulate into `t->grad` buffers in `pool_params`.

### Tape mechanics

Each differentiable op:
1. Reads input tensors (pooled, anywhere)
2. Allocates output tensor from `pool_scratch`
3. If any input requires grad, allocates `grad_fn` from `pool_scratch`
4. Saves needed tensors in `grad_fn->saved_tensors` (from `pool_scratch`)
5. Stores grad_fn on output tensor

After `dnn_backward` returns, `pool_reset(pool_scratch)` invalidates all gradients savings.

## Memory model

```
pool_params buffer (1GB, never reset):
  [ tensor structs   ]  ~1KB
  [ weight data      ]  ~123KB
  [ bias data        ]  ~1KB
  [ grad buffers     ]  ~124KB
  [ optimizer state  ]  ~248KB (if Adam)
  ≈ 500KB used

pool_scratch buffer (1GB, reset per step):
  [ activations      ]  ~120KB
  [ saved tensors    ]  ~64KB
  [ grad_fn nodes    ]  ~1KB
  ≈ 185KB peak, reset to 0 each step

pool_data buffer (1GB, reset per batch):
  [ raw image bytes  ]  ~1.5MB (batch 64 × 28×28)
  [ decode temps     ]  ~256KB
  [ batch tensor x   ]  ~200KB
  [ batch tensor y   ]  ~256B
  ≈ 2MB peak, reset to 0 each batch
```
