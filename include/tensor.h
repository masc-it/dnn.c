#ifndef DNN_TENSOR_H
#define DNN_TENSOR_H

#define DNN_MAX_DIMS 8

#include "pool.h"  /* for mem_pool, _mem_pool_alloc */
typedef struct grad_fn grad_fn;

typedef struct tensor {
    void    *data;
    float   *grad;
    int      shape[DNN_MAX_DIMS];
    int      strides[DNN_MAX_DIMS];
    int      ndim;
    unsigned int requires_grad : 1;
    unsigned int contiguous : 1;    /* all dims packed with stride=product(right) */
    int      offset;
    struct tensor *parent;
    grad_fn *grad_fn;
    struct mem_pool *pool;
} tensor;

/* ── Lifecycle ── */
tensor *tensor_zeros(struct mem_pool *pool, int ndim, const int *shape, int requires_grad);
tensor *tensor_zeros_data(struct mem_pool *pool, int ndim, const int *shape);
tensor *tensor_randn(struct mem_pool *pool, int ndim, const int *shape, int requires_grad);
tensor *tensor_uniform(struct mem_pool *pool, int ndim, const int *shape, int requires_grad, float bound);

/* ── Scratch intermediate tensor (from explicit pool) ── */
tensor *tensor_scratch(struct mem_pool *pool, int ndim, const int *shape, int requires_grad);

/* ── Views ── */
tensor *tensor_slice(struct mem_pool *scratch, tensor *t, int dim, int start, int len);
tensor *tensor_transpose(struct mem_pool *scratch, tensor *t, int d1, int d2);
tensor *tensor_reshape(struct mem_pool *scratch, tensor *t, int ndim, const int *shape);
tensor *tensor_flatten(struct mem_pool *scratch, tensor *t);
tensor *tensor_contiguous(struct mem_pool *scratch, tensor *t);

/* ── Accessors ── */
float  *tensor_data_ptr(tensor *t);
int     tensor_numel(const tensor *t);
int     tensor_ndim(const tensor *t);
int     tensor_shape(const tensor *t, int dim);

/* ── Properties ── */
int     tensor_is_contiguous(const tensor *t);
int     tensor_requires_grad(const tensor *t);
void    tensor_set_requires_grad(tensor *t, int req);
int     tensor_is_leaf(const tensor *t);
void    tensor_retain_grad(tensor *t);
float  *tensor_grad(const tensor *t);
tensor *tensor_root(tensor *t);
void    tensor_print(const tensor *t);

#endif /* DNN_TENSOR_H */
