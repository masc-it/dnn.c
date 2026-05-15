CC ?= gcc
CFLAGS := -Wall -Wextra -pedantic -std=c11 -O3 -g
EXTRA_CFLAGS ?=
OMP_PREFIX ?= /opt/homebrew/opt/libomp
INCDIRS := -I../../include -I$(OMP_PREFIX)/include
CPPFLAGS := -DACCELERATE_NEW_LAPACK $(INCDIRS)
OMPFLAGS := -Xpreprocessor -fopenmp
LDFLAGS := -L$(OMP_PREFIX)/lib -L../../build
LDLIBS := -lomp -lz -ldnn -framework Accelerate

BUILDDIR := ../../build
LIB := $(BUILDDIR)/libdnn.a
COMMON := mnist_data.c mnist_train.c

.PHONY: all mnist_mlp mnist_cnn mnist_cnn_pool

all: $(BUILDDIR)/mnist_mlp $(BUILDDIR)/mnist_cnn $(BUILDDIR)/mnist_cnn_pool

# phony convenience targets so top-level 'make -C examples/mnist mnist_cnn' works
mnist_mlp: $(BUILDDIR)/mnist_mlp
mnist_cnn: $(BUILDDIR)/mnist_cnn
mnist_cnn_pool: $(BUILDDIR)/mnist_cnn_pool

$(BUILDDIR):
	mkdir -p $@

$(LIB):
	$(MAKE) -C ../..

$(BUILDDIR)/mnist_mlp: mnist_mlp.c mnist_mlp_model.c $(COMMON) $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_mlp.c mnist_mlp_model.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILDDIR)/mnist_cnn: mnist_cnn.c mnist_cnn_model.c $(COMMON) $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_cnn.c mnist_cnn_model.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILDDIR)/mnist_cnn_pool: mnist_cnn_pool.c mnist_cnn_pool_model.c $(COMMON) $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_cnn_pool.c mnist_cnn_pool_model.c $(COMMON) $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(BUILDDIR)/mnist_mlp $(BUILDDIR)/mnist_cnn $(BUILDDIR)/mnist_cnn_pool

# 'make -C examples/mnist distclean' also rebuilds lib
# 'make clean' in root already rm -rf build/ so no extra work needed
