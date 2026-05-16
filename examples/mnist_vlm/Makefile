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

.PHONY: all mnist_vlm mnist_vlm_preds

all: $(BUILDDIR)/mnist_vlm $(BUILDDIR)/mnist_vlm_preds

mnist_vlm: $(BUILDDIR)/mnist_vlm

mnist_vlm_preds: $(BUILDDIR)/mnist_vlm_preds

$(BUILDDIR):
	mkdir -p $@

$(LIB):
	$(MAKE) -C ../..

$(BUILDDIR)/mnist_vlm: mnist_vlm.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_vlm.c $(LDFLAGS) $(LDLIBS) -o $@

$(BUILDDIR)/mnist_vlm_preds: mnist_vlm_preds.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    mnist_vlm_preds.c $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(BUILDDIR)/mnist_vlm $(BUILDDIR)/mnist_vlm_preds
