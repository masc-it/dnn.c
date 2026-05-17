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

.PHONY: all promessi_lm

all: $(BUILDDIR)/promessi_lm

promessi_lm: $(BUILDDIR)/promessi_lm

$(BUILDDIR):
	mkdir -p $@

$(LIB):
	$(MAKE) -C ../..

$(BUILDDIR)/promessi_lm: promessi_lm.c promessi_data.c promessi.h $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) \
	    promessi_lm.c promessi_data.c $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(BUILDDIR)/promessi_lm
