CC       ?= gcc
CFLAGS   := -Wall -Wextra -pedantic -std=c11 -O3 -ffinite-math-only -fno-signed-zeros -fno-trapping-math -g -pthread
OMP_PREFIX = /opt/homebrew/opt/libomp
INCDIRS  = -I../../include -I../../src -I$(OMP_PREFIX)/include
CPPFLAGS = -DACCELERATE_NEW_LAPACK $(INCDIRS)
OMPFLAGS = -Xpreprocessor -fopenmp

BUILDDIR = ../../build
LDFLAGS  = -L$(OMP_PREFIX)/lib -L$(BUILDDIR)
LDLIBS   = -lomp -lz -ldnn -framework Accelerate

.PHONY: all imagenet_vlm clean

all: imagenet_vlm

imagenet_vlm: imagenet_vlm.c $(BUILDDIR)/libdnn.a
	$(CC) $(CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $(BUILDDIR)/$@

clean:
	rm -f $(BUILDDIR)/imagenet_vlm
