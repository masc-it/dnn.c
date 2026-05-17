CC       ?= gcc
CFLAGS   := -Wall -Wextra -pedantic -std=c11 -O3 -ffinite-math-only -fno-signed-zeros -fno-trapping-math -g
OMP_PREFIX = /opt/homebrew/opt/libomp
INCDIRS  = -I../../include -I../../src -I$(OMP_PREFIX)/include
CPPFLAGS = -DACCELERATE_NEW_LAPACK $(INCDIRS)
OMPFLAGS = -Xpreprocessor -fopenmp

BUILDDIR = ../../build
LDFLAGS  = -L$(OMP_PREFIX)/lib -L$(BUILDDIR)
LDLIBS   = -lomp -lz -ldnn -framework Accelerate

.PHONY: all imagenet_vlm vlm_init_probe clean

all: imagenet_vlm vlm_init_probe

imagenet_vlm: imagenet_vlm.c $(BUILDDIR)/libdnn.a
	$(CC) $(CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $(BUILDDIR)/$@

vlm_init_probe: vlm_init_probe.c $(BUILDDIR)/libdnn.a
	$(CC) $(CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $(BUILDDIR)/$@
	$(CC) $(CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $(BUILDDIR)/$@

clean:
	rm -f $(BUILDDIR)/imagenet_vlm $(BUILDDIR)/vlm_init_probe
