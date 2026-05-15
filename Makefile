CC        ?= gcc
CFLAGS    := -Wall -Wextra -pedantic -std=c11 -O3 -ffast-math -g
EXTRA_CFLAGS ?=
OMP_PREFIX = /opt/homebrew/opt/libomp
INCDIRS    = -Iinclude -Isrc -I$(OMP_PREFIX)/include
CPPFLAGS   = -DACCELERATE_NEW_LAPACK $(INCDIRS)
OMPFLAGS   = -Xpreprocessor -fopenmp
LDFLAGS    = -L$(OMP_PREFIX)/lib -L$(BUILDDIR)
LDLIBS     = -lomp -lz -ldnn -framework Accelerate

SRCDIR   = src
OBJDIR   = obj
BUILDDIR = build
TESTDIR  = test
BENCHDIR = bench

SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
DEPS     = $(OBJS:.o=.d)

TEST_SRCS = $(wildcard $(TESTDIR)/*.c)
TEST_OBJDIR = $(BUILDDIR)/test
TEST_BINS = $(patsubst $(TESTDIR)/%.c, $(TEST_OBJDIR)/%, $(TEST_SRCS))

BENCH_SRCS = $(wildcard $(BENCHDIR)/*.c)
BENCH_BINS = $(patsubst $(BENCHDIR)/%.c, $(BUILDDIR)/%, $(BENCH_SRCS))

LIB      = $(BUILDDIR)/libdnn.a

.PHONY: all clean test run bench main main_lm main_prep_data run_lm
.PHONY: bench_conv2d bench_matmul bench_ops bench_multihead bench_all

all: $(OBJDIR) $(TEST_OBJDIR) $(LIB)

$(OBJDIR) $(TEST_OBJDIR) $(BUILDDIR):
	mkdir -p $@

# compile .c -> .o with dep generation
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) -MMD -MP -c $< -o $@

# static library
$(LIB): $(OBJS) | $(BUILDDIR)
	ar rcs $@ $^

# NOTE: test/bench_conv2d.c also exists — matches this pattern but is a bench,
# not a test. The bench/ dir has its own copy used by the bench pattern rule.
# test targets: each test compiles against the lib
test: $(TEST_BINS)
	@echo "--- Running tests ---"
	@for t in $(TEST_BINS); do echo "  $$t:"; $$t && echo "    PASS" || echo "    FAIL"; done

$(TEST_OBJDIR)/%: $(TESTDIR)/%.c $(LIB) | $(TEST_OBJDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: $(BUILDDIR)/main $(LIB)
	./$(BUILDDIR)/main

run_lm: $(BUILDDIR)/main_lm $(LIB)
	./$(BUILDDIR)/main_lm

# phony shortcuts: make main  →  build + run
main: $(BUILDDIR)/main
	$(BUILDDIR)/main

main_lm: $(BUILDDIR)/main_lm
	$(BUILDDIR)/main_lm

main_prep_data: $(BUILDDIR)/main_prep_data
	$(BUILDDIR)/main_prep_data

$(BUILDDIR)/main: main.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(BUILDDIR)/main_prep_data: main_prep_data.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(BUILDDIR)/main_lm: main_lm.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

# pattern rule for building bench binaries
$(BUILDDIR)/bench_%: $(BENCHDIR)/bench_%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMPFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

# individual bench: build + run
bench_conv2d: $(BUILDDIR)/bench_conv2d
	$(BUILDDIR)/bench_conv2d

bench_matmul: $(BUILDDIR)/bench_matmul
	$(BUILDDIR)/bench_matmul

bench_ops: $(BUILDDIR)/bench_ops
	$(BUILDDIR)/bench_ops

bench_multihead: $(BUILDDIR)/bench_multihead
	$(BUILDDIR)/bench_multihead

# build all benches then run each
bench_all: $(BENCH_BINS)
	@echo "=== Benchmarks ==="
	@for b in $(BENCH_BINS); do echo "  $$b:"; $$b; done

clean:
	rm -rf $(OBJDIR) $(BUILDDIR) $(LIB)

# include auto-generated deps
-include $(DEPS)
