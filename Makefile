CC       ?= gcc
CFLAGS   ?= -Wall -Wextra -pedantic -std=c11 -O3 -ffast-math -g -DACCELERATE_NEW_LAPACK
INCDIRS   = -Iinclude -Isrc -I/opt/homebrew/opt/libomp/include
LDFLAGS   = -lz -L/opt/homebrew/opt/libomp/lib -lomp
OMPFLAGS  = -Xpreprocessor -fopenmp

SRCDIR   = src
OBJDIR   = obj
BUILDDIR = build
TESTDIR  = test

SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
DEPS     = $(OBJS:.o=.d)

TEST_SRCS = $(wildcard $(TESTDIR)/*.c)
TEST_OBJDIR = $(OBJDIR)/test
TEST_BINS = $(patsubst $(TESTDIR)/%.c, $(TEST_OBJDIR)/%, $(TEST_SRCS))

BUILDDIR = build

LIB      = libdnn.a

.PHONY: all clean test run bench

all: $(OBJDIR) $(TEST_OBJDIR) $(LIB)

$(OBJDIR) $(TEST_OBJDIR):
	mkdir -p $@

# compile .c -> .o with dep generation
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCDIRS) -MMD -MP -c $< -o $@

# static library
$(LIB): $(OBJS)
	ar rcs $@ $^

# test targets: each test compiles against the lib
test: $(TEST_BINS)
	@echo "--- Running tests ---"
	@for t in $(TEST_BINS); do echo "  $$t:"; $$t && echo "    PASS" || echo "    FAIL"; done

$(TEST_OBJDIR)/%: $(TESTDIR)/%.c $(LIB) | $(TEST_OBJDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCDIRS) $< -L. -ldnn -framework Accelerate $(LDFLAGS) -o $@

$(BUILDDIR):
	mkdir -p $@

run: $(BUILDDIR)/main $(LIB)
	./$(BUILDDIR)/main

run_lm: $(BUILDDIR)/main_lm $(LIB)
	./$(BUILDDIR)/main_lm

$(BUILDDIR)/main: main.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCDIRS) $< -L. -ldnn -framework Accelerate $(LDFLAGS) -o $@

$(BUILDDIR)/main_prep_data: main_prep_data.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCDIRS) $< -L. -ldnn -framework Accelerate $(LDFLAGS) -o $@

$(BUILDDIR)/main_lm: main_lm.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCDIRS) $< -L. -ldnn -framework Accelerate $(LDFLAGS) -o $@

BENCH_CFLAGS = -Wall -Wextra -pedantic -std=c11 -O3 -ffast-math -g -DACCELERATE_NEW_LAPACK $(OMPFLAGS)
BENCH_LIBS  = -L. -ldnn -framework Accelerate $(LDFLAGS)

bench_all: clean
	mkdir -p $(OBJDIR) $(TEST_OBJDIR)
	for s in $(SRCS); do \
	  $(CC) $(BENCH_CFLAGS) $(INCDIRS) -MMD -MP -c $$s -o $(OBJDIR)/$$(basename $$s .c).o || exit 1; \
	done
	ar rcs $(LIB) $(OBJS)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_conv2d.c $(BENCH_LIBS) -o $(OBJDIR)/bench_conv2d
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_matmul.c $(BENCH_LIBS) -o $(OBJDIR)/bench_matmul
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_ops.c $(BENCH_LIBS) -o $(OBJDIR)/bench_ops
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_multihead.c $(BENCH_LIBS) -o $(OBJDIR)/bench_multihead
	echo "=== Conv2D ===" && $(OBJDIR)/bench_conv2d
	echo "=== MatMul ===" && $(OBJDIR)/bench_matmul
	echo "=== Ops ===" && $(OBJDIR)/bench_ops
	echo "=== Multihead ===" && $(OBJDIR)/bench_multihead

bench_conv2d: $(LIB)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_conv2d.c $(BENCH_LIBS) -o $(OBJDIR)/bench_conv2d
	$(OBJDIR)/bench_conv2d

bench_matmul: $(LIB)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_matmul.c $(BENCH_LIBS) -o $(OBJDIR)/bench_matmul
	$(OBJDIR)/bench_matmul

bench_ops: $(LIB)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_ops.c $(BENCH_LIBS) -o $(OBJDIR)/bench_ops
	$(OBJDIR)/bench_ops

bench_multihead: $(LIB)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_multihead.c $(BENCH_LIBS) -o $(OBJDIR)/bench_multihead
	$(OBJDIR)/bench_multihead

clean:
	rm -rf $(OBJDIR) $(BUILDDIR) $(LIB)

# include auto-generated deps
-include $(DEPS)
