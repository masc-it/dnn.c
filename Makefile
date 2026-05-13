CC       ?= gcc
CFLAGS   ?= -Wall -Wextra -pedantic -std=c11 -O3 -ffast-math -g -DACCELERATE_NEW_LAPACK
INCDIRS   = -Iinclude -Isrc -I/opt/homebrew/opt/libomp/include
LDFLAGS   = -lz -L/opt/homebrew/opt/libomp/lib -lomp
OMPFLAGS  = -Xpreprocessor -fopenmp

SRCDIR   = src
OBJDIR   = obj
TESTDIR  = test

SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
DEPS     = $(OBJS:.o=.d)

TEST_SRCS = $(wildcard $(TESTDIR)/*.c)
TEST_OBJDIR = $(OBJDIR)/test
TEST_BINS = $(patsubst $(TESTDIR)/%.c, $(TEST_OBJDIR)/%, $(TEST_SRCS))

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

run: main $(LIB)
	./main

main: main.c $(LIB)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCDIRS) $< -L. -ldnn -framework Accelerate $(LDFLAGS) -o $@

BENCH_CFLAGS = -Wall -Wextra -pedantic -std=c11 -O3 -ffast-math -g -DACCELERATE_NEW_LAPACK $(OMPFLAGS)
BENCH_LIBS  = -L. -ldnn -framework Accelerate $(LDFLAGS)

bench_all: clean
	mkdir -p $(OBJDIR) $(TEST_OBJDIR)
	for s in $(SRCS); do \
	  $(CC) $(BENCH_CFLAGS) $(INCDIRS) -MMD -MP -c $$s -o $(OBJDIR)/$$(basename $$s .c).o || exit 1; \
	done
	ar rcs $(LIB) $(OBJS)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_conv2d.c $(BENCH_LIBS) -o bench_conv2d
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_matmul.c $(BENCH_LIBS) -o bench_matmul
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_ops.c $(BENCH_LIBS) -o bench_ops
	echo "=== Conv2D ===" && ./bench_conv2d
	echo "=== MatMul ===" && ./bench_matmul
	echo "=== Ops ===" && ./bench_ops

bench_conv2d: $(LIB)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_conv2d.c $(BENCH_LIBS) -o bench_conv2d
	./bench_conv2d

bench_matmul: $(LIB)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_matmul.c $(BENCH_LIBS) -o bench_matmul
	./bench_matmul

bench_ops: $(LIB)
	$(CC) $(BENCH_CFLAGS) $(INCDIRS) bench_ops.c $(BENCH_LIBS) -o bench_ops
	./bench_ops

clean:
	rm -rf $(OBJDIR) $(LIB) main bench_conv2d bench_matmul bench_ops bench_{baseline,after}

# include auto-generated deps
-include $(DEPS)
