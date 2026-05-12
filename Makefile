CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -pedantic -std=c11 -O2 -g
INCDIRS  = -Iinclude
LDFLAGS  =

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
	$(CC) $(CFLAGS) $(INCDIRS) -MMD -MP -c $< -o $@

# static library
$(LIB): $(OBJS)
	ar rcs $@ $^

# test targets: each test compiles against the lib
test: $(TEST_BINS)
	@echo "--- Running tests ---"
	@for t in $(TEST_BINS); do echo "  $$t:"; $$t && echo "    PASS" || echo "    FAIL"; done

$(TEST_OBJDIR)/%: $(TESTDIR)/%.c $(LIB) | $(TEST_OBJDIR)
	$(CC) $(CFLAGS) $(INCDIRS) $< -L. -ldnn -framework Accelerate -o $@

run: main $(LIB)
	./main

main: main.c $(LIB)
	$(CC) $(CFLAGS) $(INCDIRS) $< -L. -ldnn -framework Accelerate -o $@

BENCH_FLAGS = -Wall -Wextra -pedantic -std=c11 -O3 -ffast-math -g -DACCELERATE_NEW_LAPACK
BENCH_LIBS  = -L. -ldnn -framework Accelerate

bench: clean
	mkdir -p $(OBJDIR) $(TEST_OBJDIR)
	for s in $(SRCS); do \
	  $(CC) $(BENCH_FLAGS) $(INCDIRS) -MMD -MP -c $$s -o $(OBJDIR)/$$(basename $$s .c).o || exit 1; \
	done
	ar rcs $(LIB) $(OBJS)
	$(CC) $(BENCH_FLAGS) $(INCDIRS) bench_conv2d.c $(BENCH_LIBS) -o bench_conv2d
	@echo "=== Conv2D benchmark (im2col + Accelerate BLAS) ==="
	./bench_conv2d | tee bench_conv2d_baseline.txt

clean:
	rm -rf $(OBJDIR) $(LIB) main bench_conv2d bench_conv2d_baseline.txt

# include auto-generated deps
-include $(DEPS)
