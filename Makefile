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

.PHONY: all clean test run

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
	$(CC) $(CFLAGS) $(INCDIRS) $< -L. -ldnn -o $@

run: main $(LIB)
	./main

main: main.c $(LIB)
	$(CC) $(CFLAGS) $(INCDIRS) $< -L. -ldnn -o $@

clean:
	rm -rf $(OBJDIR) $(LIB) main

# include auto-generated deps
-include $(DEPS)
