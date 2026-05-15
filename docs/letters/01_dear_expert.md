# Letter to Expert — Makefile audit

**From:** dnn.c maintainer
**Tags:** `makefile` `c-build-system` `compiler-flags` `project-organization` `toolchain`

---

## Request

Improve the Makefile so that:

1. **Intermediate object files** (`.o`, `.d`) land in `obj/`
2. **Executables** (test bins, bench bins, main entrypoints) land in `build/`
3. Clarify whether `-O3` and `-ffast-math` are currently active

---

## Current state audit

### What works

| Concern | Status | Detail |
|---|---|---|
| `src/*.c` → `obj/*.o` | ✅ | `$(OBJDIR)/%.o: $(SRCDIR)/%.c` pattern works |
| `libdnn.a` in root | ✅ | `ar rcs $@ $^` at project root |
| `main` → `build/main` | ✅ | Explicit rule, `$(BUILDDIR)/main: main.c` |
| `main_lm` → `build/main_lm` | ✅ | Explicit rule |
| `main_prep_data` → `build/main_prep_data` | ✅ | Explicit rule |

### What doesn't

| Concern | Status | Detail |
|---|---|---|
| Test binaries → `build/` | ❌ | `TEST_OBJDIR = obj/test` — tests land in `obj/test/` instead of `build/test/` |
| Bench binaries → `build/` | ❌ | All `bench_*` targets compile to `$(OBJDIR)/bench_*` i.e. `obj/bench_conv2d` etc. |
| Twin `BUILDDIR` assignment | ⚠️ | `BUILDDIR = build` set twice (lines ~10 and ~30), no effect but confusing |
| `bench_ops.c` missing | ❌ | Makefile references `bench_ops.c` but no such file exists in project |
| `bench_all` manual loop duplicates compile | ⚠️ | `bench_all` recompiles all src/ manually instead of reusing `$(OBJS)` |

### Compiler flags

```makefile
CFLAGS ?= -Wall -Wextra -pedantic -std=c11 -O3 -ffast-math -g -DACCELERATE_NEW_LAPACK
```

- **`-O3` is active** — yes, in the default `CFLAGS`
- **`-ffast-math` is active** — yes, in the default `CFLAGS`
- **Risk:** `?=` means export `CFLAGS=...` on the command line silently drops both. In practice the Makefile always uses `$(CFLAGS)` so if the user invokes `make CFLAGS="-O0 -g"`, they lose `-ffast-math` and `-DACCELERATE_NEW_LAPACK` too.

### Summary of gaps

1. **Test/bench executables** not under `build/` — only `main`, `main_lm`, `main_prep_data` go there.
2. **Missing source** `bench_ops.c` — any `make bench_all` or `make bench_ops` will fail.
3. **`?=` fragility** — override-friendly but silently drops critical flags.
4. **`bench_all`** has a brittle manual compile loop that duplicates the pattern rule.

## Suggested direction

- Move `TEST_OBJDIR` under `build/` (e.g. `build/test/` or build directly as `$(BUILDDIR)/test_%`).
- Move bench targets to `$(BUILDDIR)/bench_%`.
- Consider `CFLAGS :=` with `?=` guard only for user additions (append via `override` or separate `EXTRA_CFLAGS`).
- Delete or implement `bench_ops.c`.

---

*Filed for expert review.*
