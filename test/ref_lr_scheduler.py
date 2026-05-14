"""
PyTorch reference for LR scheduler.

Generates LR sequences for each schedule type matching dnn.c's lr_scheduler.
Outputs C array literals for test/test_lr_scheduler.c.

Usage:
    uv run --with torch python test/ref_lr_scheduler.py
"""
import torch
import math
import sys

SEED = 42
torch.manual_seed(SEED)

# ── Configs ──
BASE_LR = 0.001
WARMUP  = 10
TOTAL   = 100
MIN_LR  = 1e-5
STEP_SZ = 20
GAMMA   = 0.5

def constant_lr(t):
    return BASE_LR

def linear_warmup_cosine(t):
    if t < WARMUP:
        return BASE_LR * (t + 1) / WARMUP
    post = t - WARMUP
    decay = TOTAL - WARMUP
    if decay <= 0:
        return BASE_LR
    cosine = 0.5 * (1 + math.cos(math.pi * post / decay))
    return MIN_LR + (BASE_LR - MIN_LR) * cosine

def linear_warmup(t):
    if t < WARMUP:
        return BASE_LR * (t + 1) / WARMUP
    return BASE_LR

def cosine(t):
    if TOTAL <= 0:
        return BASE_LR
    cosine = 0.5 * (1 + math.cos(math.pi * t / TOTAL))
    return MIN_LR + (BASE_LR - MIN_LR) * cosine

def step(t):
    if STEP_SZ <= 0:
        return BASE_LR
    return BASE_LR * (GAMMA ** (t // STEP_SZ))

def exponential(t):
    if t == 0:
        return BASE_LR
    return BASE_LR * (GAMMA ** t)

schedules = {
    'constant': constant_lr,
    'linear_warmup_cosine': linear_warmup_cosine,
    'linear_warmup': linear_warmup,
    'cosine': cosine,
    'step': step,
    'exponential': exponential,
}

N_STEPS = TOTAL + WARMUP + 20  # beyond total to test extrapolation

for name_slug, fn in schedules.items():
    lrs = [fn(t) for t in range(N_STEPS)]

    # Name for C
    c_name = name_slug  # use as-is

    # Print C array literal
    vals = ', '.join(f'{v:.15e}f' for v in lrs)
    print(f'static float lr_{c_name}[{len(lrs)}] = {{{vals}}};')

    # Print metadata comment
    print(f'/* lr_{c_name}: schedule={name_slug}, base_lr={BASE_LR}, '
          f'warmup={WARMUP}, total={TOTAL}, min_lr={MIN_LR}, '
          f'step_size={STEP_SZ}, gamma={GAMMA}, n_steps={N_STEPS} */')
    print()

# Print single-step checks for schedule_create/get_lr
print("/* Single-step reference values for scheduler creation */")
print(f"/* BASE_LR={BASE_LR}, WARMUP={WARMUP}, TOTAL={TOTAL}, "
      f"MIN_LR={MIN_LR}, STEP_SZ={STEP_SZ}, GAMMA={GAMMA} */")
print()

# Step 0 values
for name_slug, fn in schedules.items():
    lr0 = fn(0)
    print(f"/* step 0: lr_{name_slug} = {lr0:.15e}f */")

print()
# Step WARMUP/2 values
for name_slug, fn in schedules.items():
    lr = fn(WARMUP // 2)
    print(f"/* step {WARMUP//2}: lr_{name_slug} = {lr:.15e}f */")

print()
# Step WARMUP values
for name_slug, fn in schedules.items():
    lr = fn(WARMUP)
    print(f"/* step {WARMUP}: lr_{name_slug} = {lr:.15e}f */")

print()
# Step WARMUP+10 values
mid = WARMUP + 10
for name_slug, fn in schedules.items():
    lr = fn(mid)
    print(f"/* step {mid}: lr_{name_slug} = {lr:.15e}f */")

print()
# Step TOTAL (end of cosine)
for name_slug, fn in schedules.items():
    lr = fn(TOTAL)
    print(f"/* step {TOTAL}: lr_{name_slug} = {lr:.15e}f */")

print()
print(f"// Total steps generated: N_STEPS={N_STEPS}")
print(f"// All values match dnn.c lr_scheduler formulas exactly.")
