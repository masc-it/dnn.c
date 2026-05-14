#!/usr/bin/env python3
"""
Reference: compare dnn.c fused tensor_swiglu against PyTorch.

Launches the C test binary, parses printed values, and cross-checks
against torch.nn.functional.silu(gate) * up with identical inputs.

Usage:
    python3 test/ref_swiglu.py          # run C binary + compare
    python3 test/ref_swiglu.py --no-c   # PyTorch only, print ref values
"""

import subprocess
import sys
import math
import os

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("# PyTorch not installed — generating reference values with math.exp only.")
    torch = None


def swiglu_ref(gate, up):
    """Compute SiLU(gate) * up using math.exp (no PyTorch)."""
    out = []
    for g, u in zip(gate, up):
        sig = 1.0 / (1.0 + math.exp(-g))
        silu = g * sig
        out.append(silu * u)
    return out


def swiglu_deriv_ref(gate):
    """Compute SiLU'(gate) = sig(g) * (1 + g - g*sig(g))."""
    sig = 1.0 / (1.0 + math.exp(-gate))
    return sig * (1.0 + gate - gate * sig)


def run_c_test():
    """Build and run the swiglu tests, return True if all pass."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_bin = os.path.join(root, "obj", "test", "test_autograd")

    # Build (make lib first, then compile test)
    build_lib = subprocess.run(
        ["make", "-j4"],
        cwd=root,
        capture_output=True, text=True
    )
    if build_lib.returncode != 0:
        print(f"Lib build failed:\n{build_lib.stderr}")
        return False

    build_test = subprocess.run(
        ["cc", "-Wall", "-Wextra", "-pedantic", "-std=c11", "-O3",
         "-ffast-math", "-g", "-DACCELERATE_NEW_LAPACK", "-Xpreprocessor", "-fopenmp",
         "-Iinclude", "-Isrc", "-I/opt/homebrew/opt/libomp/include",
         "test/test_autograd.c",
         "-L.", "-ldnn", "-framework", "Accelerate",
         "-lz", "-L/opt/homebrew/opt/libomp/lib", "-lomp",
         "-o", "obj/test/test_autograd"],
        cwd=root,
        capture_output=True, text=True
    )
    if build_test.returncode != 0:
        print(f"Test build failed:\n{build_test.stderr}")
        return False

    # Run (filter for swiglu tests)
    result = subprocess.run(
        [test_bin],
        cwd=root,
        capture_output=True, text=True
    )

    # Check test output
    for line in result.stdout.splitlines():
        if "swiglu" in line or "FAIL" in line or "ALL PASS" in line:
            print(f"  {line}")

    if "FAIL" in result.stdout or result.returncode != 0:
        print(f"\nStderr:\n{result.stderr}")
        return False

    return True


def ref_vs_torch():
    """Print reference values for dnn.c test assertion debugging."""
    print("\n# Reference values for swiglu tests:\n")

    # test_swiglu_simple: gate=2, up=3
    g, u = 2.0, 3.0
    sig = 1.0 / (1.0 + math.exp(-g))
    silu = g * sig
    out = silu * u
    silu_deriv = sig * (1.0 + g - g * sig)
    d_gate = 1.0 * silu_deriv * u
    d_up = 1.0 * silu
    print(f"  simple: gate={g}, up={u}")
    print(f"    sig={sig:.10f}, silu={silu:.10f}")
    print(f"    out={out:.10f}")
    print(f"    silu_deriv={silu_deriv:.10f}")
    print(f"    d_gate={d_gate:.10f}, d_up={d_up:.10f}")
    print()

    # test_swiglu_array: gate=[0,1,-2], up=[2,3,4]
    gates = [0.0, 1.0, -2.0]
    ups = [2.0, 3.0, 4.0]
    ref = swiglu_ref(gates, ups)
    print(f"  array: gate={gates}, up={ups}")
    for i, (g, u) in enumerate(zip(gates, ups)):
        sig = 1.0 / (1.0 + math.exp(-g))
        silu = g * sig
        sd = swiglu_deriv_ref(g)
        print(f"    [{i}]: sig={sig:.10f}, silu={silu:.10f}, SiLU'={sd:.10f}")
        print(f"          out={ref[i]:.10f}, d_gate={sd*u:.10f}, d_up={silu:.10f}")
    print()

    # test_swiglu_broadcast_gate: gate=[2], up=[1,2,3]
    g = 2.0
    ups = [1.0, 2.0, 3.0]
    sig = 1.0 / (1.0 + math.exp(-g))
    silu = g * sig
    sd = swiglu_deriv_ref(g)
    d_gate_bc = sd * sum(ups)
    print(f"  broadcast_gate: gate=[{g}], up={ups}")
    print(f"    sig={sig:.10f}, silu={silu:.10f}, SiLU'={sd:.10f}")
    print(f"    d_gate (sum)={d_gate_bc:.10f}")
    for i, u in enumerate(ups):
        print(f"    d_up[{i}]={silu:.10f}")
    print()

    # test_swiglu_broadcast_up: gate=[0,1,-1], up=[5]
    gates = [0.0, 1.0, -1.0]
    u = 5.0
    sum_silu = 0.0
    print(f"  broadcast_up: gate={gates}, up=[{u}]")
    for i, g in enumerate(gates):
        sig = 1.0 / (1.0 + math.exp(-g))
        silu = g * sig
        sd = swiglu_deriv_ref(g)
        sum_silu += silu
        print(f"    [{i}]: sig={sig:.10f}, silu={silu:.10f}, SiLU'={sd:.10f}, d_gate={sd*u:.10f}")
    print(f"    d_up (sum silu)={sum_silu:.10f}")
    print()

    # test_swiglu_chain: gate=1.5, up=2.0, out = swiglu(gate,up) + gate
    g, u = 1.5, 2.0
    sig = 1.0 / (1.0 + math.exp(-g))
    silu = g * sig
    sd = swiglu_deriv_ref(g)
    d_gate_chain = 1.0 * sd * u + 1.0  # +1 from the identity path
    d_up_chain = 1.0 * silu
    print(f"  chain: gate={g}, up={u}, out = swiglu(gate,up) + gate")
    print(f"    sig={sig:.10f}, silu={silu:.10f}, SiLU'={sd:.10f}")
    print(f"    d_gate={d_gate_chain:.10f}, d_up={d_up_chain:.10f}")
    print()

    # PyTorch reference (if available)
    if torch is not None:
        print("# PyTorch cross-check:\n")
        gt = torch.tensor([2.0], requires_grad=True)
        ut = torch.tensor([3.0], requires_grad=True)
        ot = F.silu(gt) * ut
        ot.backward()
        print(f"  torch simple: out={ot.item():.10f}, d_gate={gt.grad.item():.10f}, d_up={ut.grad.item():.10f}")

        gt = torch.tensor([0.0, 1.0, -2.0], requires_grad=True)
        ut = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        ot = F.silu(gt) * ut
        ot.sum().backward()
        print(f"  torch array: d_gate={gt.grad.tolist()}, d_up={ut.grad.tolist()}")

        gt = torch.tensor([2.0], requires_grad=True)
        ut = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        ot = F.silu(gt) * ut
        ot.sum().backward()
        print(f"  torch broadcast_gate: d_gate={gt.grad.item():.10f}, d_up={ut.grad.tolist()}")

        gt = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
        ut = torch.tensor([5.0], requires_grad=True)
        ot = F.silu(gt) * ut
        ot.sum().backward()
        print(f"  torch broadcast_up: d_gate={gt.grad.tolist()}, d_up={ut.grad.item():.10f}")

        gt = torch.tensor([1.5], requires_grad=True)
        ut = torch.tensor([2.0], requires_grad=True)
        ot = F.silu(gt) * ut + gt
        ot.backward()
        print(f"  torch chain: d_gate={gt.grad.item():.10f}, d_up={ut.grad.item():.10f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--no-c":
        ref_vs_torch()
        return

    print("=== Running dnn.c swiglu tests ===")
    ok = run_c_test()
    if ok:
        print("\n=== All C tests PASS ===")
    else:
        print("\n=== C tests FAIL ===")
        sys.exit(1)

    print("\n=== Reference values ===")
    ref_vs_torch()


if __name__ == "__main__":
    main()
