# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Run all PyTorch reference tests and compare patterns against C impl.

Usage:
    uv run test/ref_run_all.py              # run all torch refs
    uv run test/ref_run_all.py --diff       # show C vs torch value diffs
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

TEST_DIR = Path(__file__).parent
SCRIPTS = [
    "ref_ops",
    "ref_activations",
    "ref_reductions",
    "ref_nn",
    "ref_autograd",
]


def run_script(name):
    path = TEST_DIR / f"{name}.py"
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        ["uv", "run", str(path)],
        capture_output=False,
        text=True,
    )
    return result.returncode


def run_c_tests():
    print(f"\n{'='*60}")
    print(f"  C tests (make test)")
    print(f"{'='*60}")
    result = subprocess.run(
        ["make", "test"],
        capture_output=False,
        text=True,
        cwd=TEST_DIR.parent,
    )
    return result.returncode


if __name__ == "__main__":
    diff_mode = "--diff" in sys.argv

    print("dnn.c — PyTorch Reference Comparison")
    print("=" * 60)

    failed = 0
    for script in SCRIPTS:
        rc = run_script(script)
        if rc != 0:
            print(f"  FAILED: {script}")
            failed += 1

    print(f"\n{'='*60}")
    if failed:
        print(f"  {failed} script(s) failed")
    else:
        print("  All PyTorch refs passed")

    if diff_mode:
        rc = run_c_tests()
        if rc != 0:
            print("  C tests failed!")
        else:
            print("  C tests passed — compare outputs above")
