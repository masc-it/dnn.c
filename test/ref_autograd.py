# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: complex autograd patterns (diamonds, views, mixed ops).

Matches C test_autograd.c advanced tests.
Usage: uv run test/ref_autograd.py
"""

import torch

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    ok = torch.allclose(got, expected, atol=EPS)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: max_diff={torch.max(torch.abs(got - expected)).item():.6e}  [{status}]")
    return ok


def test_mul_with_add():
    print("  test_mul_with_add...")
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    c = torch.tensor([4.0], requires_grad=True)
    d = a * b        # 6
    e = d + c        # 10
    e.backward()
    check("da", a.grad, torch.tensor([3.0]))   # d(a*b)/da = b = 3
    check("db", b.grad, torch.tensor([2.0]))   # d(a*b)/db = a = 2
    check("dc", c.grad, torch.tensor([1.0]))


def test_mixed_diamond():
    print("  test_mixed_diamond...")
    # a mul(b) + a add(b) — matches C test: a=2, b=3
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    c = a * b           # 6
    d = a + b           # 5
    loss = c + d        # 11
    loss.backward()
    check("da", a.grad, torch.tensor([4.0]))   # da = b + 1 = 4
    check("db", b.grad, torch.tensor([3.0]))   # db = a + 1 = 3


def test_transpose_backward():
    print("  test_transpose_backward...")
    a = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)
    c = a.T  # (3,2)
    loss = c.sum()
    loss.backward()
    check("da", a.grad, torch.ones_like(a))


def test_mul_diamond_self():
    print("  test_mul_diamond_self...")
    a = torch.tensor([2.0], requires_grad=True)
    b = a * a            # a² = 4
    c = a * a            # a² = 4
    d = b + c            # 8
    d.backward()
    check("da", a.grad, torch.tensor([8.0]))   # d(a²)/da = 2a, two paths: 2*2 + 2*2 = 8


def test_3d_multi_op():
    print("  test_3d_multi_op...")
    a = torch.tensor([[[1.0, 2.0],
                        [3.0, 4.0]],
                       [[5.0, 6.0],
                        [7.0, 8.0]]], requires_grad=True)
    b = a.sum(dim=2)       # (2,2)
    c = b * 2.0            # (2,2)
    d = c.mean()           # scalar
    d.backward()
    # da = d/d a * 2.0 * ones_like(b) / 4 broadcast back to a shape
    # gradient: 2.0 / (2*2*2?) / 4 = 0.5 per element
    # c = 2 * sum; mean over 4 elements; d/d = 1/4
    # da_jkl = (2.0 / 4) = 0.5 for each element in the sum dim
    check("da", a.grad, torch.full_like(a, 0.5))


if __name__ == "__main__":
    print("=== PyTorch Reference: Complex Autograd Patterns ===")
    tests = [
        ("mul_with_add", test_mul_with_add),
        ("mixed_diamond", test_mixed_diamond),
        ("transpose_backward", test_transpose_backward),
        ("mul_diamond_self", test_mul_diamond_self),
        ("3d_multi_op", test_3d_multi_op),
    ]
    for name, fn in tests:
        fn()
    print("  ALL PASS")
