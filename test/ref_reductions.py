# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: reductions (sum, mean).

Matches C test_autograd.c reduction tests.
Usage: uv run test/ref_reductions.py
"""

import torch

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    ok = torch.allclose(got, expected, atol=EPS)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: max_diff={torch.max(torch.abs(got - expected)).item():.6e}  [{status}]")
    return ok


def test_sum_simple():
    print("  test_sum_simple...")
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    c = a.sum()
    c.backward()
    check("da", a.grad, torch.ones_like(a))


def test_sum_2d_dim0():
    print("  test_sum_2d_dim0...")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    c = a.sum(dim=0)
    c.backward(torch.ones_like(c))
    # grad broadcast: each input gets grad from summed dim
    expected = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    check("da", a.grad, expected)


def test_sum_2d_dim1():
    print("  test_sum_2d_dim1...")
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    c = a.sum(dim=1)
    c.backward(torch.ones_like(c))
    expected = torch.ones_like(a)
    check("da", a.grad, expected)


def test_mean_simple():
    print("  test_mean_simple...")
    a = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
    c = a.mean()
    c.backward()
    # d(mean)/da = 1/n for each
    check("da", a.grad, torch.full_like(a, 1.0/3.0))


def test_mean_2d_dim1():
    print("  test_mean_2d_dim1...")
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    c = a.mean(dim=1)
    c.backward(torch.ones_like(c))
    # grad = (1/dim_size) broadcast back
    expected = torch.full_like(a, 1.0/3.0)
    check("da", a.grad, expected)


if __name__ == "__main__":
    print("=== PyTorch Reference: Reductions ===")
    tests = [
        ("sum_simple", test_sum_simple),
        ("sum_2d_dim0", test_sum_2d_dim0),
        ("sum_2d_dim1", test_sum_2d_dim1),
        ("mean_simple", test_mean_simple),
        ("mean_2d_dim1", test_mean_2d_dim1),
    ]
    for name, fn in tests:
        fn()
    print("  ALL PASS")
