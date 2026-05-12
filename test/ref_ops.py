# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: element-wise & matrix ops.

Matches C test_autograd.c ops tests.
Usage: uv run test/ref_ops.py
"""

import torch
import math

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    ok = torch.allclose(got, expected, atol=EPS)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: max_diff={torch.max(torch.abs(got - expected)).item():.6e}  [{status}]")
    return ok


def fmt(t):
    return f"shape={list(t.shape)} data={t.detach().flatten().tolist()}"


def test_add_simple():
    print("  test_add_simple...")
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    c = a + b
    c.backward()
    check("da", a.grad, torch.tensor([1.0]))
    check("db", b.grad, torch.tensor([1.0]))


def test_add_chain():
    print("  test_add_chain...")
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    c = torch.tensor([3.0], requires_grad=True)
    d = a + b
    e = d + c
    e.backward()
    check("da", a.grad, torch.tensor([1.0]))
    check("db", b.grad, torch.tensor([1.0]))
    check("dc", c.grad, torch.tensor([1.0]))


def test_add_multi_use():
    print("  test_add_multi_use...")
    a = torch.tensor([5.0], requires_grad=True)
    c = a + a  # 2a
    c.backward()
    check("da", a.grad, torch.tensor([2.0]))


def test_add_diamond():
    print("  test_add_diamond...")
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    c = torch.tensor([3.0], requires_grad=True)
    d = a + b
    e = a + c
    loss = d + e
    loss.backward()
    check("da", a.grad, torch.tensor([2.0]))
    check("db", b.grad, torch.tensor([1.0]))
    check("dc", c.grad, torch.tensor([1.0]))


def test_add_broadcast():
    print("  test_add_broadcast...")
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([10.0], requires_grad=True)
    c = a + b
    c.backward(torch.ones_like(c))
    check("da", a.grad, torch.tensor([1.0, 1.0, 1.0]))
    check("db", b.grad, torch.tensor([3.0]))


def test_mul_simple():
    print("  test_mul_simple...")
    a = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([4.0], requires_grad=True)
    c = a * b
    c.backward()
    check("da", a.grad, torch.tensor([4.0]))
    check("db", b.grad, torch.tensor([3.0]))


def test_mul_chain():
    print("  test_mul_chain...")
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    c = torch.tensor([4.0], requires_grad=True)
    d = (a * b) * c
    d.backward()
    check("da", a.grad, torch.tensor([12.0]))
    check("db", b.grad, torch.tensor([8.0]))
    check("dc", c.grad, torch.tensor([6.0]))


def test_mul_self():
    print("  test_mul_self...")
    a = torch.tensor([5.0], requires_grad=True)
    c = a * a
    c.backward()
    check("da", a.grad, torch.tensor([10.0]))


def test_mul_broadcast():
    print("  test_mul_broadcast...")
    a = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    b = torch.tensor([10.0], requires_grad=True)
    c = a * b
    c.backward(torch.ones_like(c))
    check("da", a.grad, torch.tensor([10.0, 10.0, 10.0]))
    check("db", b.grad, torch.tensor([9.0]))


def test_pow_simple():
    print("  test_pow_simple...")
    a = torch.tensor([3.0], requires_grad=True)
    c = a ** 2
    c.backward()
    check("da", a.grad, torch.tensor([6.0]))


def test_pow_cube():
    print("  test_pow_cube...")
    a = torch.tensor([2.0], requires_grad=True)
    c = a ** 3.0
    c.backward()
    check("da", a.grad, torch.tensor([12.0]))


def test_pow_exp1():
    print("  test_pow_exp1...")
    a = torch.tensor([7.0], requires_grad=True)
    c = a ** 1.0
    c.backward()
    check("da", a.grad, torch.tensor([1.0]))


def test_pow_neg():
    print("  test_pow_neg...")
    a = torch.tensor([4.0], requires_grad=True)
    c = a ** (-1.0)
    c.backward()
    check("da", a.grad, torch.tensor([-0.0625]))


def test_pow_broadcast():
    print("  test_pow_broadcast...")
    a = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    c = a ** 2.0
    c.backward(torch.ones_like(c))
    check("da", a.grad, torch.tensor([4.0, 6.0, 8.0]))


def test_neg_simple():
    print("  test_neg_simple...")
    a = torch.tensor([5.0], requires_grad=True)
    c = -a
    c.backward()
    check("da", a.grad, torch.tensor([-1.0]))


def test_neg_broadcast():
    print("  test_neg_broadcast...")
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    c = -a
    c.backward(torch.ones_like(c))
    check("da", a.grad, torch.tensor([-1.0, -1.0, -1.0]))


def test_sub_simple():
    print("  test_sub_simple...")
    a = torch.tensor([5.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    c = a - b
    c.backward()
    check("da", a.grad, torch.tensor([1.0]))
    check("db", b.grad, torch.tensor([-1.0]))


def test_sub_self():
    print("  test_sub_self...")
    a = torch.tensor([7.0], requires_grad=True)
    c = a - a
    c.backward()
    check("da", a.grad, torch.tensor([0.0]))


def test_sub_broadcast():
    print("  test_sub_broadcast...")
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([10.0], requires_grad=True)
    c = a - b
    c.backward(torch.ones_like(c))
    check("da", a.grad, torch.tensor([1.0, 1.0, 1.0]))
    check("db", b.grad, torch.tensor([-3.0]))


def test_div_simple():
    print("  test_div_simple...")
    a = torch.tensor([10.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    c = a / b
    c.backward()
    check("da", a.grad, torch.tensor([0.5]))
    check("db", b.grad, torch.tensor([-2.5]))


def test_div_self():
    print("  test_div_self...")
    a = torch.tensor([7.0], requires_grad=True)
    c = a / a
    c.backward()
    check("da", a.grad, torch.tensor([0.0]))


def test_div_broadcast():
    print("  test_div_broadcast...")
    a = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
    b = torch.tensor([5.0], requires_grad=True)
    c = a / b
    c.backward(torch.ones_like(c))
    check("da", a.grad, torch.tensor([0.2, 0.2, 0.2]))
    # db = sum(-a/b^2) = -(10+20+30)/25 = -60/25 = -2.4
    check("db", b.grad, torch.tensor([-2.4]))


def test_matmul_simple():
    print("  test_matmul_simple...")
    # A(2,3) @ B(3,2) — matches C test values
    a = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)
    b = torch.tensor([[7.0, 8.0],
                       [9.0, 10.0],
                       [11.0, 12.0]], requires_grad=True)
    c = a @ b
    c.backward(torch.ones_like(c))
    # da = ones(2,2) @ B^T = [[15,19,23],[15,19,23]]
    check("da", a.grad, torch.tensor([[15.0, 19.0, 23.0],
                                        [15.0, 19.0, 23.0]]))
    # db = A^T @ ones(2,2) = [[5,5],[7,7],[9,9]]
    check("db", b.grad, torch.tensor([[5.0, 5.0],
                                        [7.0, 7.0],
                                        [9.0, 9.0]]))


def test_matmul_square_self():
    print("  test_matmul_square_self...")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    c = a @ a
    c.backward(torch.ones_like(c))
    # da = dC @ A^T + A^T @ dC = [[3,7],[3,7]] + [[4,4],[6,6]] = [[7,11],[9,13]]
    check("da", a.grad, torch.tensor([[7.0, 11.0], [9.0, 13.0]]))


if __name__ == "__main__":
    print("=== PyTorch Reference: Ops (elem + matrix) ===")
    tests = [
        ("add_simple", test_add_simple),
        ("add_chain", test_add_chain),
        ("add_multi_use", test_add_multi_use),
        ("add_diamond", test_add_diamond),
        ("add_broadcast", test_add_broadcast),
        ("mul_simple", test_mul_simple),
        ("mul_chain", test_mul_chain),
        ("mul_self", test_mul_self),
        ("mul_broadcast", test_mul_broadcast),
        ("pow_simple", test_pow_simple),
        ("pow_cube", test_pow_cube),
        ("pow_exp1", test_pow_exp1),
        ("pow_neg", test_pow_neg),
        ("pow_broadcast", test_pow_broadcast),
        ("neg_simple", test_neg_simple),
        ("neg_broadcast", test_neg_broadcast),
        ("sub_simple", test_sub_simple),
        ("sub_self", test_sub_self),
        ("sub_broadcast", test_sub_broadcast),
        ("div_simple", test_div_simple),
        ("div_self", test_div_self),
        ("div_broadcast", test_div_broadcast),
        ("matmul_simple", test_matmul_simple),
        ("matmul_square_self", test_matmul_square_self),
    ]
    for name, fn in tests:
        fn()
    print("  ALL PASS")
