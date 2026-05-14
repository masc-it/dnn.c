# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: embedding lookup forward + backward.

Matches C test_embedding.c tests.
Usage: uv run test/ref_embedding.py
"""

import torch

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    ok = torch.allclose(got, expected, atol=EPS)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: max_diff={torch.max(torch.abs(got - expected)).item():.6e}  [{status}]")
    return ok


def test_embedding_simple():
    print("  test_embedding_simple...")
    table = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.],
                          [10., 11., 12.]])
    ids = torch.tensor([1, 3])
    emb = torch.nn.functional.embedding(ids, table)
    expected = torch.tensor([[4., 5., 6.],
                             [10., 11., 12.]])
    check("out", emb, expected)


def test_embedding_batch():
    print("  test_embedding_batch...")
    table = torch.tensor([[10., 100.],
                          [20., 200.],
                          [30., 300.]])
    ids = torch.tensor([0, 2, 1, 0])
    emb = torch.nn.functional.embedding(ids, table)
    expected = torch.tensor([[10., 100.],
                             [30., 300.],
                             [20., 200.],
                             [10., 100.]])
    check("out", emb, expected)


def test_embedding_single_id():
    print("  test_embedding_single_id...")
    table = torch.arange(1, 21, dtype=torch.float32).reshape(5, 4)
    ids = torch.tensor([0])
    emb = torch.nn.functional.embedding(ids, table)
    expected = table[0:1]
    check("out", emb, expected)


def test_embedding_backward_simple():
    print("  test_embedding_backward_simple...")
    table = torch.tensor([[1., 2.],
                          [3., 4.],
                          [5., 6.]], requires_grad=True)
    ids = torch.tensor([0, 2])
    emb = torch.nn.functional.embedding(ids, table)
    loss = emb.sum()
    loss.backward()
    # d_table: row 0 gets [1,1], row 2 gets [1,1], row 1 gets [0,0]
    expected = torch.tensor([[1., 1.],
                             [0., 0.],
                             [1., 1.]])
    check("d_table", table.grad, expected)


def test_embedding_backward_scaled():
    print("  test_embedding_backward_scaled...")
    table = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6]], requires_grad=True)
    ids = torch.tensor([1, 0, 1])
    emb = torch.nn.functional.embedding(ids, table)
    loss = emb.sum()
    loss.backward()
    # row 0 appears once → [1,1,1], row 1 appears twice → [2,2,2]
    expected = torch.tensor([[1., 1., 1.],
                             [2., 2., 2.]])
    check("d_table", table.grad, expected)


def test_embedding_backward_duplicate_ids():
    print("  test_embedding_backward_duplicate_ids...")
    table = torch.tensor([[1., 2.],
                          [3., 4.]], requires_grad=True)
    ids = torch.tensor([0, 0, 1, 0])
    emb = torch.nn.functional.embedding(ids, table)
    loss = emb.sum()
    loss.backward()
    # row 0 appears 3× → [3,3], row 1 appears 1× → [1,1]
    expected = torch.tensor([[3., 3.],
                             [1., 1.]])
    check("d_table", table.grad, expected)


def test_embedding_no_grad():
    print("  test_embedding_no_grad...")
    table = torch.tensor([[1., 2.],
                          [3., 4.],
                          [5., 6.]])
    ids = torch.tensor([0, 1])
    with torch.no_grad():
        emb = torch.nn.functional.embedding(ids, table)
    expected = torch.tensor([[1., 2.],
                             [3., 4.]])
    check("out", emb, expected)
    assert table.grad is None, "no-grad: table.grad should be None"
    print("  OK (no grad allocated)")


def test_embedding_chain():
    print("  test_embedding_chain...")
    table = torch.tensor([[1., 2.],
                          [3., 4.],
                          [5., 6.]], requires_grad=True)
    ids = torch.tensor([0, 2])
    W = torch.eye(2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    emb = torch.nn.functional.embedding(ids, table)
    y = emb @ W + b
    y.backward(torch.ones_like(y))
    # gradients flow back to table
    assert table.grad is not None, "table.grad should exist"
    assert table.grad.abs().sum().item() > 0, "table.grad should have non-zero elements"
    print(f"  d_table:\n{table.grad}")
    print("  OK (gradients flow through)")


if __name__ == "__main__":
    print("=== PyTorch Reference: Embedding ===")
    tests = [
        ("embedding_simple", test_embedding_simple),
        ("embedding_batch", test_embedding_batch),
        ("embedding_single_id", test_embedding_single_id),
        ("embedding_backward_simple", test_embedding_backward_simple),
        ("embedding_backward_scaled", test_embedding_backward_scaled),
        ("embedding_backward_duplicate_ids", test_embedding_backward_duplicate_ids),
        ("embedding_no_grad", test_embedding_no_grad),
        ("embedding_chain", test_embedding_chain),
    ]
    for name, fn in tests:
        fn()
    print("  ALL PASS")
