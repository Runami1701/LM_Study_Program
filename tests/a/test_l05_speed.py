"""Checks for a-l05 — Making the engine fast."""

import time

import numpy as np
import pytest

from tests.a.graph import run_backward
from tests.helpers import assert_close, load


@pytest.mark.fn("shared_weight_grad")
def test_values_match_batched_sum():
    ops = load("autograd.ops")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4, 5))
    g = rng.standard_normal((3, 4, 6))
    got = ops.shared_weight_grad(a, g)
    expected = sum(a[b].T @ g[b] for b in range(3))          # the sum over the batch, written out
    assert_close(got, expected, "shared_weight_grad", atol=1e-10,
                 hint="stack all rows: a reshaped to (-1, n), transposed, times g reshaped to (-1, p)")
    a4 = rng.standard_normal((2, 3, 4, 5))
    g4 = rng.standard_normal((2, 3, 4, 6))
    assert_close(ops.shared_weight_grad(a4, g4), np.einsum("xytn,xytp->np", a4, g4), "shared_weight_grad (two batch axes)",
                 atol=1e-10, hint="reshape(-1, n) stacks every leading axis, however many there are")


@pytest.mark.fn("shared_weight_grad")
def test_matmul_uses_it():
    ops = load("autograd.ops")
    T = load("autograd.tensor").Tensor
    calls = {"n": 0}
    original = ops.shared_weight_grad

    def counting(a, g):
        calls["n"] += 1
        return original(a, g)

    ops.shared_weight_grad = counting
    try:
        rng = np.random.default_rng(1)
        x = T(rng.standard_normal((2, 3, 4)))
        W = T(rng.standard_normal((4, 5)))
        out = ops.matmul(x, W)
        G = rng.standard_normal((2, 3, 5))
        run_backward(out, G)
    finally:
        ops.shared_weight_grad = original
    assert calls["n"] == 1, (
        "matmul's backward should call shared_weight_grad when a has batch axes and b is a 2-D weight"
    )
    assert_close(W.grad, np.einsum("btn,btp->np", x.data, G), "W.grad from matmul")


@pytest.mark.fn("Tensor")
def test_new_tensor_grad_is_cheap():
    T = load("autograd.tensor").Tensor
    data = np.zeros((32, 4, 128, 128), dtype=np.float32)    # 8 MB, the size of an attention matrix
    T(data)
    best = min(_time(lambda: T(data)) for _ in range(5))
    assert best < 0.002, (
        f"creating a Tensor of 8 MB takes {best * 1000:.1f} ms — allocate grad with np.zeros(shape, dtype) "
        "(memory the operating system zeroes lazily) instead of np.zeros_like (which writes every byte)"
    )


def _time(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0
