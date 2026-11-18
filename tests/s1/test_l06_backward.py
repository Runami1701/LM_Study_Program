"""Checks for s1-l06 — Backward through linear and tanh."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load, numerical_grad, rel_error

MOD = "stage1.layers"


@pytest.mark.fn("linear_backward")
def test_linear_backward_worked_example():
    m = load(MOD)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    W = np.array([[1.0, 0.0], [-1.0, 2.0]], dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    dout = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dx, dW, db = m.linear_backward(dout, (x, W, b))
    check_array(dW, "dW", shape=(2, 2), dtype=np.float32)
    assert_close(dW, [[1, 3], [2, 4]], "dW", hint="dW = x^T @ dout")
    assert_close(dx, [[1, -1], [0, 2]], "dx", hint="dx = dout @ W^T")
    assert_close(db, [1, 1], "db", hint="db = dout summed over the batch")


@pytest.mark.fn("linear_backward")
def test_linear_backward_shapes():
    m = load(MOD)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3)).astype(np.float32)
    W = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    dx, dW, db = m.linear_backward(np.ones((5, 4), np.float32), (x, W, b))
    check_array(dx, "dx", shape=(5, 3), dtype=np.float32, shape_desc="(B, n)")
    check_array(dW, "dW", shape=(3, 4), dtype=np.float32, shape_desc="(n, m)")
    check_array(db, "db", shape=(4,), dtype=np.float32, shape_desc="(m,)",
                hint="sum dout over the batch axis (axis=0)")


@pytest.mark.fn("linear_backward")
def test_linear_backward_gradcheck():
    m = load(MOD)
    rng = np.random.default_rng(1)
    x, W, b = rng.standard_normal((4, 3)), rng.standard_normal((3, 2)), rng.standard_normal(2)
    G = rng.standard_normal((4, 2))           # L = sum(G * out)  =>  dL/dout = G
    f = lambda: float(np.sum(G * (x @ W + b)))  # noqa: E731
    dx, dW, db = m.linear_backward(G, (x, W, b))
    for name, ana, arr in (("dx", dx, x), ("dW", dW, W), ("db", db, b)):
        err = rel_error(ana, numerical_grad(f, arr))
        assert err < 1e-7, f"{name} differs from the numerical gradient (max relative error {err:.2e})"


@pytest.mark.fn("tanh_backward")
def test_tanh_backward_values():
    m = load(MOD)
    a = np.array([0.0, 1.0, -2.0, 5.0], dtype=np.float32)
    out = np.tanh(a)
    dx = m.tanh_backward(np.ones(4, np.float32), out)
    check_array(dx, "dx", shape=(4,), dtype=np.float32)
    assert_close(dx, 1 - np.tanh(a.astype(np.float64)) ** 2, "dx", atol=1e-6,
                 hint="the local derivative is 1 - tanh(a)^2, and the cache already holds tanh(a)")


@pytest.mark.fn("tanh_backward")
def test_tanh_backward_gradcheck():
    m = load(MOD)
    rng = np.random.default_rng(2)
    a = rng.standard_normal((3, 4))
    G = rng.standard_normal((3, 4))
    dx = m.tanh_backward(G, np.tanh(a))
    err = rel_error(dx, numerical_grad(lambda: float(np.sum(G * np.tanh(a))), a))
    assert err < 1e-7, f"dx differs from the numerical gradient (max relative error {err:.2e}) — multiply by the upstream dout"
