"""Checks for s1-l03 — Linear layer and tanh (forward)."""

import warnings

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load

MOD = "stage1.layers"


@pytest.mark.fn("linear_forward")
def test_linear_worked_example():
    """The 2x3 worked example from the lesson."""
    m = load(MOD)
    x = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32)            # (B=2, n=2)
    W = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]], dtype=np.float32)  # (n=2, m=3)
    b = np.array([0.5, -1.0, 0.0], dtype=np.float32)                      # (m=3,)
    out, _ = m.linear_forward(x, W, b)
    check_array(out, "out", shape=(2, 3), dtype=np.float32, shape_desc="(B, m)")
    assert_close(out, [[5.5, 1.0, 0.0], [-1.5, -2.0, -0.5]], "out")


@pytest.mark.fn("linear_forward")
def test_linear_random_matches_sum_definition():
    m = load(MOD)
    rng = np.random.default_rng(0)
    B, n, mm = 6, 5, 4
    x = rng.standard_normal((B, n)).astype(np.float32)
    W = rng.standard_normal((n, mm)).astype(np.float32)
    b = rng.standard_normal(mm).astype(np.float32)
    out, _ = m.linear_forward(x, W, b)
    check_array(out, "out", shape=(B, mm), dtype=np.float32, shape_desc="(B, m)")
    expected = np.einsum("bi,ij->bj", x.astype(np.float64), W.astype(np.float64)) + b
    assert_close(out, expected, "out", rtol=1e-5, atol=1e-5,
                 hint="out[b, j] = sum_i x[b, i] W[i, j] + b[j]")


@pytest.mark.fn("linear_forward")
def test_linear_bias_added_to_every_row():
    m = load(MOD)
    x = np.zeros((3, 2), dtype=np.float32)
    W = np.ones((2, 4), dtype=np.float32)
    b = np.array([1, 2, 3, 4], dtype=np.float32)
    out, _ = m.linear_forward(x, W, b)
    assert_close(out, np.tile(b, (3, 1)), "out (x = 0)",
                 hint="with x = 0 every row of the output must equal b")


@pytest.mark.fn("linear_forward")
def test_linear_cache():
    m = load(MOD)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    W = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    _, cache = m.linear_forward(x, W, b)
    assert isinstance(cache, tuple) and len(cache) == 3, (
        "cache should be the tuple (x, W, b) — the backward pass in s1-l06 needs x (for dW) and W (for dx)"
    )
    for got, want, name in zip(cache, (x, W, b), ("x", "W", "b")):
        assert np.array_equal(got, want), f"cache should be (x, W, b); the entry for {name} differs"


@pytest.mark.fn("tanh_forward")
def test_tanh_values_and_cache():
    m = load(MOD)
    x = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    out, cache = m.tanh_forward(x)
    check_array(out, "out", shape=(2, 3), dtype=np.float32)
    assert_close(out, [[-0.96402758, -0.76159416, 0.0], [0.76159416, 0.96402758, 0.99505475]], "out")
    assert isinstance(cache, np.ndarray) and np.array_equal(cache, out), (
        "cache should be the output array itself (backward uses 1 - out²)"
    )


@pytest.mark.fn("tanh_forward")
def test_tanh_any_shape():
    m = load(MOD)
    x = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32)
    out, _ = m.tanh_forward(x)
    check_array(out, "out", shape=(2, 3, 4), dtype=np.float32)
    assert_close(out, np.tanh(x.astype(np.float64)), "out", rtol=1e-5, atol=1e-6)


@pytest.mark.fn("tanh_forward")
def test_tanh_large_inputs_saturate_without_nan():
    m = load(MOD)
    x = np.array([-1000.0, -20.0, 0.0, 20.0, 1000.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, _ = m.tanh_forward(x)
    assert np.all(np.isfinite(out)), (
        f"tanh of [-1000, -20, 0, 20, 1000] gave {out} — computing it from exp(x) overflows "
        "for large |x|; use NumPy's built-in tanh"
    )
    assert_close(out, [-1, -1, 0, 1, 1], "tanh(large inputs)")
