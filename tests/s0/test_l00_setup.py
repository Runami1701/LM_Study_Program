"""Checks for s0-l00 — Setup & NumPy pitfalls."""

import math

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load, load_reference

MOD = "stage0.numpy_basics"


@pytest.mark.fn("to_f32")
def test_to_f32_returns_tuple():
    m = load(MOD)
    out = m.to_f32(np.zeros(3))
    assert isinstance(out, tuple), (
        f"to_f32 should return a tuple, got {type(out).__name__} — "
        "even a single argument comes back as a 1-tuple, so `(a,) = to_f32(a)` always works"
    )
    assert len(out) == 1, f"to_f32 got 1 argument but returned {len(out)} items"
    out3 = m.to_f32(np.zeros(1), np.zeros(2), np.zeros(3))
    assert len(out3) == 3, f"to_f32 got 3 arguments but returned {len(out3)} items"
    assert [a.shape for a in out3] == [(1,), (2,), (3,)], "outputs must keep the order of the arguments"


@pytest.mark.fn("to_f32")
def test_to_f32_converts_dtypes():
    m = load(MOD)
    f64 = np.array([[0.1, 0.2, 0.3], [1.5, -2.0, 3e5]])
    ints = np.arange(4)
    a, b, c, d = m.to_f32(f64, ints, [1, 2, 3], 2.5)
    check_array(a, "to_f32(float64 array)", shape=(2, 3), dtype=np.float32)
    check_array(b, "to_f32(int64 array)", shape=(4,), dtype=np.float32)
    check_array(c, "to_f32([1, 2, 3])", shape=(3,), dtype=np.float32,
                hint="Python lists should become arrays too")
    check_array(d, "to_f32(2.5)", shape=(), dtype=np.float32,
                hint="a Python scalar becomes a 0-d array")
    assert_close(a, f64, "to_f32(float64 array)", rtol=1e-6)
    assert_close(b, [0, 1, 2, 3], "to_f32(int64 array)")
    assert f64.dtype == np.float64, "to_f32 must not change its input's dtype in place"


@pytest.mark.fn("to_f32")
def test_to_f32_no_copy_when_already_float32():
    m = load(MOD)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    (y,) = m.to_f32(x)
    assert np.shares_memory(x, y), (
        "an input that is already float32 should be returned without copying — "
        "`.astype(np.float32)` always copies; which function converts only when it has to?"
    )


@pytest.mark.fn("matmul_gflops")
def test_matmul_gflops_returns_positive_float():
    m = load(MOD)
    g = m.matmul_gflops(128, 3, np.random.default_rng(0))
    assert isinstance(g, float), (
        f"matmul_gflops should return a Python float, got {type(g).__name__} — wrap the result in float(...)"
    )
    assert math.isfinite(g) and g > 0, f"GFLOPS must be a positive finite number, got {g}"
    assert g < 1e5, (
        f"{g:.3g} GFLOPS is impossibly fast for a desktop CPU — did you forget to divide by 1e9 "
        "(FLOPS -> GFLOPS)?"
    )


@pytest.mark.fn("matmul_gflops")
def test_matmul_gflops_plausible():
    """Compare with the reference measurement on this machine; the bound is
    deliberately loose (30x) so timing noise never fails a correct solution."""
    m = load(MOD)
    ref = load_reference(MOD)
    rng = np.random.default_rng(0)
    yours = max(m.matmul_gflops(256, 5, rng) for _ in range(3))
    theirs = max(ref.matmul_gflops(256, 5, rng) for _ in range(3))
    ratio = yours / theirs
    assert 1 / 30 < ratio < 30, (
        f"your measurement ({yours:.2f} GFLOPS) is {ratio:.3g}x the reference ({theirs:.2f}) — "
        "check the operation count (2·n³ per product) and that you divide by the total time of all reps"
    )
