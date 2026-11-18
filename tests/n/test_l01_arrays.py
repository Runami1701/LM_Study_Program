"""Checks for n-l01 — Arrays, shapes and number types."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load

MOD = "numpy_basics.n01_arrays"


@pytest.mark.fn("zeros_grid")
def test_zeros_grid():
    m = load(MOD)
    g = m.zeros_grid(2, 3)
    check_array(g, "zeros_grid(2, 3)", shape=(2, 3), dtype=np.float32,
                hint="pass the shape as one tuple: np.zeros((rows, cols), dtype=np.float32)")
    assert not g.any(), "every entry should be 0"
    check_array(m.zeros_grid(1, 5), "zeros_grid(1, 5)", shape=(1, 5), dtype=np.float32)


@pytest.mark.fn("describe")
def test_describe():
    m = load(MOD)
    got = m.describe(np.zeros((2, 3)))
    assert isinstance(got, tuple) and len(got) == 4, f"describe should return a tuple of 4 things, got {got!r}"
    assert got == ((2, 3), 2, 6, "float64"), (
        f"describe(np.zeros((2, 3))) should be ((2, 3), 2, 6, 'float64'), got {got!r} — "
        "use a.shape, a.ndim, a.size and a.dtype.name"
    )
    assert m.describe(np.arange(5)) == ((5,), 1, 5, "int64"), "describe(np.arange(5)) should be ((5,), 1, 5, 'int64')"


@pytest.mark.fn("to_float32")
def test_to_float32():
    m = load(MOD)
    a = m.to_float32([1, 2, 3])
    check_array(a, "to_float32([1, 2, 3])", shape=(3,), dtype=np.float32, hint="a list must become an array too")
    assert_close(a, [1, 2, 3], "to_float32([1, 2, 3])")
    b = m.to_float32(np.array([[0.5, 1.5]]))
    check_array(b, "to_float32(a float64 matrix)", shape=(1, 2), dtype=np.float32)
    check_array(m.to_float32(2.5), "to_float32(2.5)", shape=(), dtype=np.float32)


@pytest.mark.fn("multiply_both_ways")
def test_multiply_both_ways():
    m = load(MOD)
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[10, 0], [0, 1]])
    out = m.multiply_both_ways(A, B)
    assert isinstance(out, tuple) and len(out) == 2, "return a tuple of two arrays: (entry by entry, matrix product)"
    each, matmul = out
    assert_close(each, [[10, 0], [0, 4]], "first result (entry by entry)",
                 hint="use * for the entry-by-entry product")
    assert_close(matmul, [[10, 2], [30, 4]], "second result (matrix product)",
                 hint="use @ for the matrix product; * multiplies matching entries only")
