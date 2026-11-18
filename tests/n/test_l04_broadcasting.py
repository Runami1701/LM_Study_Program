"""Checks for n-l04 — Sums along an axis, and broadcasting."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load

MOD = "numpy_basics.n04_broadcasting"
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.fn("column_sums")
def test_column_sums():
    m = load(MOD)
    out = m.column_sums(A)
    check_array(out, "column_sums", shape=(3,), hint="axis=0 adds down the rows, leaving one number per column")
    assert_close(out, [5, 7, 9], "column_sums")


@pytest.mark.fn("row_means")
def test_row_means():
    m = load(MOD)
    out = m.row_means(A)
    check_array(out, "row_means", shape=(2,), hint="axis=1 works across the columns, leaving one number per row")
    assert_close(out, [2, 5], "row_means")


@pytest.mark.fn("normalize_rows")
def test_normalize_rows():
    m = load(MOD)
    S = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = m.normalize_rows(S)
    check_array(out, "normalize_rows", shape=(2, 2))
    assert_close(out.sum(axis=1), [1, 1], "row sums after normalize_rows",
                 hint="divide by A.sum(axis=1, keepdims=True); without keepdims a square matrix is divided column by column")
    assert_close(m.normalize_rows(A), A / A.sum(axis=1, keepdims=True), "normalize_rows(A)")


@pytest.mark.fn("outer_sum")
def test_outer_sum():
    m = load(MOD)
    out = m.outer_sum(np.array([0, 10, 20]), np.arange(4))
    check_array(out, "outer_sum", shape=(3, 4), hint="a[:, None] is (3, 1), b[None, :] is (1, 4)")
    assert_close(out[2], [20, 21, 22, 23], "row 2 of outer_sum")


@pytest.mark.fn("shift_rows")
def test_shift_rows():
    m = load(MOD)
    out = m.shift_rows(np.array([[1.0, 5.0, 3.0], [-2.0, -1.0, -4.0]]))
    check_array(out, "shift_rows", shape=(2, 3))
    assert_close(out, [[-4, 0, -2], [-1, 0, -3]], "shift_rows",
                 hint="subtract A.max(axis=1, keepdims=True): each row's own maximum")
