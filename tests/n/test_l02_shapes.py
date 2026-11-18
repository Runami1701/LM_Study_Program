"""Checks for n-l02 — Changing shapes."""

import numpy as np
import pytest

from tests.helpers import assert_equal, check_array, load

MOD = "numpy_basics.n02_shapes"


@pytest.mark.fn("to_rows")
def test_to_rows():
    m = load(MOD)
    out = m.to_rows(np.arange(6), 2)
    check_array(out, "to_rows(np.arange(6), 2)", shape=(2, 3))
    assert_equal(out, [[0, 1, 2], [3, 4, 5]], "to_rows(np.arange(6), 2)")
    check_array(m.to_rows(np.arange(12), 3), "to_rows(np.arange(12), 3)", shape=(3, 4),
                hint="use -1 for the number of columns so it works for any length")


@pytest.mark.fn("flatten_each")
def test_flatten_each():
    m = load(MOD)
    x = np.arange(24).reshape(2, 3, 4)
    out = m.flatten_each(x)
    check_array(out, "flatten_each(shape (2, 3, 4))", shape=(2, 12))
    assert_equal(out[1], np.arange(12, 24), "row 1 of the result", hint="row b must hold example b's numbers in order")
    check_array(m.flatten_each(np.zeros((1, 8, 24))), "flatten_each(shape (1, 8, 24))", shape=(1, 192),
                hint="keep the batch axis: reshape to (x.shape[0], -1), not (-1,)")


@pytest.mark.fn("swap_last_two")
def test_swap_last_two():
    m = load(MOD)
    x = np.arange(24).reshape(2, 3, 4)
    out = m.swap_last_two(x)
    check_array(out, "swap_last_two(shape (2, 3, 4))", shape=(2, 4, 3))
    assert out[1, 3, 2] == x[1, 2, 3], "entry [b, j, i] of the result must be entry [b, i, j] of the input — swap, don't reshape"
    M = np.arange(6).reshape(2, 3)
    assert_equal(m.swap_last_two(M), M.T, "swap_last_two of a matrix", hint="for a matrix this is the transpose")


@pytest.mark.fn("stack_two")
def test_stack_two():
    m = load(MOD)
    out = m.stack_two(np.array([1, 2, 3]), np.array([4, 5, 6]))
    check_array(out, "stack_two of two (3,) arrays", shape=(2, 3),
                hint="np.stack adds a new axis; np.concatenate would give (6,)")
    assert_equal(out, [[1, 2, 3], [4, 5, 6]], "stack_two result")
