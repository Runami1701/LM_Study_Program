"""Checks for n-l03 — Picking entries."""

import numpy as np
import pytest

from tests.helpers import assert_close, assert_equal, check_array, load

MOD = "numpy_basics.n03_indexing"


@pytest.mark.fn("last_column")
def test_last_column():
    m = load(MOD)
    A = np.array([[10, 11, 12], [20, 21, 22]])
    out = m.last_column(A)
    check_array(out, "last_column", shape=(2,), hint="A[:, -1] gives a 1-D array of one entry per row")
    assert_equal(out, [12, 22], "last_column")


@pytest.mark.fn("window")
def test_window():
    m = load(MOD)
    d = np.array([5, 6, 7, 8, 9])
    assert_equal(m.window(d, 1, 3), [6, 7, 8], "window(d, 1, 3)", hint="the slice start:start+k has exactly k entries")
    assert_equal(m.window(d, 3, 2), [8, 9], "window(d, 3, 2)")


@pytest.mark.fn("lookup_rows")
def test_lookup_rows():
    m = load(MOD)
    table = np.array([[0.1, 0.2], [0.3, -0.4], [0.5, 0.6], [-0.7, 0.8]])
    ids = np.array([[1, 0, 3], [3, 3, 2]])
    out = m.lookup_rows(table, ids)
    check_array(out, "lookup_rows", shape=(2, 3, 2), hint="the shape of ids, then the length of one row")
    assert_close(out[0, 0], [0.3, -0.4], "result[0, 0] (row 1 of the table)")
    assert_close(out[1, 1], [-0.7, 0.8], "result[1, 1] (row 3 of the table)")


@pytest.mark.fn("pick_per_row")
def test_pick_per_row():
    m = load(MOD)
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]])
    out = m.pick_per_row(P, np.array([0, 2, 1]))
    check_array(out, "pick_per_row", shape=(3,),
                hint="P[rows, cols] with rows = np.arange(n); P[:, cols] would give a (3, 3) table instead")
    assert_close(out, [0.7, 0.8, 0.3], "pick_per_row")


@pytest.mark.fn("count_ids")
def test_count_ids():
    m = load(MOD)
    out = m.count_ids(np.array([1, 1, 3]), 4)
    check_array(out, "count_ids([1, 1, 3], 4)", shape=(4,), dtype=np.int64)
    assert_equal(out, [0, 2, 0, 1], "count_ids([1, 1, 3], 4)",
                 hint="counts[ids] += 1 loses repeats; np.add.at(counts, ids, 1) keeps them")
    assert_equal(m.count_ids(np.array([0, 0, 0, 2]), 3), [3, 0, 1], "count_ids([0, 0, 0, 2], 3)")
