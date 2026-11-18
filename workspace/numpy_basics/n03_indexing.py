"""Lesson n-l03 — Picking entries (indexing).

Fill in each function, then run:  python -m course check n-l03
Replace each `raise NotImplementedError(...)` line with your code. No loops needed.
"""

import numpy as np


def last_column(A: np.ndarray) -> np.ndarray:
    """Return the last column of the matrix A, as a 1-D array.

    Example: for [[10 11 12] [20 21 22]] return [12 22].
    """
    raise NotImplementedError("last_column")


def window(data: np.ndarray, start: int, k: int) -> np.ndarray:
    """Return the k entries of the 1-D array `data` that begin at position `start`.

    Example: window(np.array([5, 6, 7, 8, 9]), 1, 3) returns [6 7 8].
    """
    raise NotImplementedError("window")


def lookup_rows(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return the rows of `table` chosen by the integer array `ids`.
    The result has the shape of ids, followed by the length of one row.

    Example: table of shape (4, 2), ids of shape (2, 3) -> result of shape (2, 3, 2).
    """
    raise NotImplementedError("lookup_rows")


def pick_per_row(P: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """P has shape (n, m); cols has shape (n,). Return the n entries P[r, cols[r]].

    Example: P = [[0.7 0.2 0.1] [0.1 0.1 0.8]], cols = [0 2] returns [0.7 0.8].
    """
    raise NotImplementedError("pick_per_row")


def count_ids(ids: np.ndarray, V: int) -> np.ndarray:
    """Count how often each id 0 .. V-1 appears in the 1-D array `ids`.
    Return an int64 array of length V. Repeated ids must all be counted.

    Example: count_ids(np.array([1, 1, 3]), 4) returns [0 2 0 1].
    """
    raise NotImplementedError("count_ids")
