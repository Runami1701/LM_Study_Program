"""Lesson n-l02 — Changing shapes.

Fill in each function, then run:  python -m course check n-l02
Replace each `raise NotImplementedError(...)` line with your code.
"""

import numpy as np


def to_rows(v: np.ndarray, rows: int) -> np.ndarray:
    """Reshape the 1-D array `v` into a matrix with `rows` rows.
    Work out the number of columns with -1.

    Example: to_rows(np.arange(6), 2) returns [[0 1 2] [3 4 5]].
    """
    raise NotImplementedError("to_rows")


def flatten_each(x: np.ndarray) -> np.ndarray:
    """x has shape (B, k, e). Return shape (B, k*e): one long row per example.

    Example: an input of shape (2, 3, 4) gives shape (2, 12).
    """
    raise NotImplementedError("flatten_each")


def swap_last_two(x: np.ndarray) -> np.ndarray:
    """Swap the last two axes. Shape (B, n, m) becomes (B, m, n).
    (For a matrix (n, m), this is the transpose (m, n).)
    """
    raise NotImplementedError("swap_last_two")


def stack_two(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a and b have the same shape. Stack them along a NEW first axis.

    Example: two arrays of shape (3,) give shape (2, 3).
    """
    raise NotImplementedError("stack_two")
