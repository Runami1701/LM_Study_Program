"""Lesson n-l04 — Sums along an axis, and broadcasting.

Fill in each function, then run:  python -m course check n-l04
Replace each `raise NotImplementedError(...)` line with your code. No loops needed.
"""

import numpy as np


def column_sums(A: np.ndarray) -> np.ndarray:
    """Return one sum per column of A. Shape (n, m) gives shape (m,)."""
    raise NotImplementedError("column_sums")


def row_means(A: np.ndarray) -> np.ndarray:
    """Return one mean per row of A. Shape (n, m) gives shape (n,)."""
    raise NotImplementedError("row_means")


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """Divide each row of A by that row's sum, so every row adds up to 1. Same shape as A."""
    raise NotImplementedError("normalize_rows")


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a has shape (n,), b has shape (m,). Return the (n, m) table with entry (i, j) = a[i] + b[j]."""
    raise NotImplementedError("outer_sum")


def shift_rows(A: np.ndarray) -> np.ndarray:
    """Subtract each row's largest entry from that row, so every row's maximum becomes 0."""
    raise NotImplementedError("shift_rows")
