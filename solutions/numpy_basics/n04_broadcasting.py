"""Lesson n-l04 — Sums along an axis, and broadcasting (reference solution)."""

import numpy as np


def column_sums(A: np.ndarray) -> np.ndarray:
    """axis=0 removes the rows."""
    return A.sum(axis=0)


def row_means(A: np.ndarray) -> np.ndarray:
    """axis=1 removes the columns."""
    return A.mean(axis=1)


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """Row sums as a column (n, 1), broadcast across each row."""
    return A / A.sum(axis=1, keepdims=True)


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 1) + (1, m) -> (n, m)."""
    return a[:, None] + b[None, :]


def shift_rows(A: np.ndarray) -> np.ndarray:
    """Row maxima as a column, subtracted from each row."""
    return A - A.max(axis=1, keepdims=True)
