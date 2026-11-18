"""Lesson n-l03 — Picking entries (reference solution)."""

import numpy as np


def last_column(A: np.ndarray) -> np.ndarray:
    """All rows, last column."""
    return A[:, -1]


def window(data: np.ndarray, start: int, k: int) -> np.ndarray:
    """k entries from position start."""
    return data[start:start + k]


def lookup_rows(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """One row of table per id."""
    return table[ids]


def pick_per_row(P: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """P[r, cols[r]] for every row r."""
    return P[np.arange(len(cols)), cols]


def count_ids(ids: np.ndarray, V: int) -> np.ndarray:
    """How often each id appears, repeats included."""
    counts = np.zeros(V, dtype=np.int64)
    np.add.at(counts, ids, 1)
    return counts
