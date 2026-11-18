"""Lesson n-l02 — Changing shapes (reference solution)."""

import numpy as np


def to_rows(v: np.ndarray, rows: int) -> np.ndarray:
    """Reshape v into `rows` rows."""
    return v.reshape(rows, -1)


def flatten_each(x: np.ndarray) -> np.ndarray:
    """(B, k, e) -> (B, k*e)."""
    return x.reshape(x.shape[0], -1)


def swap_last_two(x: np.ndarray) -> np.ndarray:
    """Swap the last two axes."""
    return np.swapaxes(x, -1, -2)


def stack_two(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack along a new first axis."""
    return np.stack([a, b])
