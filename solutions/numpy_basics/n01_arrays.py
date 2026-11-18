"""Lesson n-l01 — Arrays, shapes and number types (reference solution)."""

import numpy as np


def zeros_grid(rows: int, cols: int) -> np.ndarray:
    """Zeros of shape (rows, cols), float32."""
    return np.zeros((rows, cols), dtype=np.float32)


def describe(a: np.ndarray) -> tuple:
    """(shape, ndim, size, dtype name)."""
    return (a.shape, a.ndim, a.size, a.dtype.name)


def to_float32(values) -> np.ndarray:
    """Any list, number or array as float32."""
    return np.array(values, dtype=np.float32)


def multiply_both_ways(A: np.ndarray, B: np.ndarray) -> tuple:
    """(entry-by-entry product, matrix product)."""
    return (A * B, A @ B)
