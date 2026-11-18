"""Lesson n-l01 — Arrays, shapes and number types.

Fill in each function, then run:  python -m course check n-l01
Replace each `raise NotImplementedError(...)` line with your code.
"""

import numpy as np


def zeros_grid(rows: int, cols: int) -> np.ndarray:
    """Return an array of zeros with `rows` rows and `cols` columns, in float32.

    Example: zeros_grid(2, 3) has shape (2, 3) and dtype float32.
    """
    raise NotImplementedError("zeros_grid")


def describe(a: np.ndarray) -> tuple:
    """Return four facts about the array `a`, as a tuple:
    (its shape, its number of sides, its number of entries, the name of its dtype).

    Example: describe(np.zeros((2, 3))) returns ((2, 3), 2, 6, "float64").
    """
    raise NotImplementedError("describe")


def to_float32(values) -> np.ndarray:
    """Return `values` (a Python list, a number, or an array) as a float32 array.

    Example: to_float32([1, 2, 3]) returns array([1., 2., 3.], dtype=float32).
    """
    raise NotImplementedError("to_float32")


def multiply_both_ways(A: np.ndarray, B: np.ndarray) -> tuple:
    """A and B are square matrices of the same shape. Return a tuple of two arrays:
    (A and B multiplied entry by entry, the matrix product of A and B).
    """
    raise NotImplementedError("multiply_both_ways")
