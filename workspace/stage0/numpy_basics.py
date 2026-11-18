"""Lesson s0-l00 — Setup & NumPy pitfalls.

Implement the functions below, then run:  python -m course check s0-l00
"""

import time

import numpy as np


def to_f32(*arrays) -> tuple:
    """Convert every argument to a float32 NumPy array.

    Args:
        *arrays: any number of array-likes (NumPy arrays of any dtype,
            Python lists, Python scalars).

    Returns:
        A tuple with one float32 ndarray per argument, in the same order
        (a tuple even when there is a single argument).
        An argument that is already a float32 ndarray is returned as-is:
        no copy (the result shares memory with the input).
    """
    raise NotImplementedError("to_f32")


def matmul_gflops(n: int, reps: int, rng: np.random.Generator) -> float:
    """Measure this machine's float32 matrix-multiply speed in GFLOPS.

    Args:
        n: matrix size; multiply two (n, n) float32 matrices.
        reps: how many timed multiplications to average over.
        rng: a NumPy Generator, used to create the random matrices.

    Returns:
        Billions of floating-point operations per second, as a Python float.
        Count 2·n³ operations per (n, n) @ (n, n) product.
        Do one untimed warm-up multiplication before timing.
    """
    raise NotImplementedError("matmul_gflops")
