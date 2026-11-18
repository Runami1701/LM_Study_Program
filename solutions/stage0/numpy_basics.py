"""Lesson s0-l00 — Setup & NumPy pitfalls (reference solution)."""

import time

import numpy as np


def to_f32(*arrays) -> tuple:
    """Convert every argument to a float32 NumPy array (no copy if already float32)."""
    return tuple(np.asarray(a, dtype=np.float32) for a in arrays)


def matmul_gflops(n: int, reps: int, rng: np.random.Generator) -> float:
    """Measure float32 matmul speed in GFLOPS (2·n³ operations per product)."""
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    a @ b  # warm-up: first call pays one-off costs (thread pool start, page faults)
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    seconds = time.perf_counter() - t0
    flops = 2.0 * n**3 * reps
    return float(flops / seconds / 1e9)
