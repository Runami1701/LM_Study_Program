"""Shared test helpers: implementation switch, friendly asserts, gradient checks.

Every test imports learner code through :func:`load`, never directly, so the same
tests run against ``workspace/`` (default) or ``solutions/`` (COURSE_IMPL=solutions).
"""

from __future__ import annotations

import importlib
import os
import time

import numpy as np

IMPLS = ("workspace", "solutions")


def impl_name() -> str:
    name = os.environ.get("COURSE_IMPL", "workspace")
    if name not in IMPLS:
        raise ValueError(f"COURSE_IMPL must be one of {IMPLS}, got {name!r}")
    return name


def load(module: str):
    """Import ``<impl>.<module>``, e.g. load("stage1.data")."""
    return importlib.import_module(f"{impl_name()}.{module}")


def load_reference(module: str):
    """Import the reference solution regardless of COURSE_IMPL (used only for
    relative speed measurements, never to compute expected values that the
    learner's code is compared against)."""
    return importlib.import_module(f"solutions.{module}")


# ----------------------------------------------------------------- friendly asserts

def _fmt_shape(shape) -> str:
    return "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"


def check_array(arr, name: str, shape=None, dtype=None, shape_desc: str | None = None, hint: str | None = None):
    """Assert `arr` is an ndarray with the given shape and dtype, with a teaching message."""
    assert isinstance(arr, np.ndarray), (
        f"{name} should be a NumPy array, got {type(arr).__name__}"
        + (f" — {hint}" if hint else "")
    )
    if shape is not None and tuple(arr.shape) != tuple(shape):
        desc = f"{shape_desc} = " if shape_desc else ""
        msg = f"expected {name} shape {desc}{_fmt_shape(shape)}, got {_fmt_shape(arr.shape)}"
        raise AssertionError(msg + (f" — {hint}" if hint else ""))
    if dtype is not None and arr.dtype != np.dtype(dtype):
        msg = f"expected {name} dtype {np.dtype(dtype).name}, got {arr.dtype.name}"
        if np.dtype(dtype) == np.float32 and arr.dtype == np.float64:
            msg += " — something upcast to float64 (a float64 array, np.float64 scalar, or default-dtype constructor?)"
        elif np.dtype(dtype) == np.int64 and arr.dtype.kind in "iu":
            msg += " — convert with .astype(np.int64)"
        raise AssertionError(msg + (f" — {hint}" if hint and "—" not in msg else ""))


def assert_close(actual, expected, name: str, rtol: float = 1e-5, atol: float = 1e-6, hint: str | None = None):
    """allclose with a message that shows where and by how much values differ."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape {_fmt_shape(actual.shape)} doesn't match expected {_fmt_shape(expected.shape)}"
        )
    a64 = actual.astype(np.float64)
    e64 = expected.astype(np.float64)
    bad = ~np.isclose(a64, e64, rtol=rtol, atol=atol, equal_nan=False)
    if bad.any():
        idx = tuple(int(i) for i in np.argwhere(bad)[0])
        diff = np.abs(a64 - e64)
        diff = np.where(np.isfinite(diff), diff, np.inf)
        where = f"[{', '.join(map(str, idx))}]" if idx else ""
        msg = (
            f"{name}: {int(bad.sum())} of {bad.size} values are wrong. "
            f"First at {name}{where}: got {a64[idx]:.6g}, expected {e64[idx]:.6g} "
            f"(max abs error {diff.max():.3g})"
        )
        raise AssertionError(msg + (f" — {hint}" if hint else ""))


def assert_equal(actual, expected, name: str, hint: str | None = None):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        if actual.shape == expected.shape:
            idx = tuple(int(i) for i in np.argwhere(actual != expected)[0])
            where = f"[{', '.join(map(str, idx))}]" if idx else ""
            msg = f"{name}{where}: got {actual[idx]!r}, expected {expected[idx]!r}"
        else:
            msg = f"{name}: shape {_fmt_shape(actual.shape)}, expected {_fmt_shape(expected.shape)}"
        raise AssertionError(msg + (f" — {hint}" if hint else ""))


# --------------------------------------------------------------------- timing

def best_time(fn, repeats: int = 5, number: int = 10) -> float:
    """Best-of-`repeats` wall time (seconds) for `number` calls of fn()."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ------------------------------------------------------------- gradient checking

def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Centered-difference gradient of scalar f() with respect to array x
    (perturbed in place, float64). Used by the course's own tests."""
    assert x.dtype == np.float64, "numerical gradients need float64"
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        grad[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def rel_error(a, b, delta: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), delta)))
