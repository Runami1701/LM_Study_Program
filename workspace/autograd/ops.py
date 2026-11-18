"""Differentiable operations (lesson a-l02  ->  python -m course check a-l02; shared_weight_grad: a-l05).

Each op computes its output with NumPy, wraps it in a Tensor whose parents are
the inputs, and attaches a _backward closure that ACCUMULATES (+=) the gradient
into every input that is a Tensor. Use unbroadcast wherever NumPy may have broadcast.
"""

import numpy as np

from .tensor import Tensor, unbroadcast


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """a @ b for (n, m) @ (m, p), and batched (..., n, m) @ (..., m, p) or (..., n, m) @ (m, p)."""
    raise NotImplementedError("matmul")


def shared_weight_grad(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Lesson a-l05. The gradient of a weight W of shape (n, p) that was used as a @ W,
    where a has shape (..., n) (any number of leading batch axes) and g = out.grad has
    shape (..., p). Return the (n, p) gradient, computed as ONE matrix product of all
    rows stacked together (no batched product followed by a sum).
    """
    raise NotImplementedError("shared_weight_grad")


def tanh(a: Tensor) -> Tensor:
    """Elementwise tanh."""
    raise NotImplementedError("tanh")


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:  # noqa: A001 - mirrors np.sum
    """Sum over `axis` (an int, a tuple of ints, or None for everything), like np.sum."""
    raise NotImplementedError("sum")


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Mean over `axis`, like np.mean (you may build it from sum and *)."""
    raise NotImplementedError("mean")


def reshape(a: Tensor, shape: tuple) -> Tensor:
    """Reshape (a -1 entry is allowed, as in NumPy)."""
    raise NotImplementedError("reshape")


def transpose(a: Tensor, axes: tuple) -> Tensor:
    """Permute axes, like np.transpose(a, axes)."""
    raise NotImplementedError("transpose")


def getitem(a: Tensor, key) -> Tensor:
    """a[key] for slices and integer arrays. An embedding lookup is getitem(C, ids)."""
    raise NotImplementedError("getitem")


def cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """Mean cross-entropy of (N, V) logits against (N,) int targets, as ONE fused op:
    a scalar Tensor whose _backward uses the s1-l05 gradient (times out.grad)."""
    raise NotImplementedError("cross_entropy")
