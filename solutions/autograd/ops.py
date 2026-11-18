"""Differentiable operations for the autograd engine (reference solution)."""

import numpy as np

from .tensor import Tensor, unbroadcast


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """a @ b, including batched (..., n, m) @ (..., m, p) with broadcasting."""
    out = Tensor(a.data @ b.data, (a, b), "@")

    def _backward():
        g = out.grad
        a.grad += unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.data.shape)   # dA = dOut B^T
        b.grad += unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.data.shape)   # dB = A^T dOut

    out._backward = _backward
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    out = Tensor(t, (a,), "tanh")

    def _backward():
        a.grad += out.grad * (1.0 - t * t)

    out._backward = _backward
    return out


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:  # noqa: A001 - mirrors np.sum
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum")

    def _backward():
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)                  # put the summed axes back as length 1
        a.grad += np.broadcast_to(g, a.data.shape)       # every input got weight 1

    out._backward = _backward
    return out


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    n = a.data.size if axis is None else int(np.prod([a.data.shape[i] for i in np.atleast_1d(axis)]))
    return sum(a, axis=axis, keepdims=keepdims) * (1.0 / n)


def reshape(a: Tensor, shape: tuple) -> Tensor:
    out = Tensor(a.data.reshape(shape), (a,), "reshape")

    def _backward():
        a.grad += out.grad.reshape(a.data.shape)

    out._backward = _backward
    return out


def transpose(a: Tensor, axes: tuple) -> Tensor:
    out = Tensor(np.transpose(a.data, axes), (a,), "transpose")
    inverse = np.argsort(axes)                           # the permutation that undoes `axes`

    def _backward():
        a.grad += np.transpose(out.grad, inverse)

    out._backward = _backward
    return out


def getitem(a: Tensor, key) -> Tensor:
    """a[key] for slices or integer arrays (an embedding lookup is getitem)."""
    out = Tensor(a.data[key], (a,), "getitem")

    def _backward():
        np.add.at(a.grad, key, out.grad)                 # repeated indices accumulate

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """Mean cross-entropy of (N, V) logits against (N,) integer targets (fused)."""
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    N = z.shape[0]
    loss = -(z[np.arange(N), y] - lse).mean()
    out = Tensor(np.asarray(loss, dtype=logits.data.dtype), (logits,), "cross_entropy")

    def _backward():
        d = np.exp(z - lse[:, None])                    # softmax
        d[np.arange(N), y] -= 1.0
        logits.grad += d * (out.grad / N)               # s1-l05, times the upstream scalar

    out._backward = _backward
    return out
