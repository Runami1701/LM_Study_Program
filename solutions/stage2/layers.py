"""Stage 2 building blocks on the autograd engine (reference solution)."""

import math

import numpy as np

from ..autograd import ops
from ..autograd.tensor import Tensor, unbroadcast


class Module:
    """Given: collects Tensor attributes (and sub-modules, also inside lists)."""

    def named_parameters(self, prefix: str = "") -> dict:
        out = {}
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                out[prefix + name] = value
            elif isinstance(value, Module):
                out.update(value.named_parameters(prefix + name + "."))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        out.update(item.named_parameters(f"{prefix}{name}.{i}."))
        return out

    def parameters(self) -> list:
        return list(self.named_parameters().values())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


def get_batch_seq(data: np.ndarray, B: int, T: int, rng: np.random.Generator):
    """x (B, T) and y (B, T) int64, with y[:, t] = the character after x[:, t]."""
    starts = rng.integers(0, len(data) - T, size=B)           # (B,)
    idx = starts[:, None] + np.arange(T)[None, :]               # (B, T)
    return data[idx].astype(np.int64), data[idx + 1].astype(np.int64)


class Linear(Module):
    """x @ W + b with W ~ N(0, 0.02^2)."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, bias: bool = True):
        self.W = Tensor((rng.standard_normal((n_in, n_out)) * 0.02).astype(np.float32))
        self.b = Tensor(np.zeros(n_out, dtype=np.float32)) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.W
        return out + self.b if self.b is not None else out


class Embeddings(Module):
    """Token embedding + learned position embedding."""

    def __init__(self, V: int, T: int, d: int, rng: np.random.Generator):
        self.wte = Tensor((rng.standard_normal((V, d)) * 0.02).astype(np.float32))
        self.wpe = Tensor((rng.standard_normal((T, d)) * 0.02).astype(np.float32))

    def __call__(self, idx: np.ndarray) -> Tensor:
        t = idx.shape[1]
        return self.wte[idx] + self.wpe[np.arange(t)]          # (B, t, d) + (t, d)


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalize over the last axis, then scale by gamma and shift by beta (fused op)."""
    mu = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)                             # 1 / sigma
    xhat = (x.data - mu) * rstd
    out = Tensor(xhat * gamma.data + beta.data, (x, gamma, beta), "layernorm")

    def _backward():
        g = out.grad
        gamma.grad += unbroadcast(g * xhat, gamma.data.shape)
        beta.grad += unbroadcast(g, beta.data.shape)
        dxhat = g * gamma.data
        x.grad += rstd * (dxhat
                          - dxhat.mean(axis=-1, keepdims=True)
                          - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))

    out._backward = _backward
    return out


class LayerNorm(Module):
    def __init__(self, d: int):
        self.gamma = Tensor(np.ones(d, dtype=np.float32))
        self.beta = Tensor(np.zeros(d, dtype=np.float32))

    def __call__(self, x: Tensor) -> Tensor:
        return layernorm(x, self.gamma, self.beta)


_C = math.sqrt(2.0 / math.pi)   # a Python float: np.sqrt would return float64 and upcast (s0-l00)


def gelu(x: Tensor) -> Tensor:
    """GELU, tanh approximation: 0.5 x (1 + tanh(c (x + 0.044715 x^3)))."""
    a = x.data
    u = _C * (a + 0.044715 * a * a * a)
    t = np.tanh(u)
    out = Tensor(0.5 * a * (1.0 + t), (x,), "gelu")

    def _backward():
        du = _C * (1.0 + 3 * 0.044715 * a * a)
        x.grad += out.grad * (0.5 * (1.0 + t) + 0.5 * a * (1.0 - t * t) * du)

    out._backward = _backward
    return out
