"""Attention (reference solution)."""

import numpy as np

from ..autograd import ops
from ..autograd.tensor import Tensor
from .layers import Linear, Module

NEG = -1e9   # "minus infinity" that stays finite in float32


def softmax(x: Tensor) -> Tensor:
    """Softmax over the last axis as one op: dS = P * (dP - rowsum(dP * P))."""
    z = x.data - x.data.max(axis=-1, keepdims=True)
    e = np.exp(z)
    P = e / e.sum(axis=-1, keepdims=True)
    out = Tensor(P, (x,), "softmax")

    def _backward():
        g = out.grad
        x.grad += P * (g - (g * P).sum(axis=-1, keepdims=True))

    out._backward = _backward
    return out


def causal_mask(T: int) -> np.ndarray:
    """(T, T) float32: 0 where key <= query (allowed), NEG above the diagonal."""
    return np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)


def attention_single_head(x: Tensor, Wq: Tensor, Wk: Tensor, Wv: Tensor, causal: bool = False) -> Tensor:
    """softmax(Q K^T / sqrt(d_h)) V for x (B, T, d) -> (B, T, d_h)."""
    q, k, v = x @ Wq, x @ Wk, x @ Wv                       # (B, T, d_h) each
    d_h = q.shape[-1]
    scores = (q @ ops.transpose(k, (0, 2, 1))) * (1.0 / np.sqrt(d_h))   # (B, T, T)
    if causal:
        scores = scores + causal_mask(x.shape[1])
    return softmax(scores) @ v                             # (B, T, d_h)


class MultiHeadAttention(Module):
    """h heads of size d/h, one fused QKV projection and an output projection."""

    def __init__(self, d: int, n_heads: int, rng: np.random.Generator):
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.qkv = Linear(d, 3 * d, rng)
        self.proj = Linear(d, d, rng)

    def __call__(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, d = x.shape
        h = self.n_heads
        dh = d // h
        qkv = ops.reshape(self.qkv(x), (B, T, 3, h, dh))          # (B, T, 3, h, dh)
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))                # (3, B, h, T, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]                         # (B, h, T, dh) each
        scores = (q @ ops.transpose(k, (0, 1, 3, 2))) * (1.0 / np.sqrt(dh))   # (B, h, T, T)
        if causal:
            scores = scores + causal_mask(T)
        y = softmax(scores) @ v                                  # (B, h, T, dh)
        y = ops.reshape(ops.transpose(y, (0, 2, 1, 3)), (B, T, d))   # heads side by side
        return self.proj(y)
