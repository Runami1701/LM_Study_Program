"""Attention.

Lesson s2-l04: softmax, attention_single_head   ->  python -m course check s2-l04
Lesson s2-l05: causal_mask (+ causal=True)      ->  python -m course check s2-l05
Lesson s2-l06: MultiHeadAttention               ->  python -m course check s2-l06
"""

import numpy as np

from ..autograd import ops
from ..autograd.tensor import Tensor
from .layers import Linear, Module

NEG = -1e9   # "minus infinity" that stays finite in float32


def softmax(x: Tensor) -> Tensor:
    """Softmax over the last axis as ONE op (stable forward; derived backward
    dx = P * (g - sum(g * P, last axis))). Any number of leading axes."""
    raise NotImplementedError("softmax")


def causal_mask(T: int) -> np.ndarray:
    """(T, T) float32 array: 0 where key position <= query position, NEG above the diagonal."""
    raise NotImplementedError("causal_mask")


def attention_single_head(x: Tensor, Wq: Tensor, Wk: Tensor, Wv: Tensor, causal: bool = False) -> Tensor:
    """One attention head.

    x: (B, T, d); Wq, Wk, Wv: (d, d_h).
    q, k, v = x @ Wq, x @ Wk, x @ Wv                       # (B, T, d_h)
    scores = q @ k^T / sqrt(d_h)                            # (B, T, T)
    if causal: add causal_mask(T) to the scores (s2-l05)
    return softmax(scores) @ v                              # (B, T, d_h)
    """
    raise NotImplementedError("attention_single_head")


class MultiHeadAttention(Module):
    """n_heads heads of size d_h = d // n_heads.

    __init__: self.n_heads; self.qkv = Linear(d, 3*d, rng) then self.proj = Linear(d, d, rng).
    __call__(x (B, T, d), causal=True) -> (B, T, d):
        one fused projection to (B, T, 3d); split into q, k, v of shape (B, n_heads, T, d_h)
        (the 3d axis is laid out as [q | k | v], each [head 0 | head 1 | ...]);
        attention per head (scaled, masked if causal); put the heads side by side
        again to (B, T, d); apply self.proj.
    """

    def __init__(self, d: int, n_heads: int, rng: np.random.Generator):
        raise NotImplementedError("MultiHeadAttention")

    def __call__(self, x: Tensor, causal: bool = True) -> Tensor:
        raise NotImplementedError("MultiHeadAttention")
