"""Lesson s2-l07 — Transformer block.  ->  python -m course check s2-l07"""

import numpy as np

from .attention import MultiHeadAttention
from .layers import LayerNorm, Linear, Module, gelu


class MLP(Module):
    """self.fc = Linear(d, 4d, rng) then self.proj = Linear(4d, d, rng); proj(gelu(fc(x)))."""

    def __init__(self, d: int, rng: np.random.Generator):
        raise NotImplementedError("MLP")

    def __call__(self, x):
        raise NotImplementedError("MLP")


class Block(Module):
    """Pre-LN residual block. Create, in this order: self.ln1 = LayerNorm(d),
    self.attn = MultiHeadAttention(d, n_heads, rng), self.ln2 = LayerNorm(d), self.mlp = MLP(d, rng).

    __call__: x = x + attn(ln1(x)); x = x + mlp(ln2(x)); return x
    """

    def __init__(self, d: int, n_heads: int, rng: np.random.Generator):
        raise NotImplementedError("Block")

    def __call__(self, x):
        raise NotImplementedError("Block")
