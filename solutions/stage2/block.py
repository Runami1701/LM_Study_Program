"""Transformer block (reference solution)."""

import numpy as np

from .attention import MultiHeadAttention
from .layers import LayerNorm, Linear, Module, gelu


class MLP(Module):
    """d -> 4d -> GELU -> d."""

    def __init__(self, d: int, rng: np.random.Generator):
        self.fc = Linear(d, 4 * d, rng)
        self.proj = Linear(4 * d, d, rng)

    def __call__(self, x):
        return self.proj(gelu(self.fc(x)))


class Block(Module):
    """Pre-LN residual block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self, d: int, n_heads: int, rng: np.random.Generator):
        self.ln1 = LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, rng)
        self.ln2 = LayerNorm(d)
        self.mlp = MLP(d, rng)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
