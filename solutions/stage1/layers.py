"""Stage 1 layers (reference solution)."""

import numpy as np


def embedding_forward(C: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(V, e) table, (B, k) ids -> (B, k, e) embeddings."""
    return C[x]  # integer-array indexing: one row of C per entry of x


def flatten_context(emb: np.ndarray) -> np.ndarray:
    """(B, k, e) -> (B, k*e), concatenating the k vectors of each example."""
    B = emb.shape[0]
    return emb.reshape(B, -1)  # C-order: the last axis (e) varies fastest


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, tuple]:
    """out = x @ W + b; cache = (x, W, b)."""
    out = x @ W + b  # (B, n) @ (n, m) -> (B, m); b (m,) broadcasts over the B rows
    return out, (x, W, b)


def tanh_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """out = tanh(x); cache = out (since d tanh/dx = 1 - tanh²)."""
    out = np.tanh(x)
    return out, out


def linear_backward(dout: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients of out = x @ W + b: (dx, dW, db)."""
    x, W, b = cache
    dx = dout @ W.T          # (B, m) @ (m, n) -> (B, n)
    dW = x.T @ dout          # (n, B) @ (B, m) -> (n, m)
    db = dout.sum(axis=0)    # b was broadcast over the B rows -> sum them back
    return dx, dW, db


def tanh_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """d tanh(a)/da = 1 - tanh(a)^2, and cache holds tanh(a)."""
    return dout * (1.0 - cache * cache)


def embedding_backward(demb: np.ndarray, x: np.ndarray, V: int) -> np.ndarray:
    """Scatter-add (B, k, e) gradients into the rows of a (V, e) table."""
    dC = np.zeros((V, demb.shape[-1]), dtype=demb.dtype)
    np.add.at(dC, x, demb)   # unbuffered: repeated ids accumulate
    return dC
