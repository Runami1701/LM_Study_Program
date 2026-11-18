"""Stage 1 layers: forward passes (and, later, backward passes).

Lesson s1-l02: embedding_forward, flatten_context   ->  python -m course check s1-l02
Lesson s1-l03: linear_forward, tanh_forward          ->  python -m course check s1-l03

Conventions: row vectors (x @ W), float32 model math, no Python loops over the batch.
"""

import numpy as np


def embedding_forward(C: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Look up the embedding of every context character.

    Args:
        C: (V, e) float32 embedding table; row i is the vector for id i.
        x: (B, k) integer ids.

    Returns:
        emb: (B, k, e) float32, emb[b, j] = C[x[b, j]].
    """
    raise NotImplementedError("embedding_forward")


def flatten_context(emb: np.ndarray) -> np.ndarray:
    """Concatenate the k embeddings of each example into one row.

    Args:
        emb: (B, k, e)

    Returns:
        (B, k*e), where row b is [emb[b, 0] | emb[b, 1] | ... | emb[b, k-1]].
    """
    raise NotImplementedError("flatten_context")


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Affine layer: out = x @ W + b.

    Args:
        x: (B, n) float32 inputs.
        W: (n, m) float32 weights.
        b: (m,)   float32 bias.

    Returns:
        out:   (B, m) float32.
        cache: the tuple (x, W, b), saved for the backward pass in s1-l06.
    """
    raise NotImplementedError("linear_forward")


def tanh_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise tanh.

    Args:
        x: any shape, float32.

    Returns:
        out:   same shape and dtype as x, out = tanh(x).
        cache: the array `out` itself (the backward pass in s1-l06 needs only it).
    """
    raise NotImplementedError("tanh_forward")
