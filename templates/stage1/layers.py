"""Stage 1 layers: forward passes (and, later, backward passes).

Lesson s1-l02: embedding_forward, flatten_context   ->  python -m course check s1-l02
Lesson s1-l03: linear_forward, tanh_forward          ->  python -m course check s1-l03
Lesson s1-l06: linear_backward, tanh_backward        ->  python -m course check s1-l06
Lesson s1-l07: embedding_backward                    ->  python -m course check s1-l07

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


def linear_backward(dout: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass of out = x @ W + b.

    Args:
        dout:  (B, m) gradient of the loss with respect to out.
        cache: (x, W, b) from linear_forward.

    Returns:
        dx: (B, n), dW: (n, m), db: (m,) — each the same shape as what it is the gradient of.
    """
    raise NotImplementedError("linear_backward")


def tanh_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """Backward pass of out = tanh(x).

    Args:
        dout:  gradient with respect to out (any shape).
        cache: the output of tanh_forward (same shape).

    Returns:
        dx, same shape as dout.
    """
    raise NotImplementedError("tanh_backward")


def embedding_backward(demb: np.ndarray, x: np.ndarray, V: int) -> np.ndarray:
    """Backward pass of emb = C[x].

    Args:
        demb: (B, k, e) gradient with respect to emb.
        x:    (B, k) the ids that were looked up.
        V:    vocabulary size (number of rows of C).

    Returns:
        dC: (V, e), same dtype as demb. Row i is the sum of demb over every
        position where x == i (zero for ids that never appear). Repeated ids
        must accumulate.
    """
    raise NotImplementedError("embedding_backward")
