"""The MLP character model, assembled from the layers (reference solution)."""

import numpy as np

from .layers import (
    embedding_backward, embedding_forward, flatten_context,
    linear_backward, linear_forward, tanh_backward, tanh_forward,
)
from .loss import cross_entropy, cross_entropy_backward


def init_params(V: int, k: int, e: int, h: int, rng: np.random.Generator) -> dict:
    """C ~ N(0,1); W1 ~ N(0,1)/sqrt(k*e); W2 ~ 0.01*N(0,1); zero biases. float32."""
    C = rng.standard_normal((V, e))
    W1 = rng.standard_normal((k * e, h)) / np.sqrt(k * e)
    W2 = rng.standard_normal((h, V)) * 0.01
    params = {"C": C, "W1": W1, "b1": np.zeros(h), "W2": W2, "b2": np.zeros(V)}
    return {name: p.astype(np.float32) for name, p in params.items()}


def predict(params: dict, x: np.ndarray) -> np.ndarray:
    """Logits (B, V) for contexts x (B, k)."""
    flat = flatten_context(embedding_forward(params["C"], x))
    h, _ = tanh_forward(linear_forward(flat, params["W1"], params["b1"])[0])
    logits, _ = linear_forward(h, params["W2"], params["b2"])
    return logits


def forward(params: dict, x: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    """Mean cross-entropy loss and everything backward() needs."""
    emb = embedding_forward(params["C"], x)              # (B, k, e)
    flat = flatten_context(emb)                          # (B, k*e)
    a, lin1 = linear_forward(flat, params["W1"], params["b1"])   # (B, h)
    h, act = tanh_forward(a)                             # (B, h)
    logits, lin2 = linear_forward(h, params["W2"], params["b2"]) # (B, V)
    loss, ce = cross_entropy(logits, y)
    V = params["C"].shape[0]
    return loss, (x, emb.shape, V, lin1, act, lin2, ce)


def backward(cache: tuple) -> dict:
    """Gradients of the loss for every parameter, same names as params."""
    x, emb_shape, V, lin1, act, lin2, ce = cache
    dlogits = cross_entropy_backward(ce)                 # (B, V)
    dh, dW2, db2 = linear_backward(dlogits, lin2)        # (B, h)
    da = tanh_backward(dh, act)                          # (B, h)
    dflat, dW1, db1 = linear_backward(da, lin1)          # (B, k*e)
    demb = dflat.reshape(emb_shape)                      # un-flatten: (B, k, e)
    dC = embedding_backward(demb, x, V)                  # (V, e)
    return {"C": dC, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def count_params(params: dict) -> int:
    """Total number of scalars in all parameter arrays."""
    return int(sum(p.size for p in params.values()))
