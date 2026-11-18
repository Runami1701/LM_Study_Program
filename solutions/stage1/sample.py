"""Sampling text from the trained MLP (reference solution)."""

import numpy as np

from .model import predict


def sample_next(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Draw one id from softmax(logits / temperature)."""
    z = logits.astype(np.float64) / temperature        # float64: tiny temperatures stay exact
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def sample(params: dict, itos: list, stoi: dict, prompt: str, n_chars: int,
           temperature: float, rng: np.random.Generator) -> str:
    """Generate n_chars characters after `prompt` (prompt not included)."""
    k = params["W1"].shape[0] // params["C"].shape[1]
    pad = stoi.get("\n", 0)
    context = ([pad] * k + [stoi[c] for c in prompt])[-k:]   # last k ids, left-padded
    out = []
    for _ in range(n_chars):
        logits = predict(params, np.array([context], dtype=np.int64))[0]   # (V,)
        nxt = sample_next(logits, temperature, rng)
        out.append(itos[nxt])
        context = context[1:] + [nxt]                         # slide the window
    return "".join(out)
