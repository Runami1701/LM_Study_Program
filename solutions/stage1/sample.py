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


def main(temperature: float = 1.0, n_chars: int = 500, seed: int = 0) -> None:
    """Load runs/s1/params.npz (written by train_mlp) and print a sample."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    saved = np.load(root / "runs" / "s1" / "params.npz")
    params = {name: saved[name] for name in ("C", "W1", "b1", "W2", "b2")}
    itos = [str(c) for c in saved["itos"]]
    stoi = {c: i for i, c in enumerate(itos)}
    if len(sys.argv) > 1:
        temperature = float(sys.argv[1])
    print(sample(params, itos, stoi, "\n", n_chars, temperature, np.random.default_rng(seed)))


if __name__ == "__main__":
    main()
