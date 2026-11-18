"""Lesson s1-l13 — Sampling.  ->  python -m course check s1-l13"""

import numpy as np

from .model import predict


def sample_next(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Draw one id from softmax(logits / temperature).

    Args:
        logits: (V,) scores (any float dtype).
        temperature: > 0. Small -> almost always the argmax; large -> almost uniform.
        rng: Generator; draw with a single rng call per sample.

    Returns:
        A Python int in [0, V). Must not produce nan even for temperature = 1e-3.
    """
    raise NotImplementedError("sample_next")


def sample(params: dict, itos: list, stoi: dict, prompt: str, n_chars: int,
           temperature: float, rng: np.random.Generator) -> str:
    """Generate n_chars characters that continue `prompt`.

    The context is the last k ids of the prompt (k = W1 rows / C columns), padded
    on the left with the id of "\\n" if the prompt is shorter than k. After each
    draw, slide the window: drop the oldest id, append the new one.

    Returns:
        Only the generated characters (length n_chars), not the prompt.
    """
    raise NotImplementedError("sample")


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
