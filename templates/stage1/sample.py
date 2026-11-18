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
