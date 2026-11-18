"""Train the character-level transformer on Tiny Shakespeare.

Lesson s2-l09: generate, train   ->  python -m course check s2-l09

Run the full training from the repository root (about 1.5 h; see the s2-boss lesson):
    python -m workspace.stage2.train_tf
It writes runs/s2/results.json and runs/s2/ckpt.npz. Set CONFIG["resume"] = True to
start from the last checkpoint after an interruption.
"""

import gc
import json
import os
import math
import time
from pathlib import Path

import numpy as np

from ..stage1.data import build_vocab, encode, train_val_split
from ..stage1.sample import sample_next
from .layers import get_batch_seq
from .model import GPT, count_params, load_checkpoint, save_checkpoint
from .optim import AdamW, clip_grad_norm
from .schedule import lr_at

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "T": 128, "d": 128, "n_heads": 4, "n_layers": 4,
    "batch_size": 32,
    "max_steps": 5000,
    "warmup": 200,
    "lr_max": 1e-3, "lr_min": 1e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_every": 250, "eval_batches": 20,
    "seed": 1337,
    "data": "data/tinyshakespeare.txt",
    "out_dir": "runs/s2",
    "resume": False,
}


def estimate_loss(model: GPT, data: np.ndarray, B: int, T: int, n_batches: int, rng) -> float:
    """Given: mean loss over n_batches random (B, T) batches."""
    losses = []
    for _ in range(n_batches):
        x, y = get_batch_seq(data, B, T, rng)
        _, loss = model(x, y)
        losses.append(float(loss.data))
        del loss
        gc.collect()   # each graph is a reference cycle (out -> _backward -> out): free it now
    return float(np.mean(losses))


def generate(model: GPT, ids: list, n: int, temperature: float, rng) -> list:
    """Return ids extended by n sampled ids.

    Each step: run the model on (at most) the last model.T ids as a (1, t) batch,
    take the logits at the LAST position, draw the next id with your stage-1
    sample_next(logits, temperature, rng), append it. Call gc.collect() after each
    step (see train).
    """
    raise NotImplementedError("generate")


def train(config: dict, train_data: np.ndarray, val_data: np.ndarray, V: int, log=print) -> dict:
    """Train a fresh GPT; return {"model": model, "train_loss": [floats], "val_loss": [(step, v)]}.

    1. rng = np.random.default_rng(config["seed"]); model = GPT(V, T, d, n_heads, n_layers, rng).
       If config.get("resume") and runs/<out_dir>/ckpt.npz exists, load_checkpoint into it.
    2. opt = AdamW(model.named_parameters(), lr=config["lr_max"], weight_decay=config["weight_decay"]).
    3. For each step in range(config["max_steps"]):
         lr = lr_at(step, warmup, max_steps, lr_max, lr_min)
         x, y = get_batch_seq(train_data, B, T, rng)
         model.zero_grad(); _, loss = model(x, y); loss.backward()
         clip_grad_norm(model.parameters(), config["grad_clip"])
         opt.lr = lr; opt.step()
         record float(loss.data); then `del loss` and gc.collect()  <- frees the graph:
             every node's _backward closure refers to the node itself (a reference cycle),
             so without this, memory grows by hundreds of MB per step.
         every config["eval_every"] steps (and on the last step): estimate_loss on val_data
             with config["eval_batches"], log it, save_checkpoint to <out_dir>/ckpt.npz.
    """
    raise NotImplementedError("train")


def main(config: dict = CONFIG) -> dict:
    if os.environ.get("COURSE_RESUME"):          # the website's "Continue from its checkpoint" button
        config = dict(config, resume=True)
    text = (ROOT / config["data"]).read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    train_data, val_data = train_val_split(encode(text, stoi))
    V = len(itos)
    t0 = time.perf_counter()
    res = train(config, train_data, val_data, V)
    seconds = time.perf_counter() - t0
    model = res["model"]
    val = estimate_loss(model, val_data, 32, config["T"], 100, np.random.default_rng(0))
    sample_ids = generate(model, [stoi["\n"]], 400, 0.8, np.random.default_rng(0))
    final = {
        "val_loss": val,
        "val_bpc": val / math.log(2),
        "train_loss": float(np.mean(res["train_loss"][-200:])),
        "seconds": round(seconds, 1),
        "tokens_per_sec": round(config["max_steps"] * config["batch_size"] * config["T"] / seconds),
        "n_params": count_params(model),
        "config": config,
        "sample": "".join(itos[i] for i in sample_ids[1:]),
    }
    out = ROOT / config["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out / "ckpt.npz")
    (out / "results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\nval loss {val:.4f} ({final['val_bpc']:.3f} bits/char) in {seconds / 60:.1f} min\n")
    print(final["sample"])
    return final


if __name__ == "__main__":
    main()
