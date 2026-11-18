"""Train the character-level transformer on Tiny Shakespeare (reference solution).

Run from the repository root:  python -m solutions.stage2.train_tf
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
    """Append n sampled ids to `ids`, feeding at most the last T ids each step."""
    ids = list(ids)
    for _ in range(n):
        context = np.array([ids[-model.T:]], dtype=np.int64)    # (1, t <= T)
        logits, _ = model(context)
        ids.append(sample_next(logits.data[0, -1], temperature, rng))   # last position predicts the next id
        del logits
        gc.collect()
    return ids


def train(config: dict, train_data: np.ndarray, val_data: np.ndarray, V: int, log=print) -> dict:
    rng = np.random.default_rng(config["seed"])
    model = GPT(V, config["T"], config["d"], config["n_heads"], config["n_layers"], rng)
    out = ROOT / config.get("out_dir", "runs/s2")
    ckpt = out / "ckpt.npz"
    if config.get("resume") and ckpt.exists():
        load_checkpoint(model, ckpt)
        log(f"resumed weights from {ckpt}")
    opt = AdamW(model.named_parameters(), lr=config["lr_max"], weight_decay=config["weight_decay"])
    B, T = config["batch_size"], config["T"]
    train_loss, val_loss = [], []
    t0 = time.perf_counter()
    for step in range(config["max_steps"]):
        lr = lr_at(step, config["warmup"], config["max_steps"], config["lr_max"], config["lr_min"])
        x, y = get_batch_seq(train_data, B, T, rng)
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        clip_grad_norm(model.parameters(), config["grad_clip"])
        opt.lr = lr
        opt.step()
        train_loss.append(float(loss.data))
        del loss
        gc.collect()   # free this step's graph (reference cycles) before building the next
        every = config.get("eval_every")
        if every and (step % every == 0 or step == config["max_steps"] - 1):
            v = estimate_loss(model, val_data, B, T, config["eval_batches"], rng)
            val_loss.append((step, v))
            tps = (step + 1) * B * T / (time.perf_counter() - t0)
            log(f"step {step:5d}  lr {lr:.2e}  train {np.mean(train_loss[-every:]):.4f}  val {v:.4f}  ({tps:,.0f} tok/s)")
            out.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, ckpt)
    return {"model": model, "train_loss": train_loss, "val_loss": val_loss}


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
