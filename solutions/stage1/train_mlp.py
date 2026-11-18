"""Train the stage-1 MLP on Tiny Shakespeare (reference solution).

Run from the repository root:  python -m solutions.stage1.train_mlp
"""

import json
import time
from pathlib import Path

import numpy as np

from .data import build_vocab, encode, get_batch, train_val_split
from .model import backward, count_params, forward, init_params
from .optim import Adam, sgd_step

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "k": 8,              # context length
    "e": 24,             # embedding size
    "h": 512,            # hidden units
    "batch_size": 128,
    "steps": 40_000,
    "optimizer": "adam", # "sgd" or "adam"
    "lr": 3e-3,          # learning rate at step 0
    "lr_final": 3e-4,    # learning rate at the last step (exponential decay); None = constant
    "eval_every": 2_000,
    "eval_batches": 50,
    "seed": 1337,
    "data": "data/tinyshakespeare.txt",
    "out_dir": "runs/s1",
}


def estimate_loss(params: dict, data: np.ndarray, B: int, k: int, n_batches: int,
                  rng: np.random.Generator) -> float:
    """Mean loss over n_batches random batches drawn with get_batch(data, B, k, rng)."""
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, B, k, rng)
        loss, _ = forward(params, x, y)
        losses.append(loss)
    return float(np.mean(losses))


def lr_at(step: int, config: dict) -> float:
    """Exponential decay from lr to lr_final over the run."""
    lr0, lr1 = config["lr"], config.get("lr_final")
    if not lr1:
        return lr0
    frac = step / max(1, config["steps"] - 1)
    return lr0 * (lr1 / lr0) ** frac


def train(config: dict, train_data: np.ndarray, val_data: np.ndarray, V: int, log=print) -> dict:
    """Train from scratch; return params and loss histories."""
    rng = np.random.default_rng(config["seed"])
    k, B = config["k"], config["batch_size"]
    params = init_params(V, k, config["e"], config["h"], rng)
    opt = Adam(params, lr=config["lr"]) if config["optimizer"] == "adam" else None
    train_loss, val_loss = [], []
    for step in range(config["steps"]):
        lr = lr_at(step, config)
        x, y = get_batch(train_data, B, k, rng)
        loss, cache = forward(params, x, y)
        grads = backward(cache)
        if opt is not None:
            opt.lr = lr
            opt.step(grads)
        else:
            sgd_step(params, grads, lr)
        train_loss.append(loss)
        every = config.get("eval_every")
        if every and (step % every == 0 or step == config["steps"] - 1):
            v = estimate_loss(params, val_data, B, k, config["eval_batches"], rng)
            val_loss.append((step, v))
            log(f"step {step:6d}  lr {lr:.2e}  train {np.mean(train_loss[-every:]):.4f}  val {v:.4f}")
    return {"params": params, "train_loss": train_loss, "val_loss": val_loss}


def main(config: dict = CONFIG) -> dict:
    text = (ROOT / config["data"]).read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    train_data, val_data = train_val_split(data)
    V = len(itos)
    t0 = time.perf_counter()
    result = train(config, train_data, val_data, V)
    seconds = time.perf_counter() - t0
    params = result["params"]
    eval_rng = np.random.default_rng(0)
    final = {
        "val_loss": estimate_loss(params, val_data, 512, config["k"], 200, eval_rng),
        "train_loss": estimate_loss(params, train_data, 512, config["k"], 200, eval_rng),
        "seconds": round(seconds, 1),
        "steps": config["steps"],
        "n_params": count_params(params),
        "config": config,
    }
    out = ROOT / config["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "params.npz", **params, itos=np.array(itos))
    try:
        from .sample import sample
        final["sample"] = sample(params, itos, stoi, "\n", 300, 0.8, np.random.default_rng(0))
    except NotImplementedError:
        final["sample"] = None
    (out / "results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\nval loss {final['val_loss']:.4f}  train loss {final['train_loss']:.4f}  "
          f"({seconds / 60:.1f} min, {final['n_params']:,} params)")
    if final["sample"]:
        print("\n" + final["sample"])
    return final


if __name__ == "__main__":
    main()
