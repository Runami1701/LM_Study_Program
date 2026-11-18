"""Train the stage-1 MLP on Tiny Shakespeare.

Lesson s1-l10: train                 ->  python -m course check s1-l10
Lesson s1-l12: estimate_loss, lr_at  ->  python -m course check s1-l12

Run the full training from the repository root:
    python -m workspace.stage1.train_mlp
It writes runs/s1/results.json (read by `python -m course check s1-boss`).
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
    """Average loss over many random batches (s1-l12).

    Draw exactly n_batches batches with get_batch(data, B, k, rng), compute each
    batch's loss with forward(), and return the mean as a Python float.
    """
    raise NotImplementedError("estimate_loss")


def lr_at(step: int, config: dict) -> float:
    """Learning rate for this step (s1-l12).

    Until s1-l12 this just returns config["lr"]. In s1-l12, make it decay
    exponentially from config["lr"] at step 0 to config["lr_final"] at step
    config["steps"] - 1 (keep it constant when lr_final is None).
    """
    # TODO (s1-l12): exponential decay from lr to lr_final.
    return config["lr"]


def train(config: dict, train_data: np.ndarray, val_data: np.ndarray, V: int, log=print) -> dict:
    """Train a fresh MLP (s1-l10; s1-l11 adds Adam, s1-l12 adds evaluation).

    1. rng = np.random.default_rng(config["seed"]); params = init_params(V, k, e, h, rng).
    2. If config["optimizer"] == "adam", create Adam(params, lr=config["lr"]).
    3. For each step in range(config["steps"]):
         lr = lr_at(step, config)
         x, y = get_batch(train_data, config["batch_size"], k, rng)
         forward -> loss, backward -> grads
         update: sgd_step(params, grads, lr), or set opt.lr = lr and call opt.step(grads)
         append the loss to the train_loss list
         if config["eval_every"] is truthy and (step % eval_every == 0 or it is the last step):
             v = estimate_loss(params, val_data, B, k, config["eval_batches"], rng)
             append (step, v) to the val_loss list and log a line
    4. Return {"params": params, "train_loss": [floats], "val_loss": [(step, v), ...]}.
    """
    raise NotImplementedError("train")


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
