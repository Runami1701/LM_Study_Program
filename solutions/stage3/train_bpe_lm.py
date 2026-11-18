"""Train a subword (BPE) transformer on TinyStories (reference solution).

Run from the repository root:  python -m solutions.stage3.train_bpe_lm
"""

import gc
import json
import math
import time
from pathlib import Path

import numpy as np

from ..stage2.model import count_params
from ..stage2.train_tf import generate, train
from .bpe import decode, train_bpe
from .dataset import open_tokens, tokenize_file
from .metrics import evaluate_bpc

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "vocab_size": 4096, "bpe_train_chars": 10_000_000,
    "T": 256, "d": 256, "n_heads": 8, "n_layers": 6,
    "batch_size": 16,
    "max_steps": 6000,
    "warmup": 300,
    "lr_max": 6e-4, "lr_min": 6e-5,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_every": 500, "eval_batches": 10,
    "eval_chars": 300_000,
    "seed": 1337,
    "out_dir": "runs/s3",
    "resume": False,
}


def estimate_hours(N: int, D: int, flops_per_s: float) -> float:
    """Training time for N parameters and D tokens at about 6 N D FLOPs."""
    return 6.0 * N * D / flops_per_s / 3600.0


def tokens_for_hours(N: int, hours: float, flops_per_s: float) -> int:
    """How many tokens fit into `hours` of training (the inverse of estimate_hours)."""
    return int(hours * 3600.0 * flops_per_s / (6.0 * N))


def prepare(config: dict, log=print):
    """Given: learn the BPE merges once and tokenize the train and validation texts once."""
    out = ROOT / config["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    merges_path = out / "merges.json"
    train_txt = ROOT / "data" / "tinystories_train.txt"
    valid_txt = ROOT / "data" / "tinystories_valid.txt"
    if not train_txt.exists():
        raise FileNotFoundError("TinyStories is missing: run  python -m course data tinystories")
    if merges_path.exists():
        merges = [tuple(p) for p in json.loads(merges_path.read_text())]
    else:
        log(f"Training BPE ({config['vocab_size']} tokens) on {config['bpe_train_chars']:,} characters ...")
        with open(train_txt, encoding="utf-8") as f:
            sample = f.read(config["bpe_train_chars"])
        t0 = time.perf_counter()
        merges = train_bpe(sample, config["vocab_size"])
        log(f"  done in {time.perf_counter() - t0:.0f} s")
        merges_path.write_text(json.dumps(merges))
    for txt, name in ((train_txt, "train.bin"), (valid_txt, "valid.bin")):
        if not (out / name).exists():
            log(f"Tokenizing {txt.name} ...")
            tokenize_file(txt, out / name, merges, log=log)
    return merges, open_tokens(out / "train.bin"), open_tokens(out / "valid.bin")


def main(config: dict = CONFIG) -> dict:
    merges, train_data, val_data = prepare(config)
    V = 256 + len(merges)
    t0 = time.perf_counter()
    res = train(config, train_data, val_data, V)
    seconds = time.perf_counter() - t0
    model = res["model"]
    with open(ROOT / "data" / "tinystories_valid.txt", encoding="utf-8") as f:
        held_out = f.read(config["eval_chars"])
    bpc = evaluate_bpc(model, held_out, merges, config["T"])
    start = merges and [ord("O")]                       # stories often start with "Once"
    story_ids = generate(model, start, 200, 0.8, np.random.default_rng(0))
    sample = decode(story_ids, merges).split("<|endoftext|>")[0]
    tokens = config["max_steps"] * config["batch_size"] * config["T"]
    final = {
        "val_bpc": bpc,
        "val_loss": res["val_loss"][-1][1] if res["val_loss"] else None,
        "seconds": round(seconds, 1),
        "tokens_trained": tokens,
        "tokens_per_sec": round(tokens / seconds),
        "n_params": count_params(model),
        "config": config,
        "sample": sample,
    }
    out = ROOT / config["out_dir"]
    (out / "results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n{final['val_bpc']:.3f} bits per character on held-out stories ({seconds / 3600:.2f} h)\n")
    print(sample)
    gc.collect()
    return final


if __name__ == "__main__":
    main()
