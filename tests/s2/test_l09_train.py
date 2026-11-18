"""Checks for s2-l09 — Train, checkpoint, sample."""

import numpy as np
import pytest

from tests.helpers import assert_close, load


def tiny(seed):
    return load("stage2.model").GPT(12, 16, 8, 2, 1, np.random.default_rng(seed))


@pytest.mark.fn("save_checkpoint")
def test_round_trip(tmp_path):
    mod = load("stage2.model")
    a, b = tiny(0), tiny(1)
    x = np.random.default_rng(2).integers(0, 12, (2, 16))
    mod.save_checkpoint(a, tmp_path / "ck.npz")
    assert (tmp_path / "ck.npz").exists(), "save_checkpoint should write the given path"
    W_before = b.head.W
    mod.load_checkpoint(b, tmp_path / "ck.npz")
    assert b.head.W is W_before, "load into the existing Tensors (p.data[...] = ...), don't replace them — the optimizer holds them"
    assert_close(b(x)[0].data, a(x)[0].data, "logits after save -> load", atol=0,
                 hint="every parameter, by its named_parameters() name")


@pytest.mark.fn("load_checkpoint")
def test_load_rejects_wrong_shapes(tmp_path):
    mod = load("stage2.model")
    big = load("stage2.model").GPT(12, 16, 16, 2, 1, np.random.default_rng(0))
    mod.save_checkpoint(big, tmp_path / "big.npz")
    with pytest.raises((ValueError, KeyError)):
        mod.load_checkpoint(tiny(0), tmp_path / "big.npz")


@pytest.mark.fn("generate")
def test_generate():
    tr = load("stage2.train_tf")
    model = tiny(0)
    ids = tr.generate(model, [1, 2, 3], 20, 1.0, np.random.default_rng(0))
    assert isinstance(ids, list) and len(ids) == 23 and ids[:3] == [1, 2, 3], "return the prompt ids followed by n new ids"
    assert all(0 <= int(i) < 12 for i in ids), "every id must be in [0, V)"
    assert ids == tr.generate(model, [1, 2, 3], 20, 1.0, np.random.default_rng(0)), "same seed, same output"
    long = tr.generate(model, list(range(12)) * 3, 2, 1.0, np.random.default_rng(0))
    assert len(long) == 38, "prompts longer than T must work: feed only the last T ids"


@pytest.mark.fn("train")
def test_train_smoke(tmp_path):
    tr = load("stage2.train_tf")
    data = np.arange(3000) % 12
    cfg = dict(T=16, d=16, n_heads=2, n_layers=1, batch_size=8, max_steps=60, warmup=5, lr_max=1e-2, lr_min=1e-3,
               weight_decay=0.0, grad_clip=1.0, eval_every=0, eval_batches=1, seed=0, out_dir=str(tmp_path), resume=False)
    res = tr.train(cfg, data, data, 12, log=lambda *a: None)
    h = res["train_loss"]
    assert len(h) == 60, f"one loss per step expected, got {len(h)}"
    assert np.mean(h[-5:]) < h[0] - 1.0, f"loss should fall a lot on a periodic sequence: {h[0]:.2f} -> {np.mean(h[-5:]):.2f}"
