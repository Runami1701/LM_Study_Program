"""Checks for s1-l10 — Training loop + SGD."""

import numpy as np
import pytest

from tests.helpers import load

TINY = dict(k=3, e=8, h=32, batch_size=32, steps=200, optimizer="sgd", lr=0.5, lr_final=None,
            eval_every=0, eval_batches=1, seed=0)


@pytest.mark.fn("sgd_step")
def test_sgd_step_in_place():
    m = load("stage1.optim")
    W = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([0.5], dtype=np.float32)
    params = {"W": W, "b": b}
    out = m.sgd_step(params, {"W": np.array([[1.0, -1.0]], np.float32), "b": np.array([2.0], np.float32)}, 0.1)
    assert out is None, "sgd_step updates in place and returns nothing"
    assert params["W"] is W and params["b"] is b, (
        "sgd_step must modify the arrays in place (p -= ...), not put new arrays in the dict — "
        "an optimizer holding references to the old arrays would stop seeing updates"
    )
    np.testing.assert_allclose(W, [[0.9, 2.1]], rtol=1e-6, err_msg="W after one step with lr 0.1")
    np.testing.assert_allclose(b, [0.3], rtol=1e-6, err_msg="b after one step with lr 0.1")
    assert W.dtype == np.float32, "the update must keep float32"


@pytest.mark.fn("train")
def test_train_loss_decreases():
    t = load("stage1.train_mlp")
    data = np.arange(6000) % 20          # perfectly predictable sequence
    res = t.train(dict(TINY), data, data, 20, log=lambda *a: None)
    assert isinstance(res, dict) and {"params", "train_loss", "val_loss"} <= set(res), (
        "train should return a dict with keys params, train_loss, val_loss"
    )
    h = res["train_loss"]
    assert len(h) == 200, f"expected one train loss per step (200), got {len(h)}"
    assert abs(h[0] - np.log(20)) < 0.2, f"the first loss should be near ln 20 = 3.0, got {h[0]:.3f}"
    assert np.mean(h[-20:]) < h[0] - 1.5, (
        f"after 200 SGD steps the loss went from {h[0]:.3f} to {np.mean(h[-20:]):.3f} — it should fall by well over 1.5"
    )


@pytest.mark.fn("train")
def test_train_is_reproducible():
    t = load("stage1.train_mlp")
    data = np.random.default_rng(5).integers(0, 10, 3000)
    cfg = dict(TINY, steps=30)
    a = t.train(dict(cfg), data, data, 10, log=lambda *a: None)["train_loss"]
    b = t.train(dict(cfg), data, data, 10, log=lambda *a: None)["train_loss"]
    assert np.allclose(a, b), "two runs with the same seed must give identical losses — use one rng from config['seed'] for init and batches"
