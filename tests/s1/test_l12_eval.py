"""Checks for s1-l12 — Evaluation and learning-rate decay."""

import numpy as np
import pytest

from tests.helpers import load


@pytest.mark.fn("estimate_loss")
def test_estimate_loss_is_mean_of_batches():
    t = load("stage1.train_mlp")
    model = load("stage1.model")
    data_mod = load("stage1.data")
    p = model.init_params(30, 4, 5, 16, np.random.default_rng(0))
    data = np.random.default_rng(1).integers(0, 30, 4000)
    got = t.estimate_loss(p, data, 16, 4, 7, np.random.default_rng(9))
    assert isinstance(got, float), f"estimate_loss should return a Python float, got {type(got).__name__}"
    rng = np.random.default_rng(9)
    losses = [model.forward(p, *data_mod.get_batch(data, 16, 4, rng))[0] for _ in range(7)]
    assert abs(got - np.mean(losses)) < 1e-6, (
        f"expected the mean of 7 batch losses ({np.mean(losses):.5f}), got {got:.5f} — draw exactly "
        "n_batches batches with get_batch(data, B, k, rng)"
    )
    again = t.estimate_loss(p, data, 16, 4, 7, np.random.default_rng(9))
    assert again == got, "same seed must give the same estimate"


@pytest.mark.fn("lr_at")
def test_lr_decay_endpoints_and_midpoint():
    t = load("stage1.train_mlp")
    cfg = {"lr": 1e-2, "lr_final": 1e-4, "steps": 101}
    assert abs(t.lr_at(0, cfg) - 1e-2) < 1e-12, "lr_at(0) must be config['lr']"
    assert abs(t.lr_at(100, cfg) - 1e-4) < 1e-10, (
        f"the last step (steps - 1 = 100) must use lr_final = 1e-4, got {t.lr_at(100, cfg):.3g}"
    )
    assert abs(t.lr_at(50, cfg) - 1e-3) < 1e-9, (
        f"exponential decay reaches the geometric mean sqrt(1e-2 * 1e-4) = 1e-3 halfway, got {t.lr_at(50, cfg):.3g}"
    )


@pytest.mark.fn("lr_at")
def test_lr_constant_without_final():
    t = load("stage1.train_mlp")
    cfg = {"lr": 3e-3, "lr_final": None, "steps": 10}
    assert all(t.lr_at(s, cfg) == 3e-3 for s in range(10)), "lr_final = None means a constant learning rate"
