"""Checks for s2-l06 — Multi-head attention."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load
from tests.s2.common import T, grad_check_module, ref_softmax, to64

MOD = "stage2.attention"


def ref_mha(x, Wqkv, bqkv, Wp, bp, h, causal=True):
    B, Tn, d = x.shape
    dh = d // h
    qkv = x @ Wqkv + bqkv
    q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
    heads = []
    for i in range(h):
        sl = slice(i * dh, (i + 1) * dh)
        s = q[..., sl] @ np.swapaxes(k[..., sl], -1, -2) / np.sqrt(dh)
        if causal:
            s = s + np.triu(np.full((Tn, Tn), -1e9), 1)
        heads.append(ref_softmax(s) @ v[..., sl])
    return np.concatenate(heads, axis=-1) @ Wp + bp


@pytest.mark.fn("MultiHeadAttention")
def test_structure():
    m = load(MOD)
    mha = m.MultiHeadAttention(16, 4, np.random.default_rng(0))
    names = set(mha.named_parameters())
    assert names == {"qkv.W", "qkv.b", "proj.W", "proj.b"}, f"expected parameters qkv.W/b and proj.W/b, got {sorted(names)}"
    assert mha.qkv.W.data.shape == (16, 48) and mha.proj.W.data.shape == (16, 16), "qkv: Linear(d, 3d); proj: Linear(d, d)"


@pytest.mark.fn("MultiHeadAttention")
def test_matches_per_head_loop():
    m = load(MOD)
    rng = np.random.default_rng(1)
    mha = m.MultiHeadAttention(16, 4, np.random.default_rng(2))
    for p in mha.parameters():
        p.data = (rng.standard_normal(p.data.shape) * 0.3).astype(np.float32)
    x = rng.standard_normal((2, 6, 16)).astype(np.float32)
    for causal in (True, False):
        out = mha(T()(x), causal=causal)
        check_array(out.data, "MHA output", shape=(2, 6, 16), dtype=np.float32, shape_desc="(B, T, d)")
        ref = ref_mha(x.astype(np.float64), mha.qkv.W.data, mha.qkv.b.data, mha.proj.W.data, mha.proj.b.data, 4, causal)
        assert_close(out.data, ref, f"MHA output (causal={causal})", atol=1e-5,
                     hint="split 3d as [q | k | v], each as [head 0 | head 1 | ...]; reshape (B,T,3,h,dh) then transpose to (3,B,h,T,dh)")


@pytest.mark.fn("MultiHeadAttention")
def test_gradcheck_tiny():
    m = load(MOD)
    rng = np.random.default_rng(3)
    mha = m.MultiHeadAttention(4, 2, np.random.default_rng(4))
    to64(mha)
    for p in mha.parameters():
        p.data[...] = rng.standard_normal(p.data.shape) * 0.5
    x = T()(rng.standard_normal((2, 3, 4)))
    G = rng.standard_normal((2, 3, 4))
    ops = load("autograd.ops")
    f = lambda: ops.sum(mha(x) * G)  # noqa: E731
    grad_check_module(f, {"x": x, **mha.named_parameters()}, n=6, tol=1e-5)
