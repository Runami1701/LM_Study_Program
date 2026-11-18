"""Checks for s1-l01 — Vocabulary, encoding and batches."""

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import assert_equal, best_time, check_array, load, load_reference

MOD = "stage1.data"
ROOT = Path(__file__).resolve().parents[2]
SAMPLE = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n"
)


def _text():
    path = ROOT / "data" / "tinyshakespeare.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return SAMPLE


# ------------------------------------------------------------------ build_vocab

@pytest.mark.fn("build_vocab")
def test_build_vocab_hello():
    m = load(MOD)
    stoi, itos = m.build_vocab("hello")
    assert isinstance(stoi, dict), f"stoi should be a dict, got {type(stoi).__name__}"
    assert isinstance(itos, list), f"itos should be a list, got {type(itos).__name__}"
    assert itos == ["e", "h", "l", "o"], (
        f"for 'hello' expected itos ['e', 'h', 'l', 'o'] (sorted, each character once), got {itos}"
    )
    assert stoi == {"e": 0, "h": 1, "l": 2, "o": 3}, f"for 'hello' expected stoi {{'e': 0, 'h': 1, 'l': 2, 'o': 3}}, got {stoi}"


@pytest.mark.fn("build_vocab")
def test_build_vocab_sorted_unique_inverse():
    m = load(MOD)
    text = _text()
    stoi, itos = m.build_vocab(text)
    assert len(itos) == len(set(itos)), "itos contains a character more than once"
    assert itos == sorted(itos), "itos must be in sorted order, so ids are reproducible across runs"
    assert set(itos) == set(text), "every character of the text must be in the vocabulary (and nothing else)"
    for i, ch in enumerate(itos):
        assert stoi[ch] == i, f"stoi and itos disagree: itos[{i}] = {ch!r} but stoi[{ch!r}] = {stoi[ch]}"
    if len(text) > 1_000_000:
        assert len(itos) == 65, f"Tiny Shakespeare has V = 65 distinct characters, got {len(itos)}"


# ------------------------------------------------------------------- encode / decode

@pytest.mark.fn("encode")
def test_encode_hello():
    m = load(MOD)
    stoi = {"e": 0, "h": 1, "l": 2, "o": 3}
    ids = m.encode("hello", stoi)
    check_array(ids, "encode('hello')", shape=(5,), dtype=np.int64)
    assert_equal(ids, [1, 0, 2, 2, 3], "encode('hello')")


@pytest.mark.fn("encode")
def test_encode_empty():
    m = load(MOD)
    ids = m.encode("", {"a": 0})
    check_array(ids, "encode('')", shape=(0,), dtype=np.int64,
                hint="an empty string should give an empty int64 array")


@pytest.mark.fn("decode")
def test_decode_hello():
    m = load(MOD)
    itos = ["e", "h", "l", "o"]
    assert m.decode([1, 0, 2, 2, 3], itos) == "hello", "decode([1, 0, 2, 2, 3]) should be 'hello'"
    out = m.decode(np.array([1, 0, 2, 2, 3], dtype=np.int64), itos)
    assert out == "hello", f"decode should also accept a NumPy array; got {out!r}"
    assert isinstance(out, str), f"decode should return a str, got {type(out).__name__}"


@pytest.mark.fn("decode")
def test_round_trip_full_text():
    m = load(MOD)
    text = _text()
    stoi, itos = m.build_vocab(text)
    ids = m.encode(text, stoi)
    check_array(ids, "encode(text)", shape=(len(text),), dtype=np.int64)
    assert ids.min() >= 0 and ids.max() < len(itos), "ids must lie in [0, V)"
    assert m.decode(ids, itos) == text, "decode(encode(text)) must give back exactly the original text"


# ---------------------------------------------------------------- train_val_split

@pytest.mark.fn("train_val_split")
def test_split_sizes_and_order():
    m = load(MOD)
    data = np.arange(1000)
    train, val = m.train_val_split(data)
    assert len(train) == 900 and len(val) == 100, (
        f"frac=0.9 of 1000 should give 900 train / 100 val, got {len(train)} / {len(val)}"
    )
    assert_equal(np.concatenate([train, val]), data, "concatenate([train, val])",
                 hint="split by position: keep the original order, no shuffling")


@pytest.mark.fn("train_val_split")
def test_split_frac_argument():
    m = load(MOD)
    data = np.arange(1001)
    train, val = m.train_val_split(data, frac=0.75)
    n_train = int(0.75 * 1001)
    assert len(train) == n_train, f"frac=0.75 of 1001 should give int(750.75) = {n_train} training ids, got {len(train)}"
    assert len(train) + len(val) == 1001, "train and val together must contain every id exactly once"
    assert train[-1] + 1 == val[0], "val must start right where train ends"


# --------------------------------------------------------------------- get_batch

@pytest.mark.fn("get_batch")
def test_get_batch_shapes_and_dtypes():
    m = load(MOD)
    data = np.arange(100, dtype=np.int64)
    x, y = m.get_batch(data, 32, 8, np.random.default_rng(0))
    check_array(x, "x", shape=(32, 8), dtype=np.int64, shape_desc="(B, k)")
    check_array(y, "y", shape=(32,), dtype=np.int64, shape_desc="(B,)",
                hint="did you index with a slice instead of an integer?")


@pytest.mark.fn("get_batch")
def test_get_batch_from_uint8_gives_int64():
    m = load(MOD)
    data = (np.arange(300) % 65).astype(np.uint8)
    x, y = m.get_batch(data, 16, 4, np.random.default_rng(0))
    check_array(x, "x (from uint8 data)", shape=(16, 4), dtype=np.int64,
                hint="the data may be stored compactly, but the batch must be int64 for indexing")
    check_array(y, "y (from uint8 data)", shape=(16,), dtype=np.int64)


@pytest.mark.fn("get_batch")
def test_get_batch_windows_are_contiguous():
    """With data = [0, 1, 2, ...], every value equals its position, so each
    row of x must be a run of consecutive numbers and y the number after it."""
    m = load(MOD)
    k = 5
    data = np.arange(1000, dtype=np.int64)
    x, y = m.get_batch(data, 64, k, np.random.default_rng(3))
    starts = x[:, 0]
    assert_equal(x, starts[:, None] + np.arange(k)[None, :], "x",
                 hint="each row must be k consecutive characters data[s], data[s+1], ..., data[s+k-1]")
    assert_equal(y, starts + k, "y",
                 hint="y[r] must be the character right after row r's window, data[s + k]")


@pytest.mark.fn("get_batch")
def test_get_batch_matches_seeded_starts():
    m = load(MOD)
    text_ids = np.random.default_rng(7).integers(0, 65, size=5000)
    B, k = 128, 8
    x, y = m.get_batch(text_ids, B, k, np.random.default_rng(42))
    starts = np.random.default_rng(42).integers(0, len(text_ids) - k, size=B)
    for r in range(B):
        s = starts[r]
        if not np.array_equal(x[r], text_ids[s : s + k]) or y[r] != text_ids[s + k]:
            raise AssertionError(
                f"row {r}: expected the window starting at {s} (drawn by rng.integers(0, n - k, size=B)) — "
                "draw all B starts with exactly that one call so batches are reproducible"
            )


@pytest.mark.fn("get_batch")
def test_get_batch_uses_full_range():
    """With n = k + 2 there are exactly two valid starts, 0 and 1 (n - k = 2 pairs)."""
    m = load(MOD)
    k = 4
    data = np.arange(k + 2, dtype=np.int64)
    x, y = m.get_batch(data, 500, k, np.random.default_rng(0))
    seen = set(x[:, 0].tolist())
    assert seen == {0, 1}, (
        f"with n = {k + 2}, k = {k} the valid starts are {{0, 1}}; your batches used {sorted(seen)} — "
        "the largest valid start is n - k - 1, and rng.integers excludes its upper bound"
    )
    assert y.max() == k + 1, "the last character of the data should be reachable as a target"


@pytest.mark.slow
@pytest.mark.fn("get_batch")
def test_get_batch_is_vectorized():
    m = load(MOD)
    ref = load_reference(MOD)
    data = np.random.default_rng(0).integers(0, 65, size=1_115_394).astype(np.int64)
    B, k = 4096, 8
    yours = best_time(lambda: m.get_batch(data, B, k, np.random.default_rng(0)), repeats=3, number=5)
    theirs = best_time(lambda: ref.get_batch(data, B, k, np.random.default_rng(0)), repeats=3, number=5)
    # A Python loop over 4096 rows is ~10x slower than broadcasting (~3 ms vs ~0.3 ms
    # per call on a desktop). Pass outright under 0.8 ms per call; otherwise allow
    # up to 4x the reference so timing noise never fails a vectorized solution.
    if yours < 0.004:
        return
    ratio = yours / max(theirs, 1e-9)
    assert ratio < 4, (
        f"get_batch with B = 4096 is {ratio:.0f}x slower than a vectorized version "
        f"({yours / 5 * 1e3:.1f} ms per call) — is there a Python loop over the B rows? "
        "Build all windows at once with broadcasting"
    )
