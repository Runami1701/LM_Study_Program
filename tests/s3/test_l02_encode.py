"""Checks for s3-l02 — Encode / decode."""

import pytest

from tests.helpers import load

MOD = "stage3.bpe"
TEXT = "Once upon a time, a little girl named Lily found a shiny red ball. She was happy!\n" * 30


def merges():
    from solutions.stage3.bpe import train_bpe  # test fixture: a fixed, known-good merge list
    return train_bpe(TEXT, 330)


@pytest.mark.fn("encode")
def test_encode_applies_merges_in_order():
    m = load(MOD)
    mm = [(97, 97), (256, 97)]          # aa -> 256, then (aa)a -> 257
    assert m.encode("aaa", mm) == [257], "'aaa' should become [257]: first aa -> 256, then (256, 97) -> 257"
    assert m.encode("aaaa", mm) == [256, 256], "'aaaa' should become [256, 256]: merge 256 is applied everywhere first"
    assert m.encode("", mm) == [], "the empty string encodes to []"
    ids = m.encode("ab", [])
    assert ids == [97, 98] and all(type(i) is int for i in ids), "with no merges, the ids are the UTF-8 bytes, as Python ints"


@pytest.mark.fn("decode")
def test_round_trip():
    m = load(MOD)
    mm = merges()
    for text in [TEXT[:200], "관동별곡", "emoji 🙂 and\nnew\n\nlines\ttabs", "", "Lily's ball—it's red."]:
        ids = m.encode(text, mm)
        back = m.decode(ids, mm)
        assert back == text, f"decode(encode(text)) must give back the text: {text!r} came back as {back!r}"


@pytest.mark.fn("encode")
def test_compression():
    m = load(MOD)
    mm = merges()
    ids = m.encode(TEXT, mm)
    ratio = len(TEXT.encode("utf-8")) / len(ids)
    print(f"compression: {ratio:.2f} bytes per token")
    assert ratio > 2.5, f"the learned merges should shrink this text a lot; got only {ratio:.2f} bytes per token"
