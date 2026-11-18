"""Checks for s3-l01 — Training BPE."""

import pytest

from tests.helpers import load

MOD = "stage3.bpe"


@pytest.mark.fn("merge_pair")
def test_merge_pair():
    m = load(MOD)
    assert m.merge_pair([97, 98, 99, 97, 98], (97, 98), 256) == [256, 99, 256], "replace every occurrence of the pair"
    assert m.merge_pair([97, 97, 97], (97, 97), 256) == [256, 97], (
        "occurrences must not overlap: scan left to right, and after a merge skip both ids"
    )
    assert m.merge_pair([1, 2], (2, 1), 9) == [1, 2], "a pair that doesn't occur leaves the list unchanged"
    assert m.merge_pair([], (1, 2), 9) == [], "an empty list stays empty"


@pytest.mark.fn("count_pairs")
def test_count_pairs():
    m = load(MOD)
    got = dict(m.count_pairs({(1, 2, 3): 2, (2, 3): 1, (5,): 7}))
    assert got == {(1, 2): 2, (2, 3): 3}, (
        f"expected {{(1, 2): 2, (2, 3): 3}}, got {got} — each pair counts as often as its chunk occurs"
    )


@pytest.mark.fn("train_bpe")
def test_toy_corpus_merges():
    m = load(MOD)
    merges = m.train_bpe("aaabdaaabac", 259)
    assert [tuple(p) for p in merges] == [(97, 97), (97, 98), (256, 257)], (
        f"for 'aaabdaaabac' expected merges aa, then ab, then (aa)(ab): [(97, 97), (97, 98), (256, 257)], got {merges} — "
        "on a tie, pick the smallest pair"
    )


@pytest.mark.fn("train_bpe")
def test_merge_count_and_chunk_boundaries():
    m = load(MOD)
    text = "the cat sat on the mat. the cat ate. " * 20
    merges = m.train_bpe(text, 270)
    assert len(merges) == 14, f"vocab_size 270 means 270 - 256 = 14 merges, got {len(merges)}"
    space, t = ord(" "), ord("t")
    vocab = m.build_vocab(merges)
    for tok in vocab[256:]:
        assert not tok.endswith(b" ") or tok.strip() == b"", (
            f"token {tok!r} glues a word to the space after it — merges must stay inside one chunk of pretokenize()"
        )
    assert len(m.train_bpe("ab", 300)) == 1, "stop early when no pair is left to merge"
