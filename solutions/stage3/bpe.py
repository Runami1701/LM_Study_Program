"""Byte-pair encoding (reference solution)."""

import re
from collections import Counter

# Given: split text into chunks (words with their leading space, numbers, punctuation,
# whitespace). Merges never cross a chunk boundary.
CHUNK = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def pretokenize(text: str) -> list[str]:
    """Given: the chunks of `text`, in order. "".join(pretokenize(t)) == t."""
    return CHUNK.findall(text)


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` (left to right, no overlaps) by new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def count_pairs(words: dict) -> dict:
    """words: {tuple of ids: how often that chunk occurs}. Return {pair: total count}."""
    counts = Counter()
    for ids, n in words.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += n
    return counts


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int]]:
    """Learn vocab_size - 256 merges. Ties: the smallest pair wins."""
    words = Counter(tuple(chunk.encode("utf-8")) for chunk in pretokenize(text))
    merges = []
    for new_id in range(256, vocab_size):
        counts = count_pairs(words)
        if not counts:
            break                                           # every chunk is a single token
        best = min(counts, key=lambda p: (-counts[p], p))   # most frequent, then smallest
        merges.append(best)
        merged = Counter()
        for ids, n in words.items():
            merged[tuple(merge_pair(list(ids), best, new_id)) if best[0] in ids else ids] += n
        words = merged
    return merges


def build_vocab(merges: list) -> list[bytes]:
    """Given: the bytes of every token id."""
    vocab = [bytes([i]) for i in range(256)]
    for a, b in merges:
        vocab.append(vocab[a] + vocab[b])
    return vocab


def encode(text: str, merges: list) -> list[int]:
    """Text -> token ids, applying merges in the order they were learned."""
    rank = {pair: 256 + i for i, pair in enumerate(merges)}   # pair -> its new id (lower = earlier)
    cache, out = {}, []
    for chunk in pretokenize(text):
        if chunk not in cache:
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                pairs = set(zip(ids, ids[1:]))
                best = min(pairs, key=lambda p: rank.get(p, float("inf")))
                if best not in rank:
                    break                                   # no learned merge applies
                ids = merge_pair(ids, best, rank[best])
            cache[chunk] = ids
        out.extend(cache[chunk])
    return out


def decode(ids, merges: list) -> str:
    """Token ids -> text."""
    vocab = build_vocab(merges)
    return b"".join(vocab[int(i)] for i in ids).decode("utf-8", errors="replace")
