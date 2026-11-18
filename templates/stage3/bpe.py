"""Byte-pair encoding (BPE).

Lesson s3-l01: merge_pair, count_pairs, train_bpe   ->  python -m course check s3-l01
Lesson s3-l02: encode, decode                       ->  python -m course check s3-l02

Tokens 0..255 are the 256 possible bytes. Merge number k creates token 256 + k.
"""

import re
from collections import Counter

# Given: split text into chunks (words with their leading space, numbers, punctuation,
# whitespace). Merges never cross a chunk boundary.
CHUNK = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def pretokenize(text: str) -> list[str]:
    """Given: the chunks of `text`, in order. "".join(pretokenize(t)) == t."""
    return CHUNK.findall(text)


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Return a new list where every occurrence of `pair` is replaced by new_id.
    Scan left to right; occurrences do not overlap.

    Example: merge_pair([97, 97, 97], (97, 97), 256) returns [256, 97].
    """
    raise NotImplementedError("merge_pair")


def count_pairs(words: dict) -> dict:
    """words maps a tuple of ids (one chunk) to how often that chunk occurs.
    Return a dict mapping each adjacent pair to its total count.

    Example: count_pairs({(1, 2, 3): 2, (2, 3): 1}) returns {(1, 2): 2, (2, 3): 3}.
    """
    raise NotImplementedError("count_pairs")


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int]]:
    """Learn vocab_size - 256 merges from `text` (fewer if nothing is left to merge).

    1. Count the chunks: words = {tuple of the chunk's UTF-8 bytes: count}.
    2. Repeat: count all pairs; pick the most frequent pair (on a tie, the smallest
       pair, e.g. (97, 98) before (256, 97)); record it as merge k with new id 256 + k;
       replace it in every chunk.
    Return the list of merged pairs, in the order they were learned.
    """
    raise NotImplementedError("train_bpe")


def build_vocab(merges: list) -> list[bytes]:
    """Given: the bytes of every token id."""
    vocab = [bytes([i]) for i in range(256)]
    for a, b in merges:
        vocab.append(vocab[a] + vocab[b])
    return vocab


def encode(text: str, merges: list) -> list[int]:
    """Turn text into token ids (Python ints).

    For each chunk of pretokenize(text): start from its UTF-8 bytes, then keep applying
    the learned merge with the LOWEST number among the pairs present, until no learned
    merge applies. Encoding each distinct chunk once (a dict as a cache) makes this fast.
    """
    raise NotImplementedError("encode")


def decode(ids, merges: list) -> str:
    """Turn token ids back into text: join the bytes of every token, then decode UTF-8
    (use errors="replace" so a cut-off multi-byte character can't crash)."""
    raise NotImplementedError("decode")
