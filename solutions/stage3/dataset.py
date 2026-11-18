"""Token files on disk (reference solution)."""

from pathlib import Path

import numpy as np

from .bpe import encode


def write_tokens(path, ids) -> None:
    """Save token ids as raw uint16 numbers (2 bytes each)."""
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= 2**16):
        raise ValueError("uint16 holds ids 0..65535 only; the vocabulary is too large")
    arr.astype(np.uint16).tofile(path)


def open_tokens(path) -> np.ndarray:
    """Open a token file without reading it into memory (a read-only memmap)."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def tokenize_file(src, dst, merges, block_chars: int = 2_000_000, log=print) -> int:
    """Given: encode a large text file block by block and write the ids to dst.
    Blocks end at a newline, so no chunk is cut in half. Returns the number of tokens."""
    src, dst = Path(src), Path(dst)
    total = 0
    with open(src, encoding="utf-8") as f, open(dst, "wb") as out:
        rest = ""
        while True:
            block = f.read(block_chars)
            if not block:
                break
            block = rest + block
            cut = block.rfind("\n") + 1 or len(block)
            block, rest = block[:cut], block[cut:]
            ids = np.asarray(encode(block, merges), dtype=np.uint16)
            ids.tofile(out)
            total += ids.size
            log(f"  {dst.name}: {total:,} tokens", end="\r")
        if rest:
            ids = np.asarray(encode(rest, merges), dtype=np.uint16)
            ids.tofile(out)
            total += ids.size
    log(f"  {dst.name}: {total:,} tokens")
    return total
