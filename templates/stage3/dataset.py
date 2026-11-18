"""Lesson s3-l03 — Token files on disk.  ->  python -m course check s3-l03"""

from pathlib import Path

import numpy as np

from .bpe import encode


def write_tokens(path, ids) -> None:
    """Save the token ids as raw uint16 numbers (2 bytes each), in order, at `path`.
    Raise ValueError if any id is outside 0..65535 (it would not fit into uint16)."""
    raise NotImplementedError("write_tokens")


def open_tokens(path) -> np.ndarray:
    """Open a file written by write_tokens WITHOUT reading it into memory:
    return a read-only np.memmap of dtype uint16."""
    raise NotImplementedError("open_tokens")


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
