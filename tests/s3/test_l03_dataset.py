"""Checks for s3-l03 — Token dataset on disk."""

import numpy as np
import pytest

from tests.helpers import assert_equal, check_array, load

MOD = "stage3.dataset"


@pytest.mark.fn("write_tokens")
def test_write_and_size(tmp_path):
    m = load(MOD)
    path = tmp_path / "t.bin"
    m.write_tokens(path, [1, 2, 65535, 300])
    assert path.stat().st_size == 8, f"4 ids as uint16 take 8 bytes; the file has {path.stat().st_size}"
    assert_equal(np.fromfile(path, dtype=np.uint16), [1, 2, 65535, 300], "ids read back")
    with pytest.raises(ValueError):
        m.write_tokens(tmp_path / "bad.bin", [70000])


@pytest.mark.fn("open_tokens")
def test_open_is_memmap(tmp_path):
    m = load(MOD)
    path = tmp_path / "t.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    data = m.open_tokens(path)
    assert isinstance(data, np.memmap), f"open_tokens should return an np.memmap (reads from disk on demand), got {type(data).__name__}"
    check_array(np.asarray(data), "open_tokens result", shape=(1000,), dtype=np.uint16)
    x, y = load("stage2.layers").get_batch_seq(data, 4, 16, np.random.default_rng(0))
    check_array(x, "a batch drawn from the memmap", shape=(4, 16), dtype=np.int64)
    assert_equal(y, x + 1, "targets from the memmap")
