---
id: s3-l03
title: Token dataset on disk
stage: s3
order: 3
prereqs: [s3-l02]
files: [workspace/stage3/dataset.py]
functions: [write_tokens, open_tokens]
tests: tests/s3/test_l03_dataset.py
est_minutes: 30
---

::: goal
You will tokenize the training text once, save the ids as compact `uint16` numbers in a file, and read batches straight from that file with `np.memmap`, without loading it into memory.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| `uint16` | whole numbers from 0 to 65,535, stored in 2 bytes |
| $V$ | vocabulary size (4,096) |
| $N$ | number of tokens in the file |
| memmap | an array whose numbers stay in a file on disk; NumPy reads the parts you use |
:::

::: numpy
### Write raw numbers to a file, and map them back

`arr.tofile(path)` writes the numbers as raw bytes, with nothing else. `np.memmap(path, dtype=..., mode="r")` opens such a file as an array, without reading it into memory:

```python
np.array([1, 2, 65535], dtype=np.uint16).tofile("t.bin")
m = np.memmap("t.bin", dtype=np.uint16, mode="r")
print(m, len(m), m.dtype)        # [    1     2 65535] 3 uint16
```

The file stores no shape or dtype, so you must open it with the same dtype you wrote.

### Bytes per number

```python
print(np.arange(5).astype(np.uint16).nbytes, np.arange(5).nbytes)   # 10 40
```

Five `uint16` numbers take 10 bytes; five `int64` numbers take 40.
:::

::: concept
### Tokenize once, train many times

Encoding the 100 MB training text takes a minute or two. A training run reads hundreds of millions of tokens, so we tokenize once and save the result. `tokenize_file` (given) reads the text in blocks of 2 million characters, cuts each block at a line break so no chunk is split, encodes it, and appends the ids to the output file.

### Why `uint16`

The ids lie between 0 and $V - 1 = 4095$. `uint16` stores 0 to 65,535 in 2 bytes, so it fits, with room for vocabularies up to 65,536 tokens. Storage for our $N = 26.6$ million training tokens:

| dtype | bytes per token | file size |
| --- | --- | --- |
| `uint16` | 2 | 53 MB |
| `int64` | 8 | 213 MB |

`write_tokens` must refuse ids that don't fit: `np.array([70000]).astype(np.uint16)` silently wraps around to 4464, which would corrupt the data without any error.

### Reading from disk: memmap

`np.memmap` gives an array whose numbers stay in the file. When you index it, the operating system reads only the parts you touch, and keeps recently used parts in memory. To your code it looks like an ordinary array: `data[idx]` with an index table works, and so does `len(data)`.

So `get_batch_seq` from s2-l01 works unchanged on a memmap: it reads $B \times (T + 1)$ tokens per batch from random places in the file. Its `.astype(np.int64)` turns the `uint16` values into the `int64` ids the model needs, as s1-l01 planned. A file of any size, even far larger than your memory, can be trained on this way.
:::

::: think
Q: Your vocabulary had 70,000 tokens. What goes wrong with `uint16`, and what would you use instead?
A: Ids above 65,535 don't fit: converting wraps them around (70,000 becomes 4,464), and the model silently trains on wrong tokens. Use `uint32` (4 bytes) instead; `write_tokens` should refuse rather than wrap.
:::

::: think
Q: The file stores only raw numbers. What happens if you open a `uint16` file with `dtype=np.int64`?
A: NumPy reads every 8 bytes as one number, so 4 tokens become 1 meaningless large number, and the length shrinks by 4. Nothing raises an error. The reader must use the same dtype as the writer.
:::

::: hint level=1
`write_tokens` checks the range, converts to `uint16` and writes raw bytes. `open_tokens` maps the file back with the same dtype, read-only.
:::

::: hint level=2
`arr = np.asarray(ids, dtype=np.int64)`; raise `ValueError` if `arr.min() < 0` or `arr.max() >= 2**16`; then `arr.astype(np.uint16).tofile(path)`. For reading: `np.memmap(path, dtype=np.uint16, mode="r")`.
:::

::: hint level=3
```pseudo
write_tokens(path, ids):
    arr ← ids as an int64 array
    if any id < 0 or ≥ 65536: raise ValueError
    write arr as uint16 raw bytes to path

open_tokens(path):
    return a read-only uint16 memmap of path
```
:::

::: attempt
The editor below opens `workspace/stage3/dataset.py`: fill in `write_tokens` and `open_tokens`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage3/dataset.py#write_tokens

- The range check runs on `int64` values, before any conversion could wrap them around.
- `.tofile` writes exactly 2 bytes per id and nothing else.

@include solutions/stage3/dataset.py#open_tokens

- `mode="r"` opens the file read-only, so training can never change the data by accident.
:::

::: breakit
Run `np.array([70000]).astype(np.uint16)`. You get `array([4464], dtype=uint16)`, with no warning: $70{,}000 - 65{,}536 = 4{,}464$. That silent wrap-around is why `write_tokens` checks the range first.
:::

::: recap
- Tokenize once and save the ids as `uint16`: 2 bytes per token, enough for vocabularies up to 65,536.
- `np.memmap` reads batches straight from the file, so the data never has to fit in memory.
- `get_batch_seq` works unchanged on a memmap and returns `int64` batches.
:::
