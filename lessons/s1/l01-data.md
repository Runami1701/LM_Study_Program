---
id: s1-l01
title: Vocabulary, encoding and batches
stage: s1
order: 1
prereqs: [s0-l00]
files: [workspace/stage1/data.py]
functions: [build_vocab, encode, decode, train_val_split, get_batch]
tests: tests/s1/test_l01_data.py
est_minutes: 40
---

::: goal
Turn raw Tiny Shakespeare text into integer batches: `x` of shape `(B, k)` (the previous $k$ characters) and `y` of shape `(B,)` (the next character).
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $V$ | vocabulary size: the number of distinct characters (65 for Tiny Shakespeare) |
| $n$ | length of the encoded text (number of characters) |
| $k$ | context length: how many previous characters the model sees |
| $B$ | batch size: how many (x, y) examples per training step |
| $s$ | a starting position in the text; $s_r$ is the start of batch row $r$ |
| $r$ | a row of the batch, $r = 0, \dots, B-1$ |
| $j$ | a position inside a context window, $j = 0, \dots, k-1$ |
| $\text{idx}$ | the `(B, k)` array of text positions gathered into `x` |
| `stoi` / `itos` | "string to integer" / "integer to string" mappings |
:::

::: concept
A neural network only takes numbers, so every character needs an integer id. In a **character-level** model each character is one token: the model reads $k$ characters and predicts the next one.

### Vocabulary and encoding

The vocabulary is the set of distinct characters in the text, numbered $0, 1, \dots, V-1$. `stoi` maps a character to its id; `itos` (a list) maps an id back, so `itos[stoi[c]] == c`.

**Worked example.** Text `"hello"`, $k = 3$. Sorted vocabulary: `e, h, l, o` → ids `0, 1, 2, 3`.
Encoded: `h e l l o` → `[1, 0, 2, 2, 3]`.
Sliding a window of 3:

| x (context) | ids | y | id |
| --- | --- | --- | --- |
| h e l | [1, 0, 2] | l | 2 |
| e l l | [0, 2, 2] | o | 3 |

We **sort** the characters before numbering them so that the ids are the same every run (see the first think-question). For Tiny Shakespeare the sorted vocabulary starts with `'\n'`, `' '`, `'!'`, ... because those have the smallest Unicode code points.

### How many (x, y) pairs?

A window that starts at position $s$ uses the characters $s, s+1, \dots, s+k-1$ as context and $s+k$ as the target. The target must exist, so

$$
0 \le s \quad\text{and}\quad s + k \le n - 1
$$

$$
\Longrightarrow\quad s \in \{0, 1, \dots, n - k - 1\}
$$

$$
\Longrightarrow\quad \text{number of pairs} = n - k
$$

For `"hello"`: $5 - 3 = 2$, matching the table. For Tiny Shakespeare with $k = 8$: $1{,}115{,}394 - 8 = 1{,}115{,}386$ pairs.

The largest valid start is $n - k - 1$. `rng.integers(low, high)` draws from $[\text{low}, \text{high})$, **excluding** `high`, so `rng.integers(0, n - k)` produces exactly the valid starts $0, \dots, n-k-1$.

### Which dtype for the ids?

Ids lie in $[0, V)$. With $V = 65$ they would fit in `uint8` (0 to 255): Tiny Shakespeare takes 1.1 MB as `uint8` and 8.9 MB as `int64`. Either is tiny today, so `encode` returns `int64`, NumPy's native index type. But `get_batch` must accept any integer dtype and still **return** `int64`: in stage 3 the token stream is stored on disk as compact `uint16`, and the batches drawn from it must be ready for indexing.

### Train / validation split by position

We hold out data the model never trains on, to measure how well it generalizes. The first 90% of the text is for training, the last 10% for validation:

$$
n_{\text{train}} = \lfloor 0.9 \cdot n \rfloor = \lfloor 0.9 \cdot 1{,}115{,}394 \rfloor = \lfloor 1{,}003{,}854.6 \rfloor = 1{,}003{,}854
$$

leaving $1{,}115{,}394 - 1{,}003{,}854 = 111{,}540$ characters for validation. Here $\lfloor \cdot \rfloor$ rounds down, which is what Python's `int(...)` does for positive numbers.

Why a cut by position and not a random shuffle of all windows? Neighbouring windows overlap: the windows starting at $s$ and $s+1$ share $k - 1$ characters. Shuffle, and almost every validation window has a near-twin in training (second think-question), so validation loss would flatter the model.

### All pairs up front, or random starts per batch?

You could precompute every pair: `x` alone would be an $(n - k, k)$ array, $1.1\text{M} \times 8$ int64 numbers $\approx 71$ MB. Fine now, but in stage 3 the text is about 155M tokens with context 256: $155\text{M} \times 256 \times 2$ bytes $\approx 79$ GB. Instead, each batch draws $B$ random starts and gathers only $B \cdot k$ numbers from the one shared array. That scales to any text size.

### Gathering B windows at once (broadcasting)

Row $r$ of the batch needs the positions $s_r, s_r + 1, \dots, s_r + k - 1$. As a formula:

$$
\text{idx}_{r j} = s_r + j, \qquad r = 0, \dots, B-1, \quad j = 0, \dots, k-1
$$

That is an "outer sum" of a column of starts and a row of offsets. **Broadcasting** builds it without a loop: when two arrays meet in an elementwise operation, an axis of length 1 is stretched to match the other array's length on that axis.

- `starts` reshaped to a column: shape `(B, 1)`
- `offsets = [0, 1, ..., k-1]` as a row: shape `(1, k)`
- sum: `(B, 1) + (1, k)` → `idx` of shape `(B, k)`

With `"hello"`, $k = 3$ and starts `[0, 1]`:

$$
\begin{bmatrix} 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 & 1 & 2 \end{bmatrix}
= \begin{bmatrix} 0 & 1 & 2 \\ 1 & 2 & 3 \end{bmatrix}
$$

Indexing an array with an integer array returns an array **shaped like the index**, so `data[idx]` is `(B, k)`: here `[[1, 0, 2], [0, 2, 2]]`, the two contexts in the table. The targets sit at `starts + k = [3, 4]`, so `y = data[[3, 4]] = [2, 3]`. A Python loop over the $B$ rows gives the same result about ten times slower; the check includes a timing test that catches it.
:::

::: think
Q: Why sort the characters before numbering them?
A: `set()` order isn't guaranteed to be stable across runs or Python versions (string hashing is randomized per process). Sorting makes the ids reproducible, so a saved model still matches the vocabulary when you reload it.
:::

::: think
Q: Why split train/val by position instead of shuffling all windows first?
A: Neighbouring windows overlap ("hell" and "ello" share 3 characters). If you shuffle with a 90/10 split, a validation window's two neighbours are both in validation with probability only $0.1 \times 0.1 = 1\%$, so 99% of validation windows have a near-duplicate in training. Validation loss then looks better than real generalization.
:::

::: think
Q: Building all $n - k$ pairs up front vs sampling random starts per batch: what does each cost?
A: Up front: $(n-k) \cdot k$ integers (1.1M × 8 ≈ 9M numbers, fine now, but 155M tokens × 256 in stage 3 is ~79 GB). Random starts: only $B \cdot k$ numbers per batch, gathered from a single shared array. Random sampling scales.
:::

::: think
Q: With $n = 6$ and $k = 4$, list every valid start. What goes wrong if you draw starts with `rng.integers(0, n - k + 1)`?
A: Valid starts are $0$ and $1$ ($n - k = 2$ pairs). `rng.integers(0, 3)` can also return $2$, whose target index is $2 + 4 = 6$, one past the end: an `IndexError` (or, worse, a silent wrap-around if you ever use negative indices).
:::

::: hint level=1
`build_vocab`, `encode`, `decode` and `train_val_split` are each one or two lines of plain Python and slicing. For `get_batch`: a batch is just $B$ random starting positions. Everything else follows from those $B$ numbers.
:::

::: hint level=2
`sorted(set(text))` gives the vocabulary; `enumerate` numbers it. `np.array(..., dtype=np.int64)` or `np.fromiter` builds the id array; `"".join(...)` reverses it. For `get_batch`, pick the starts with `rng.integers`, then build all $B$ windows at once by **broadcasting** a column of starts (`starts[:, None]`) against a row of offsets (`np.arange(k)`). Index `data` with the result, and convert with `.astype(np.int64)`.
:::

::: hint level=3
```pseudo
get_batch(data, B, k, rng):
    n       ← length of data
    starts  ← B random ints in [0, n − k)      # one rng.integers call
    offsets ← [0, 1, ..., k−1]
    idx     ← starts as a column + offsets as a row   # (B, k)
    x       ← data at idx, as int64                   # (B, k)
    y       ← data at starts + k, as int64            # (B,)
    return x, y
```
:::

::: attempt
File: `workspace/stage1/data.py` — implement `build_vocab`, `encode`, `decode`, `train_val_split`, `get_batch`.
Check: `python -m course check s1-l01`
:::

::: solution
@include solutions/stage1/data.py#build_vocab

- `set(text)` keeps each character once; `sorted` puts them in code-point order, so ids are reproducible.
- `enumerate` pairs each character with its position in that order; the dict comprehension inverts the list.

@include solutions/stage1/data.py#encode

- `np.fromiter` fills a NumPy array straight from a generator, without building a temporary Python list; `count=len(text)` lets it allocate once. `np.array([stoi[c] for c in text], dtype=np.int64)` is equally correct.

@include solutions/stage1/data.py#decode

- `int(i)` accepts NumPy integers as well as Python ints; `"".join` is the fast way to glue many short strings.

@include solutions/stage1/data.py#train_val_split

- `int(frac * len(data))` rounds down, matching $\lfloor 0.9 n \rfloor$. Slices are views, so no data is copied.

@include solutions/stage1/data.py#get_batch

- `rng.integers(0, n - k, size=B)`: the upper bound is excluded, so the largest start is $n - k - 1$ and the largest target index is $n - 1$, the last character.
- `starts[:, None]` turns `(B,)` into a column `(B, 1)`; `offsets[None, :]` is a row `(1, k)`. Their sum broadcasts to `(B, k)`.
- `data[idx]` gathers all $B \cdot k$ ids in one vectorized operation; `.astype(np.int64)` guarantees the output dtype even if `data` is `uint8` or `uint16`.
- `data[starts + k]` reads one target per row, shape `(B,)`. Indexing with an integer array (not a slice) is what keeps it 1-D.
:::

::: breakit
Shuffle the windows before splitting, train in s1-l12, and compare val loss with the correct split. Expect the shuffled version to look better on "val" than it really is. You can see the leak today: list every 9-character window (8 context + 1 target) of the text, split them 90/10 once by position and once after a shuffle, and count how many validation windows also occur **exactly** in training. Expect roughly 36% for the position split (Shakespeare repeats itself: names, "First Citizen:") and about 43% for the shuffled split, on top of the 99% that share 8 characters with a training neighbour.
:::

::: recap
- Characters ↔ ids with sorted, reproducible mappings.
- A text of length $n$ gives $n - k$ (context, target) pairs; the largest valid start is $n - k - 1$.
- Position-based train/val split, and why shuffling leaks.
- Vectorized batch sampling with broadcasting, `(B, 1) + (1, k) → (B, k)`, no Python loops.
:::
