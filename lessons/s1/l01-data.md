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
jobs: [data-shakespeare]
---

::: goal
You will turn the Tiny Shakespeare text into numbers the model can train on. Each training example is `x`, the previous $k$ characters, and `y`, the character that comes next. A batch stacks $B$ examples: `x` has shape `(B, k)` and `y` has shape `(B,)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $V$ | vocabulary size: the number of different characters (65 for Tiny Shakespeare) |
| $n$ | length of the text, in characters |
| $k$ | context length: how many previous characters the model sees |
| $B$ | batch size: how many examples one training step uses |
| $s$ | a starting position in the text; $s_r$ is the start of batch row $r$ |
| $r$ | a row of the batch, $r = 0, \dots, B-1$ |
| $j$ | a position inside a window, $j = 0, \dots, k-1$ |
| $\text{idx}$ | the `(B, k)` table of text positions that become `x` |
| `stoi` / `itos` | "string to integer" / "integer to string": the two lookup tables |
:::

::: numpy
### Python tools for the vocabulary

```python
print(sorted(set("hello")))               # ['e', 'h', 'l', 'o']   each character once, in order
chars = ['e', 'h', 'l', 'o']
print({c: i for i, c in enumerate(chars)})   # {'e': 0, 'h': 1, 'l': 2, 'o': 3}
print("".join(['h', 'i', '!']))           # hi!   glue a list of strings together
```

`enumerate(chars)` gives the pairs `(0, 'e'), (1, 'h'), ...`. The `{c: i for ...}` line builds a dictionary from them.

### Random whole numbers

`rng.integers(low, high, size=B)` draws `B` whole numbers from `low` up to, **but not including**, `high`:

```python
rng = np.random.default_rng(0)
print(rng.integers(0, 10, size=4))        # [8 6 5 2]   four numbers from 0..9
```

### Tools from the NumPy lessons

- Slices `data[:m]` and `data[m:]` split an array in two (n-l03).
- `starts[:, None] + np.arange(k)[None, :]` builds a table of positions (n-l04).
- `data[idx]` reads the text at every position in a table `idx` (n-l03).
- `.astype(np.int64)` converts to 64-bit whole numbers (n-l01).
:::

::: concept
### Characters become numbers

A neural network computes with numbers, so every character gets a whole number, its **id**. Here each character is one token: the model reads $k$ characters and predicts the next one.

The vocabulary is the list of different characters in the text, numbered $0, 1, \dots, V-1$. `stoi` maps a character to its id. `itos` is a list that maps an id back to its character, so `itos[stoi[c]] == c`.

**Worked example.** Text `"hello"`, $k = 3$. The sorted vocabulary is `e, h, l, o`, with ids `0, 1, 2, 3`. The text becomes `h e l l o` → `[1, 0, 2, 2, 3]`. A window of 3 characters slides along the text:

| x (context) | ids | y | id |
| --- | --- | --- | --- |
| h e l | [1, 0, 2] | l | 2 |
| e l l | [0, 2, 2] | o | 3 |

We sort the characters before numbering them, so the ids come out the same in every run (see the first think-question). For Tiny Shakespeare the sorted vocabulary starts with `'\n'`, `' '`, `'!'`: those characters have the smallest codes.

### How many examples does a text give?

A window that starts at position $s$ uses the characters $s, s+1, \dots, s+k-1$ as context and the character at $s+k$ as the target. The target must exist, so

$$
0 \le s \quad\text{and}\quad s + k \le n - 1
$$

$$
\Longrightarrow\quad s \in \{0, 1, \dots, n - k - 1\}
$$

$$
\Longrightarrow\quad \text{number of examples} = n - k
$$

For `"hello"`: $5 - 3 = 2$, as in the table. For Tiny Shakespeare with $k = 8$: $1{,}115{,}394 - 8 = 1{,}115{,}386$ examples.

The largest valid start is $n - k - 1$. `rng.integers(0, n - k)` leaves out its upper limit, so it draws exactly the valid starts $0, \dots, n-k-1$.

### Which number type for the ids?

All ids lie between 0 and $V - 1 = 64$, so they would fit into `uint8`, which stores whole numbers from 0 to 255 in 1 byte. The whole text would take 1.1 MB as `uint8` or 8.9 MB as `int64`. Both are small, so `encode` returns `int64`, the type NumPy uses for positions. But `get_batch` must accept any whole-number type and still **return** `int64`. In Stage 3 the text is stored as compact `uint16`, and batches drawn from it must be ready for lookups.

### Training and validation data

We keep part of the text away from training, to measure how well the model handles text it has never seen. The first 90% is for training, the last 10% for validation:

$$
n_{\text{train}} = \lfloor 0.9 \cdot n \rfloor = \lfloor 0.9 \cdot 1{,}115{,}394 \rfloor = \lfloor 1{,}003{,}854.6 \rfloor = 1{,}003{,}854
$$

$\lfloor \cdot \rfloor$ rounds down, which is what Python's `int(...)` does for positive numbers. That leaves $1{,}115{,}394 - 1{,}003{,}854 = 111{,}540$ characters for validation.

We cut the text at one position instead of shuffling all windows. Neighbouring windows overlap: the windows that start at $s$ and $s+1$ share $k - 1$ characters. After a shuffle, almost every validation window would have a near-copy in the training data, and the validation loss would look better than the model really is (second think-question).

### Draw random windows instead of storing them all

You could store every example in advance. `x` alone would be an $(n - k) \times k$ array: $1.1\text{M} \times 8$ int64 numbers, about 71 MB. In Stage 3 the text has about 155M tokens and the context is 256, which would take about 79 GB. So each batch instead draws $B$ random starts and reads only $B \cdot k$ numbers from the one text array. This works for any text size.

### Building all B windows at once

Row $r$ of the batch needs the positions $s_r, s_r + 1, \dots, s_r + k - 1$:

$$
\text{idx}_{r j} = s_r + j, \qquad r = 0, \dots, B-1, \quad j = 0, \dots, k-1
$$

This is the outer sum from n-l04: a column of starts plus a row of offsets.

- `starts` as a column: shape `(B, 1)`
- `offsets = [0, 1, ..., k-1]` as a row: shape `(1, k)`
- their sum: `idx`, shape `(B, k)`

With `"hello"`, $k = 3$ and starts `[0, 1]`:

$$
\begin{bmatrix} 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 & 1 & 2 \end{bmatrix}
= \begin{bmatrix} 0 & 1 & 2 \\ 1 & 2 & 3 \end{bmatrix}
$$

`data[idx]` reads the text at all these positions and has the shape of `idx`: here `[[1, 0, 2], [0, 2, 2]]`, the two contexts from the table. The targets sit at `starts + k = [3, 4]`, so `y = data[[3, 4]] = [2, 3]`.

A Python loop over the $B$ rows gives the same result but runs about ten times slower. The check includes a timing test that catches such a loop.
:::

::: think
Q: Why sort the characters before numbering them?
A: `set()` does not keep a fixed order: it can differ between runs and Python versions. Sorting gives the same ids every time, so a saved model still matches its vocabulary when you load it again.
:::

::: think
Q: Why cut training and validation data at one position instead of shuffling all windows first?
A: Neighbouring windows overlap ("hell" and "ello" share 3 characters). After a 90/10 shuffle, both neighbours of a validation window are also in validation with probability only $0.1 \times 0.1 = 1\%$. So 99% of validation windows have a near-copy in training, and the validation loss looks better than the model really is.
:::

::: think
Q: Storing all $n - k$ examples in advance, or drawing random starts per batch: what does each cost?
A: Storing: $(n-k) \cdot k$ numbers (about 9M now, but about 79 GB in Stage 3). Drawing: only $B \cdot k$ numbers per batch, read from the one text array. Drawing scales to any text size.
:::

::: think
Q: With $n = 6$ and $k = 4$, list every valid start. What goes wrong with `rng.integers(0, n - k + 1)`?
A: The valid starts are 0 and 1 ($n - k = 2$ examples). `rng.integers(0, 3)` can also return 2. Its target position is $2 + 4 = 6$, one past the end of the text, so NumPy raises an `IndexError`.
:::

::: hint level=1
`build_vocab`, `encode`, `decode` and `train_val_split` each need one or two lines of plain Python and slicing. For `get_batch`: a batch is $B$ random starting positions. Everything else follows from those $B$ numbers.
:::

::: hint level=2
`sorted(set(text))` gives the vocabulary; `enumerate` numbers it. `np.array(..., dtype=np.int64)` or `np.fromiter` builds the id array; `"".join(...)` turns ids back into text. For `get_batch`: draw the starts with `rng.integers`, then build all windows at once with `starts[:, None] + np.arange(k)[None, :]`. Read `data` at those positions and convert with `.astype(np.int64)`.
:::

::: hint level=3
```pseudo
get_batch(data, B, k, rng):
    n       ← length of data
    starts  ← B random whole numbers in [0, n − k)   # one call: rng.integers(0, n − k, size=B)
    offsets ← [0, 1, ..., k−1]
    idx     ← starts as a column + offsets as a row  # (B, k)
    x       ← data at idx, as int64                  # (B, k)
    y       ← data at starts + k, as int64           # (B,)
    return x, y
```
:::

::: attempt
The editor below opens `workspace/stage1/data.py`: fill in `build_vocab`, `encode`, `decode`, `train_val_split` and `get_batch`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage1/data.py#build_vocab

- `set(text)` keeps each character once; `sorted` puts them in a fixed order.
- `enumerate` pairs each character with its position in that order; the dictionary maps back the other way.

@include solutions/stage1/data.py#encode

- `np.fromiter` fills a NumPy array straight from a generator, without building a Python list first. `np.array([stoi[c] for c in text], dtype=np.int64)` is also correct.

@include solutions/stage1/data.py#decode

- `int(i)` accepts NumPy whole numbers as well as Python ints. `"".join` glues many short strings quickly.

@include solutions/stage1/data.py#train_val_split

- `int(frac * len(data))` rounds down, like $\lfloor 0.9 n \rfloor$. Slices are views, so nothing is copied.

@include solutions/stage1/data.py#get_batch

- `rng.integers(0, n - k, size=B)` leaves out its upper limit, so the largest start is $n - k - 1$ and the largest target position is $n - 1$, the last character.
- `starts[:, None]` is a column `(B, 1)`; `offsets[None, :]` is a row `(1, k)`; their sum is `(B, k)`.
- `data[idx]` reads all $B \cdot k$ ids in one operation; `.astype(np.int64)` fixes the output type even when `data` is `uint8` or `uint16`.
- `data[starts + k]` reads one target per row, shape `(B,)`.
:::

::: breakit
List every 9-character window of the text (8 context + 1 target). Split the windows 90/10 once by position and once after a shuffle, and count how many validation windows also appear **exactly** in the training part. You get about 36% for the position split (Shakespeare repeats names and lines such as "First Citizen:") and about 43% after the shuffle. On top of that, 99% of the shuffled validation windows share 8 characters with a training neighbour. In s1-l12 you can train on the shuffled split and see a validation loss that looks too good.
:::

::: recap
- Characters ↔ ids, with sorted and repeatable tables.
- A text of length $n$ gives $n - k$ examples; the largest valid start is $n - k - 1$.
- Training and validation data are split at one position, because overlapping windows leak.
- A whole batch is built with one broadcast, `(B, 1) + (1, k) → (B, k)`, and no Python loop.
:::
