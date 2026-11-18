---
id: s1-l02
title: Embedding lookup (forward)
stage: s1
order: 2
prereqs: [s1-l01]
files: [workspace/stage1/layers.py]
functions: [embedding_forward, flatten_context]
tests: tests/s1/test_l02_embedding.py
est_minutes: 30
---

::: goal
Turn a batch of ids `x` `(B, k)` into vectors: look up each id in an embedding table `C` `(V, e)` to get `emb` `(B, k, e)`, then lay each example's $k$ vectors end to end to get `(B, k·e)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C$ | the embedding table, shape $(V, e)$; row $i$ is the vector for id $i$ (learned during training) |
| $C_{ij}$ | entry of $C$ in row $i$, column $j$ |
| $V$ | vocabulary size |
| $e$ | embedding dimension: the length of each character's vector |
| $o$ | a one-hot row vector of length $V$: all zeros except a 1 at the id |
| $o_i$ | the $i$-th entry of $o$ |
| $\text{id}$ | the integer id being looked up |
| $B, k$ | batch size and context length (from s1-l01) |
| `emb` | the looked-up vectors, shape `(B, k, e)` |
:::

::: concept
An id like `2` carries no meaning by itself: id 2 is not "twice" id 1. So the model gives every character its own learned vector of $e$ numbers, stored as row $i$ of a table $C$ of shape $(V, e)$. Training will move these rows so that characters that behave alike end up with similar vectors.

### The one-hot view: why a lookup is a matrix product

Write id $\text{id}$ as a **one-hot** row vector $o$ of length $V$: $o_i = 1$ if $i = \text{id}$, else $o_i = 0$. Multiply by $C$:

$$
(oC)_j = \sum_{i=0}^{V-1} o_i \, C_{ij}
$$

Every term with $i \ne \text{id}$ has $o_i = 0$ and vanishes, and the one remaining term has $o_{\text{id}} = 1$:

$$
(oC)_j = 1 \cdot C_{\text{id}, j} = C_{\text{id}, j}
$$

So $oC$ is exactly row $\text{id}$ of $C$. **Multiplying by a one-hot vector selects a row.**

**Worked example.** $V = 4$, $e = 2$ (the `"hello"` vocabulary `e h l o`), with

$$
C = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & -0.4 \\ 0.5 & 0.6 \\ -0.7 & 0.8 \end{bmatrix}
$$

Character `h` has id 1, so $o = [0, 1, 0, 0]$ and

$$
\begin{aligned}
oC &= 0 \cdot [0.1, 0.2] + 1 \cdot [0.3, -0.4] + 0 \cdot [0.5, 0.6] + 0 \cdot [-0.7, 0.8] \\
&= [0.3, -0.4]
\end{aligned}
$$

which is row 1 of $C$.

### Why look up instead of multiplying

The one-hot product spends $V \cdot e$ multiply-adds per character, and all but $e$ of them multiply by zero. A lookup just copies the $e$ numbers. For one batch with $B = 32$, $k = 8$, $e = 24$:

| | $V = 65$ | $V = 8000$ |
| --- | --- | --- |
| one-hot product: $B k V e$ multiply-adds | $399{,}360$ | $49{,}152{,}000$ |
| lookup: $B k e$ numbers copied | $6{,}144$ | $6{,}144$ |

The cost of the one-hot product grows with $V$; the lookup's does not. Both give the same numbers, which is why the check compares your lookup against the one-hot product.

### Looking up a whole batch at once

Indexing an array with an integer array returns an array shaped like the index, with the remaining axes of the table attached:

- `C` has shape `(V, e)`; `x` has shape `(B, k)`
- `C[x]` has shape `(B, k)` + `(e,)` = `(B, k, e)`, with `C[x][b, j] = C[x[b, j]]`

Continuing the example, the context `h e l` from s1-l01 is `x = [[1, 0, 2]]`, shape `(1, 3)`. Its embeddings are rows 1, 0, 2 of $C$:

$$
\texttt{emb}[0] = \begin{bmatrix} 0.3 & -0.4 \\ 0.1 & 0.2 \\ 0.5 & 0.6 \end{bmatrix} \quad \text{shape } (k, e) = (3, 2)
$$

### Flattening the context

The next layer (s1-l03) takes one vector per example. So we lay the $k$ embeddings of each example end to end, `(B, k, e)` → `(B, k·e)`:

$$
[\,0.3,\ -0.4 \;\big|\; 0.1,\ 0.2 \;\big|\; 0.5,\ 0.6\,] \quad \text{shape } (1, 6)
$$

Positions $0..e-1$ of the row hold the first character, the next $e$ the second, and so on. Because each slot is fed by **its own rows of weights** in the next layer, the model can tell "h then e" from "e then h". Summing or averaging the $k$ vectors would throw that order away.

This is a `reshape`. `C[x]` (fancy indexing) returns a fresh C-contiguous array (s0-l00), so reshaping it is a free view: no numbers move. Keep the batch axis explicit: reshape to `(B, -1)`, where `-1` means "whatever length makes the total fit", here $k \cdot e$.

With $k = 8$ and $e = 24$, each example becomes a vector of $8 \cdot 24 = 192$ numbers.
:::

::: think
Q: With $V = 65$, what fraction of the multiply-adds in a one-hot product are multiplications by zero?
A: All but one of the $V$ terms per output number: $\frac{V - 1}{V} = \frac{64}{65} \approx 98.5\%$.
:::

::: think
Q: Why concatenate the $k$ embeddings instead of summing them into one $e$-vector?
A: A sum is the same for any order of the characters ("dog" and "god" would look identical). Concatenation puts each position in its own slot, and the next layer has separate weights for each slot, so order is preserved.
:::

::: think
Q: You flatten with `emb.reshape(-1)` and it works when `B = 1`... or does it? What shape do you get, and why will it bite later?
A: `(k·e,)`, a 1-D vector with no batch axis. The next layer expects `(B, n)`; with a 1-D input, `x @ W` still runs but returns `(m,)`, and the bias, loss and every later shape quietly disagree. Always reshape to `(B, -1)`.
:::

::: hint level=1
Row $i$ of the table already *is* the embedding of id $i$. NumPy can fetch many rows in one indexing operation, in whatever shape you ask for.
:::

::: hint level=2
Integer-array ("fancy") indexing: indexing `C` with the `(B, k)` integer array returns `(B, k, e)`. For flattening, `reshape` with the batch size first and `-1` second.
:::

::: hint level=3
```pseudo
embedding_forward(C, x):
    return the rows of C selected by x            # (V, e) indexed by (B, k) → (B, k, e)

flatten_context(emb):
    B ← first dimension of emb
    return emb reshaped to (B, everything else)   # (B, k·e)
```
:::

::: attempt
File: `workspace/stage1/layers.py` — implement `embedding_forward` and `flatten_context`.
Check: `python -m course check s1-l02`
:::

::: solution
@include solutions/stage1/layers.py#embedding_forward

- `C[x]` is integer-array indexing: for each entry of `x` it fetches that row of `C`, and the result takes the shape of `x` followed by `(e,)`. Repeated ids are fine: the same row is copied twice.
- The dtype is `C`'s dtype (float32); the dtype of `x` (int64, uint8, ...) only affects which rows are picked.

@include solutions/stage1/layers.py#flatten_context

- `reshape(B, -1)` keeps the batch axis and merges `(k, e)` into `k·e`. In C order the last axis varies fastest, so each row is `emb[b, 0]`, then `emb[b, 1]`, ..., exactly the concatenation.
- No data is copied: `C[x]` produced a contiguous array, so the reshape is a view.
:::

::: breakit
Time the one-hot product against indexing. Build a one-hot `(B, k, V)` array for $B = 32$, $k = 8$ and multiply it by `C`, then compare with `C[x]`, for $V = 65$ and $V = 8000$ ($e = 24$). Expect indexing to take a few microseconds in both cases, the one-hot version a few times slower at $V = 65$ and several hundred times slower at $V = 8000$ (the reference machine: about 28 µs vs 6 µs, then 3000 µs vs 5 µs). Stage 3 has a vocabulary of thousands of subwords, so this gap is real.
:::

::: recap
- Multiplying by a one-hot vector selects a row; an embedding lookup is that product without the zeros.
- `C[x]` turns `(B, k)` ids into `(B, k, e)` vectors in one operation.
- Flattening to `(B, k·e)` keeps character order, and is a free view.
:::
