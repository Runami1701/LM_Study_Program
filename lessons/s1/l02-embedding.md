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
You will turn a batch of ids `x` `(B, k)` into vectors. Each id picks one row of a table `C` `(V, e)`, which gives `emb` `(B, k, e)`. Then each example's $k$ vectors are laid end to end, which gives `(B, k·e)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C$ | the embedding table, shape $(V, e)$; row $i$ is the vector of id $i$ (training changes it) |
| $C_{ij}$ | the entry of $C$ in row $i$, column $j$ |
| $V$ | vocabulary size |
| $e$ | embedding size: how many numbers describe one character |
| $o$ | a one-hot row vector of length $V$: all zeros except a 1 at the id |
| $o_i$ | the $i$-th entry of $o$ |
| $\text{id}$ | the id being looked up |
| $B, k$ | batch size and context length (from s1-l01) |
| `emb` | the looked-up vectors, shape `(B, k, e)` |
:::

::: numpy
### Look up many rows at once (n-l03)

Indexing a table with an array of ids returns one row per id. The result has the shape of the id array, followed by the length of a row:

```python
C = np.array([[ 0.1,  0.2],
              [ 0.3, -0.4],
              [ 0.5,  0.6],
              [-0.7,  0.8]])        # V = 4 rows, e = 2
x = np.array([[1, 0, 2]])           # shape (1, 3): one example, k = 3
print(C[x].shape)                   # (1, 3, 2)
```

### Glue each example's vectors together (n-l02)

`reshape(B, -1)` keeps one row per example and joins the rest:

```python
print(C[x].reshape(1, -1))          # [[ 0.3 -0.4  0.1  0.2  0.5  0.6]]
```
:::

::: concept
An id like `2` means nothing by itself; id 2 is not "twice" id 1. So the model gives each character its own vector of $e$ numbers, stored as row $i$ of a table $C$ of shape $(V, e)$. Training moves these rows, so that characters that behave alike end up with similar vectors.

### A lookup is a matrix product with a one-hot vector

Write the id as a **one-hot** row vector $o$ of length $V$: $o_i = 1$ if $i = \text{id}$, and $o_i = 0$ otherwise. Multiply it by $C$:

$$
(oC)_j = \sum_{i=0}^{V-1} o_i \, C_{ij}
$$

Every term with $i \ne \text{id}$ has $o_i = 0$ and drops out. One term is left, with $o_{\text{id}} = 1$:

$$
(oC)_j = 1 \cdot C_{\text{id}, j} = C_{\text{id}, j}
$$

So $oC$ is exactly row $\text{id}$ of $C$. **Multiplying by a one-hot vector picks a row.**

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

The one-hot product spends $V \cdot e$ multiplications per character, and all but $e$ of them multiply by zero. A lookup just copies $e$ numbers. For one batch with $B = 32$, $k = 8$, $e = 24$:

| | $V = 65$ | $V = 8000$ |
| --- | --- | --- |
| one-hot product: $B k V e$ multiply-adds | $399{,}360$ | $49{,}152{,}000$ |
| lookup: $B k e$ numbers copied | $6{,}144$ | $6{,}144$ |

The product gets slower as $V$ grows; the lookup does not. Both give the same numbers, so the check compares your lookup with the one-hot product.

### Looking up a whole batch

`C[x]` with `x` of shape `(B, k)` gives shape `(B, k, e)`, and `C[x][b, j]` is row `x[b, j]` of `C`. For the context `h e l` from s1-l01, `x = [[1, 0, 2]]` and the result holds rows 1, 0 and 2 of $C$:

$$
\texttt{emb}[0] = \begin{bmatrix} 0.3 & -0.4 \\ 0.1 & 0.2 \\ 0.5 & 0.6 \end{bmatrix} \quad \text{shape } (k, e) = (3, 2)
$$

### Laying the vectors end to end

The next layer (s1-l03) takes one vector per example. So we join each example's $k$ vectors into one row, `(B, k, e)` → `(B, k·e)`:

$$
[\,0.3,\ -0.4 \;\big|\; 0.1,\ 0.2 \;\big|\; 0.5,\ 0.6\,] \quad \text{shape } (1, 6)
$$

The first $e$ numbers of the row describe the first character, the next $e$ numbers the second, and so on. The next layer has separate weights for each of these slots, so it can tell "h then e" from "e then h". Adding the $k$ vectors together instead would lose the order.

This step is a `reshape`. `C[x]` returns a new array stored row by row, so the reshape copies nothing. Reshape to `(B, -1)` so the batch axis stays, even when $B = 1$. With $k = 8$ and $e = 24$, each example becomes a vector of $8 \cdot 24 = 192$ numbers.
:::

::: think
Q: With $V = 65$, what fraction of the multiplications in a one-hot product multiply by zero?
A: All but one of the $V$ terms per output number: $\frac{V - 1}{V} = \frac{64}{65} \approx 98.5\%$.
:::

::: think
Q: Why join the $k$ vectors end to end instead of adding them into one $e$-vector?
A: A sum is the same for any order of the characters, so "dog" and "god" would look identical. Joining gives each position its own slot, and the next layer has separate weights for each slot.
:::

::: think
Q: You flatten with `emb.reshape(-1)`. What shape do you get when $B = 1$, and why does it cause trouble later?
A: `(k·e,)`: a 1-D vector without a batch axis. The next layer expects `(B, n)`. `x @ W` still runs but returns `(m,)`, and every later shape is off by one axis. Always reshape to `(B, -1)`.
:::

::: hint level=1
Row $i$ of the table already *is* the vector of id $i$. NumPy can fetch many rows in one indexing step, in whatever shape the ids have.
:::

::: hint level=2
Index `C` with the `(B, k)` id array to get `(B, k, e)` (n-l03). For the second function, `reshape` with the batch size first and `-1` second (n-l02).
:::

::: hint level=3
```pseudo
embedding_forward(C, x):
    return the rows of C chosen by x            # (V, e) indexed by (B, k) → (B, k, e)

flatten_context(emb):
    B ← the first number of emb's shape
    return emb reshaped to (B, everything else) # (B, k·e)
```
:::

::: attempt
File: `workspace/stage1/layers.py` — fill in `embedding_forward` and `flatten_context`.
Check: `python -m course check s1-l02`
:::

::: solution
@include solutions/stage1/layers.py#embedding_forward

- `C[x]` fetches one row of `C` for each entry of `x`; the result has the shape of `x` followed by `(e,)`. Repeated ids are fine: the same row is copied twice.
- The result has `C`'s number type (float32). The type of `x` only decides which rows are picked.

@include solutions/stage1/layers.py#flatten_context

- `reshape(B, -1)` keeps the batch axis and joins `(k, e)` into `k·e`. NumPy reads row by row, so each output row is `emb[b, 0]`, then `emb[b, 1]`, and so on.
:::

::: breakit
Time both methods. Build a one-hot array of shape `(B, k, V)` for $B = 32$, $k = 8$ and multiply it by `C`; then time `C[x]`. Do this for $V = 65$ and $V = 8000$ ($e = 24$). On the reference machine the lookup took about 6 µs and 5 µs, and the one-hot product about 28 µs and 3000 µs. Stage 3 has a vocabulary of thousands of tokens, so this gap matters.
:::

::: recap
- Multiplying by a one-hot vector picks a row; a lookup does the same without the zeros.
- `C[x]` turns `(B, k)` ids into `(B, k, e)` vectors in one step.
- `reshape(B, -1)` joins each example's vectors into `(B, k·e)` and keeps the character order.
:::
