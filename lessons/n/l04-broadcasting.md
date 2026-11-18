---
id: n-l04
title: Sums along an axis, and broadcasting
stage: n
order: 4
prereqs: [n-l03]
files: [workspace/numpy_basics/n04_broadcasting.py]
functions: [column_sums, row_means, normalize_rows, outer_sum, shift_rows]
tests: tests/n/test_l04_broadcasting.py
est_minutes: 45
allow_examples: true
---

::: goal
You will add up or average an array along one axis, keep that axis with `keepdims=True`, and combine arrays of different shapes with broadcasting. With these tools the model processes a whole batch without a Python loop.
:::

::: notation
| Word | Meaning |
| --- | --- |
| `axis=0` | along the rows: the result has one number per column |
| `axis=1` | along the columns: the result has one number per row |
| `keepdims=True` | keep the summed axis, with length 1 |
| broadcasting | NumPy stretches a length-1 axis to match the other array |
| `x[:, None]` | `x` turned into a column: shape `(n,)` becomes `(n, 1)` |
| $s_r$ | the start of window $r$ |
| $j$ | a position inside the window |
:::

::: concept
All examples use this matrix:

```python
A = np.array([[1., 2., 3.],
              [4., 5., 6.]])       # shape (2, 3)
```

### Sums and means along an axis

`A.sum()` adds every entry. With `axis=`, it adds along one side only:

```python
print(A.sum())            # 21.0
print(A.sum(axis=0))      # [5. 7. 9.]    down the rows: one sum per column
print(A.sum(axis=1))      # [ 6. 15.]     across the columns: one sum per row
print(A.mean(axis=1))     # [2. 5.]       one mean per row
print(A.max(axis=1))      # [3. 6.]       one maximum per row
```

The rule: **the axis you name disappears.** `A` has shape `(2, 3)`. Summing over axis 0 removes the 2 and leaves shape `(3,)`. Summing over axis 1 removes the 3 and leaves `(2,)`.

`axis=-1` means the last axis, whatever the number of axes. The model uses `axis=-1` all the time, because the last axis holds the numbers of one vector.

### keepdims: keep the axis, with length 1

`keepdims=True` keeps the summed axis with length 1:

```python
print(A.sum(axis=1, keepdims=True))
# [[ 6.]
#  [15.]]
print(A.sum(axis=1, keepdims=True).shape)   # (2, 1)   instead of (2,)
```

A column like this can be combined with `A` directly, thanks to broadcasting. For example, it divides each row by its own sum:

```python
print(A / A.sum(axis=1, keepdims=True))
# [[0.16666667 0.33333333 0.5       ]
#  [0.26666667 0.33333333 0.4       ]]     each row now adds up to 1
```

### Broadcasting: how NumPy combines different shapes

Arithmetic needs matching shapes, with one exception. NumPy lines up the shapes **from the right**. On each axis, the two lengths must be equal, or one of them must be 1. An axis of length 1 is stretched to match the other.

A row added to every row:

```python
print(A + np.array([10, 20, 30]))
# [[11. 22. 33.]
#  [14. 25. 36.]]
```

Shapes: `(2, 3)` and `(3,)`. From the right, 3 matches 3. The `(3,)` array is treated as one row, `(1, 3)`, and the 1 stretches to 2.

A column added to every column:

```python
print(A + np.array([[100], [200]]))
# [[101. 102. 103.]
#  [204. 205. 206.]]
```

Shapes: `(2, 3)` and `(2, 1)`. 3 against 1: the column stretches across the 3 columns.

When neither length is 1, NumPy refuses. The row sums without `keepdims` have shape `(2,)`, which lines up against the 3 columns, not the 2 rows:

```python
A / A.sum(axis=1)
# ValueError: operands could not be broadcast together with shapes (2,3) (2,)
```

That is why the row-normalizing example above needs `keepdims=True`.

### Making a column or a row on purpose

`x[:, None]` turns a vector of shape `(n,)` into a column `(n, 1)`. `x[None, :]` turns it into a row `(1, n)`. A column plus a row gives a whole table:

```python
s = np.array([0, 10, 20])          # 3 starting positions
o = np.arange(4)                   # offsets 0, 1, 2, 3
print(s[:, None] + o[None, :])
# [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]]
```

Shapes: `(3, 1) + (1, 4)` gives `(3, 4)`. Entry $(r, j)$ is $s_r + j$. Each row lists the positions of one window. Stage 1 builds all the windows of a batch with this one line, then reads the text at those positions with the index arrays of n-l03.

### Subtracting each row's largest value

One more pattern the model needs: shift every row so that its largest entry becomes 0.

```python
print(A - A.max(axis=1, keepdims=True))
# [[-2. -1.  0.]
#  [-2. -1.  0.]]
```

`A.max(axis=1, keepdims=True)` has shape `(2, 1)`, so each row loses its own maximum. Stage 1 uses this shift to compute probabilities without overflow.
:::

::: think
Q: `x` has shape `(32, 65)`. What are the shapes of `x.sum(axis=1)`, `x.sum(axis=1, keepdims=True)` and `x.sum(axis=0)`?
A: `(32,)`, `(32, 1)` and `(65,)`. The named axis disappears, or keeps length 1 with `keepdims=True`.
:::

::: think
Q: Do these pairs broadcast, and what is the result shape: `(4, 3)` with `(3,)`; `(4, 3)` with `(4,)`; `(4, 1)` with `(1, 5)`?
A: `(4, 3)`: yes, the row is repeated 4 times. `(4,)`: no, because from the right 3 meets 4 and neither is 1. `(4, 5)`: yes, both length-1 axes stretch.
:::

::: think
Q: Why is `x[:, None] + y[None, :]` called an "outer sum"?
A: Entry $(i, j)$ is $x_i + y_j$: every entry of `x` meets every entry of `y`, the way the outer product $x y^\top$ pairs every $x_i$ with every $y_j$, but with $+$ instead of $\times$.
:::

::: hint level=1
Every function is one line. Pick `axis=0` or `axis=1`, use `keepdims=True` when the result must combine with the original, and use `[:, None]` to make a column.
:::

::: hint level=2
`column_sums`: `A.sum(axis=0)`. `row_means`: `A.mean(axis=1)`. `normalize_rows`: divide by `A.sum(axis=1, keepdims=True)`. `outer_sum`: `a[:, None] + b[None, :]`. `shift_rows`: subtract `A.max(axis=1, keepdims=True)`.
:::

::: hint level=3
```pseudo
column_sums(A):     return one sum per column
row_means(A):       return one mean per row
normalize_rows(A):  return A divided by its row sums (kept as a column)
outer_sum(a, b):    return the table whose entry (i, j) is a[i] + b[j]
shift_rows(A):      return A minus its row maxima (kept as a column)
```
:::

::: attempt
File: `workspace/numpy_basics/n04_broadcasting.py` — fill in the five functions.
Check: `python -m course check n-l04`
:::

::: solution
@include solutions/numpy_basics/n04_broadcasting.py#column_sums

@include solutions/numpy_basics/n04_broadcasting.py#row_means

@include solutions/numpy_basics/n04_broadcasting.py#normalize_rows

- `keepdims=True` makes the row sums a `(n, 1)` column, which broadcasts across each row.

@include solutions/numpy_basics/n04_broadcasting.py#outer_sum

- `(n, 1) + (1, m)` broadcasts to `(n, m)`.

@include solutions/numpy_basics/n04_broadcasting.py#shift_rows
:::

::: breakit
Take a square matrix `S = np.array([[1., 2.], [3., 4.]])` and run `S / S.sum(axis=1)` without `keepdims`. NumPy does not complain, because `(2, 2)` and `(2,)` broadcast. But it divides each **column** by a row sum: you get `[[0.333 0.286] [1. 0.571]]`, and the rows do not add up to 1. With `keepdims=True` you get the correct `[[0.333 0.667] [0.429 0.571]]`. Square matrices hide this mistake, so always keep the axis.
:::

::: recap
- `sum`, `mean` and `max` with `axis=` work along one side; the named axis disappears.
- `keepdims=True` keeps it with length 1, so the result combines with the original array.
- Broadcasting lines shapes up from the right; a length-1 axis stretches to match.
- `x[:, None]` makes a column; a column plus a row gives a table, e.g. all window positions at once.
:::
