---
id: n-l02
title: Changing shapes
stage: n
order: 2
prereqs: [n-l01]
files: [workspace/numpy_basics/n02_shapes.py]
functions: [to_rows, flatten_each, swap_last_two, stack_two]
tests: tests/n/test_l02_shapes.py
est_minutes: 35
allow_examples: true
---

::: goal
You will change an array's shape with `reshape`, flip its axes with `.T` and `np.transpose`, and join arrays with `np.stack` and `np.concatenate`. The model code does this in almost every lesson.
:::

::: notation
| Word | Meaning |
| --- | --- |
| axis | one side of the array; axis 0 is the first number in the shape, axis 1 the second, and so on |
| `reshape` | read the same numbers into a new shape |
| `-1` in a shape | "work this number out for me" |
| transpose | swap axes: rows become columns |
| $B, k, e$ | batch size, context length, vector size (used from Stage 1 on) |
:::

::: concept
### reshape: same numbers, new grid

`reshape` puts the same numbers into a grid of a different shape. NumPy reads the numbers row by row and fills the new grid row by row:

```python
v = np.arange(6)
print(v)                  # [0 1 2 3 4 5]
print(v.reshape(2, 3))
# [[0 1 2]
#  [3 4 5]]
print(v.reshape(3, 2))
# [[0 1]
#  [2 3]
#  [4 5]]
```

The total number of entries must stay the same. 6 numbers fit into `(2, 3)` or `(3, 2)`, but not into `(4, 2)`:

```python
v.reshape(4, 2)
# ValueError: cannot reshape array of size 6 into shape (4,2)
```

### -1 means "work it out"

You may write `-1` for one side. NumPy computes it from the total:

```python
print(v.reshape(2, -1).shape)    # (2, 3)    6 / 2 = 3
print(v.reshape(-1, 2).shape)    # (3, 2)    6 / 2 = 3
```

This is useful when you know one side but not the other.

### Three axes

Arrays can have more than two axes. In Stage 1 a batch of embeddings has shape `(B, k, e)`: $B$ examples, each with $k$ characters, each character a vector of $e$ numbers. Think of it as $B$ matrices of shape $(k, e)$ stacked on top of each other.

A very common step glues each example's $k$ vectors into one long row, `(B, k, e)` → `(B, k·e)`:

```python
x = np.arange(24).reshape(2, 3, 4)   # B = 2, k = 3, e = 4
print(x.shape)                       # (2, 3, 4)
print(x.reshape(2, -1).shape)        # (2, 12)    each example becomes one row of 3 * 4 numbers
print(x.reshape(-1).shape)           # (24,)      everything in one line
```

Write the first number (the batch size) explicitly and `-1` for the rest. Then the batch axis is kept, even when $B = 1$.

### Transpose: rows become columns

`.T` swaps rows and columns, like $A^\top$ in math:

```python
M = np.arange(6).reshape(2, 3)
print(M.T)
# [[0 3]
#  [1 4]
#  [2 5]]
print(M.T.shape)                     # (3, 2)
```

For three or more axes, say which axes to swap:

```python
x = np.zeros((2, 3, 4))
print(np.swapaxes(x, 1, 2).shape)        # (2, 4, 3)   swap axis 1 and axis 2
print(np.transpose(x, (2, 0, 1)).shape)  # (4, 2, 3)   new order: old axis 2, then 0, then 1
```

`np.transpose(x, (2, 0, 1))` means: the new axis 0 is the old axis 2, the new axis 1 is the old axis 0, and the new axis 2 is the old axis 1.

### reshape and transpose do not copy

Both return a **view**: a new array object that looks at the same numbers in memory. Changing the view changes the original:

```python
v = np.arange(6)
r = v.reshape(2, 3)
r[0, 0] = 100
print(v)                             # [100   1   2   3   4   5]
```

This makes them fast. Remember it when you change values in place.

### Joining arrays

`np.stack` puts arrays on top of each other along a **new** axis. `np.concatenate` joins them along an **existing** axis:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.stack([a, b]))
# [[1 2 3]
#  [4 5 6]]                          shape (2, 3): a new axis of length 2
print(np.concatenate([a, b]))        # [1 2 3 4 5 6]   shape (6,)
```

With matrices, `axis=` picks the side to join along:

```python
M = np.zeros((2, 3))
print(np.concatenate([M, M], axis=0).shape)   # (4, 3)   more rows
print(np.concatenate([M, M], axis=1).shape)   # (2, 6)   more columns
```
:::

::: think
Q: `x` has shape `(32, 8, 24)`. What is the shape of `x.reshape(32, -1)`, and of `x.reshape(-1, 24)`?
A: `(32, 192)` because $8 \cdot 24 = 192$, and `(256, 24)` because $32 \cdot 8 = 256$.
:::

::: think
Q: Why write `x.reshape(B, -1)` instead of `x.reshape(-1)` when you flatten each example?
A: `reshape(-1)` puts every number of every example into one long vector, so the examples are mixed together and the batch axis disappears. `reshape(B, -1)` keeps one row per example.
:::

::: think
Q: `M` has shape `(2, 3)`. What shapes do `np.stack([M, M])` and `np.concatenate([M, M])` have?
A: `(2, 2, 3)`: `stack` adds a new first axis. `(4, 3)`: `concatenate` joins along the existing axis 0 (its default).
:::

::: hint level=1
Each function is one line. Pick the tool from the lesson: `reshape`, `swapaxes` or `transpose`, `stack`.
:::

::: hint level=2
`to_rows`: `v.reshape(rows, -1)`. `flatten_each`: keep `x.shape[0]` and use `-1` for the rest. `swap_last_two`: `np.swapaxes(x, 1, 2)` (or axes `-1` and `-2`). `stack_two`: `np.stack([a, b])`.
:::

::: hint level=3
```pseudo
to_rows(v, rows):     return v reshaped to (rows, whatever fits)
flatten_each(x):      return x reshaped to (number of examples, whatever fits)
swap_last_two(x):     return x with its last two axes swapped
stack_two(a, b):      return a and b stacked along a new first axis
```
:::

::: attempt
The editor below opens `workspace/numpy_basics/n02_shapes.py`: fill in the four functions.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/numpy_basics/n02_shapes.py#to_rows

@include solutions/numpy_basics/n02_shapes.py#flatten_each

- `x.shape[0]` is the batch size. Writing it out keeps one row per example; `-1` computes `k * e`.

@include solutions/numpy_basics/n02_shapes.py#swap_last_two

- `-1` and `-2` count axes from the end, so the same line works for any number of axes.

@include solutions/numpy_basics/n02_shapes.py#stack_two
:::

::: breakit
Make `M = np.arange(6).reshape(2, 3)` and compare `M.T.reshape(-1)` with `M.reshape(-1)`. The first gives `[0 3 1 4 2 5]`, the second `[0 1 2 3 4 5]`. A transpose changes the order in which the numbers are read, so a reshape after a transpose gives a different arrangement. Stage 2 depends on getting this order right.
:::

::: recap
- `reshape` puts the same numbers into a new shape; `-1` fills in one side.
- `x.reshape(B, -1)` turns `(B, k, e)` into `(B, k·e)` and keeps one row per example.
- `.T`, `np.swapaxes` and `np.transpose` reorder axes.
- `reshape` and transpose return views of the same memory.
- `np.stack` adds a new axis; `np.concatenate` joins along an existing one.
:::
