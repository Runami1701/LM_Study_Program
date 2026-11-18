---
id: n-l03
title: Picking entries (indexing)
stage: n
order: 3
prereqs: [n-l02]
files: [workspace/numpy_basics/n03_indexing.py]
functions: [last_column, window, lookup_rows, pick_per_row, count_ids]
tests: tests/n/test_l03_indexing.py
est_minutes: 45
allow_examples: true
---

::: goal
You will pick single entries, rows, columns and ranges out of an array, pick many rows at once with a list of positions, and add values into chosen positions without losing repeats. The model uses each of these: reading a window of text, looking up character vectors, and reading the probability of the correct answer.
:::

::: notation
| Word | Meaning |
| --- | --- |
| index | a position, counted from 0 |
| slice | a range of positions, written `start:stop`; `stop` is not included |
| `:` alone | "every position on this axis" |
| negative index | counts from the end: `-1` is the last position |
| index array | an array of positions, used to pick many entries at once |
| view / copy | a view shares memory with the original; a copy does not |
:::

::: concept
All examples use this matrix:

```python
A = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])     # shape (3, 4)
```

### One entry, one row, one column

Write one index per axis, separated by a comma. `:` means "all positions on this axis":

```python
print(A[1, 2])     # 22                 row 1, column 2
print(A[1])        # [20 21 22 23]      row 1 (all columns)
print(A[:, 2])     # [12 22 32]         column 2 (all rows)
print(A[-1])       # [30 31 32 33]      the last row
print(A[0, -1])    # 13                 row 0, last column
```

### Ranges (slices)

`start:stop` picks the positions from `start` up to, but not including, `stop`. Leave out `start` to begin at 0, or `stop` to run to the end:

```python
d = np.array([5, 6, 7, 8, 9])
print(d[1:4])      # [6 7 8]     positions 1, 2, 3
print(d[:2])       # [5 6]       the first 2
print(d[3:])       # [8 9]       from position 3 to the end
print(d[-2:])      # [8 9]       the last 2
```

`d[1:4]` has $4 - 1 = 3$ entries. In general `d[s:s+k]` has exactly $k$ entries, starting at position $s$. Stage 1 cuts windows of $k$ characters out of the text this way.

Slices work on each axis:

```python
print(A[0:2])
# [[10 11 12 13]
#  [20 21 22 23]]     rows 0 and 1
print(A[:, 1:3])
# [[11 12]
#  [21 22]
#  [31 32]]           columns 1 and 2
```

Going past the end with a single index is an error: `d[5]` raises `IndexError: index 5 is out of bounds for axis 0 with size 5`.

### Many positions at once (index arrays)

Give an array (or list) of positions, and NumPy returns those entries, in that order. Positions may repeat:

```python
print(d[[0, 2, 2, 4]])   # [5 7 7 9]
```

This is how the model looks up character vectors. Store one vector per character as the rows of a table `C`. Index `C` with an array of character ids, and you get one row per id. The result has the shape of the id array, plus the length of a row:

```python
C = np.array([[ 0.1,  0.2],
              [ 0.3, -0.4],
              [ 0.5,  0.6],
              [-0.7,  0.8]])       # 4 characters, each a vector of 2 numbers
ids = np.array([[1, 0],
                [3, 3]])           # shape (2, 2)
print(C[ids].shape)                # (2, 2, 2)   shape of ids, then 2
print(C[ids])
# [[[ 0.3 -0.4]
#   [ 0.1  0.2]]
#
#  [[-0.7  0.8]
#   [-0.7  0.8]]]
```

`C[ids][0, 0]` is row 1 of `C`, because `ids[0, 0]` is 1.

### One entry per row

Two index arrays of the same length pick pairs: the first array gives the rows, the second the columns. This reads one entry from each row:

```python
P = np.array([[0.7, 0.2, 0.1],
              [0.1, 0.1, 0.8]])    # 2 rows of probabilities
y = np.array([0, 2])               # the correct column in each row
print(P[np.arange(2), y])          # [0.7 0.8]   P[0, 0] and P[1, 2]
```

`np.arange(2)` is `[0, 1]`, the row numbers. Stage 1 uses exactly this line to read the probability the model gave to the correct character.

### Views and copies

A slice returns a **view**: changing it changes the original. An index array returns a **copy**: changing it leaves the original alone.

```python
d = np.array([5, 6, 7, 8, 9])
s = d[1:3]
s[0] = -1
print(d)           # [ 5 -1  7  8  9]    the slice was a view

d = np.array([5, 6, 7, 8, 9])
f = d[[1, 2]]
f[0] = -1
print(d)           # [5 6 7 8 9]         the index-array result was a copy
```

### Adding into positions that repeat

You want to count how often each id appears in `[1, 1, 3]`. The obvious line loses one of the repeats:

```python
c = np.zeros(4)
c[np.array([1, 1, 3])] += 1
print(c)           # [0. 1. 0. 1.]       wrong: id 1 appears twice
```

NumPy reads `c` at the positions once, adds 1, and writes the results back. Both writes go to position 1, and the second simply overwrites the first. `np.add.at` adds one position at a time, so repeats add up:

```python
c = np.zeros(4)
np.add.at(c, np.array([1, 1, 3]), 1)
print(c)           # [0. 2. 0. 1.]       right
```

Stage 1 needs this when it sends gradients back to the character vectors: a character that appears twice must receive both contributions.
:::

::: think
Q: `d = np.arange(10)`. What do `d[2:5]`, `d[-3:]` and `d[[9, 0]]` return?
A: `[2 3 4]` (3 entries), `[7 8 9]` (the last 3) and `[9 0]` (the entries at positions 9 and 0, in that order).
:::

::: think
Q: A table `C` has shape `(65, 24)` and `ids` has shape `(32, 8)`. What is the shape of `C[ids]`?
A: `(32, 8, 24)`: the shape of `ids`, followed by the length of one row of `C`.
:::

::: think
Q: Why does `c[ids] += 1` give the wrong count when `ids` contains a repeated position?
A: NumPy reads all chosen positions first, adds 1 to each, then writes them back. A repeated position is written twice with the same value, so only one of the additions survives. `np.add.at` performs the additions one by one.
:::

::: hint level=1
Every function is one line that uses one kind of indexing from the lesson: a column, a slice, an index array, a pair of index arrays, or `np.add.at`.
:::

::: hint level=2
`last_column`: `A[:, -1]`. `window`: a slice `data[start:start + k]`. `lookup_rows`: `table[ids]`. `pick_per_row`: `P[np.arange(len(cols)), cols]`. `count_ids`: start from `np.zeros(V, dtype=np.int64)` and use `np.add.at`.
:::

::: hint level=3
```pseudo
last_column(A):          return every row of A, last column
window(data, start, k):  return the k entries of data that begin at position start
lookup_rows(table, ids): return the rows of table chosen by ids
pick_per_row(P, cols):   return, for each row r, the entry P[r, cols[r]]
count_ids(ids, V):       counts ← V zeros (int64); add 1 at every position in ids, repeats included
```
:::

::: attempt
File: `workspace/numpy_basics/n03_indexing.py` — fill in the five functions.
Check: `python -m course check n-l03`
:::

::: solution
@include solutions/numpy_basics/n03_indexing.py#last_column

@include solutions/numpy_basics/n03_indexing.py#window

- `stop` is not included, so `start + k` gives exactly `k` entries.

@include solutions/numpy_basics/n03_indexing.py#lookup_rows

@include solutions/numpy_basics/n03_indexing.py#pick_per_row

- `np.arange(len(cols))` gives the row numbers 0, 1, 2, …, paired with one column per row.

@include solutions/numpy_basics/n03_indexing.py#count_ids

- `np.add.at` adds at each position in turn, so repeated ids are all counted. (`np.bincount(ids, minlength=V)` gives the same counts.)
:::

::: breakit
Run `c = np.zeros(3); c[[0, 0, 0]] += 1; print(c)`. You get `[1. 0. 0.]`, not `[3. 0. 0.]`. Then run `np.add.at(c, [0, 0, 0], 1)` on a fresh `c` and get `[3. 0. 0.]`.
:::

::: recap
- `A[i, j]` picks one entry, `A[i]` a row, `A[:, j]` a column; negative indices count from the end.
- `start:stop` picks a range without `stop`; `d[s:s+k]` is a window of `k` entries.
- `table[ids]` looks up many rows at once; the result has the shape of `ids` plus a row.
- `P[np.arange(n), y]` picks one entry per row.
- Slices are views; index arrays give copies. `np.add.at` adds into repeated positions correctly.
:::
