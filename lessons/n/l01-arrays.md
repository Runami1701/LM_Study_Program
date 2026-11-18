---
id: n-l01
title: Arrays, shapes and number types
stage: n
order: 1
prereqs: []
files: [workspace/numpy_basics/n01_arrays.py]
functions: [zeros_grid, describe, to_float32, multiply_both_ways]
tests: tests/n/test_l01_arrays.py
est_minutes: 40
allow_examples: true
---

::: goal
You will make NumPy arrays, read their shape and number type, and see the difference between `*` and `@`. Every later lesson uses these four skills.
:::

::: notation
| Word | Meaning |
| --- | --- |
| array | a grid of numbers that all have the same type |
| shape | the length of each side of the grid, e.g. `(2, 3)` for 2 rows and 3 columns |
| `ndim` | the number of sides: 1 for a vector, 2 for a matrix |
| `size` | the total number of entries: $2 \times 3 = 6$ for shape `(2, 3)` |
| dtype | the number type of every entry, e.g. `int64`, `float64`, `float32` |
| $A_{ij}$ | the entry of matrix $A$ in row $i$, column $j$ |
:::

::: concept
### Before you start

Start the course by double-clicking `start_course.bat` (Windows) or running `./start_course.sh` (Linux/macOS). The first start installs everything; later starts just open this website.

Everything else happens on this website:

- **Write your code** in the editor in each lesson's *Your attempt* section, and press **Save** (or Ctrl+S).
- **Check it** with the **Run check** button in the *Check* section. It tests your functions and shows what is right and what is wrong.
- **Run training** and other long tasks with the buttons in the *Run* section of the lessons that need them.
- **Try Python** in the *Python playground* below each lesson's explanation. Every code example has a **Try it** button that copies it there and runs it.

Every example uses the name `np` for NumPy; the playground adds `import numpy as np` for you.

**See it drawn:** the <a href="explainers/matrices-attention.html#grid" target="_blank" rel="noopener">visual companion</a>, sections 1 and 2: a matrix as a grid, and the row-times-column product. Click the cells and tokens; every number is recomputed live.

### An array is a grid of numbers

`np.array` turns a Python list into an array:

```python
v = np.array([1, 2, 3])
print(v)          # [1 2 3]
```

A list of lists becomes a matrix. Each inner list is one row:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A)
# [[1 2 3]
#  [4 5 6]]
```

In math you would write this matrix as $A \in \mathbb{R}^{2 \times 3}$.

### Shape, ndim and size

Every array knows its own size. Ask it:

```python
print(A.shape)    # (2, 3)   2 rows, 3 columns
print(A.ndim)     # 2        a matrix has 2 sides
print(A.size)     # 6        2 * 3 entries
```

`A.shape` is a tuple. The first number counts rows, the second counts columns. A vector like `v` has shape `(3,)`. The comma tells Python that this is a tuple with one number in it.

You will read shapes all the time in this course. When code fails, the first thing to check is the shape of every array.

### Counting starts at 0

Math writes the top-left entry as $A_{11}$. NumPy writes it as `A[0, 0]`. Rows and columns are numbered from 0:

```python
print(A[0, 0])    # 1    row 0, column 0
print(A[1, 2])    # 6    row 1, column 2 (the last column)
```

So the math entry $A_{ij}$ is `A[i-1, j-1]` in NumPy. Lesson n-l03 covers indexing in detail.

### Making arrays without typing every number

| code | result |
| --- | --- |
| `np.zeros((2, 3))` | a 2×3 matrix of zeros |
| `np.ones(4)` | `[1. 1. 1. 1.]` |
| `np.full((2, 2), 7.5)` | a 2×2 matrix where every entry is 7.5 |
| `np.arange(5)` | `[0 1 2 3 4]`, the numbers 0 to 4 |

Note the double parentheses in `np.zeros((2, 3))`. The inner pair is the shape tuple; the outer pair belongs to the function call.

### Number types (dtype)

Every array has one number type for all its entries. NumPy picks the type from the numbers you give it:

```python
print(np.array([1, 2, 3]).dtype)     # int64     whole numbers
print(np.array([1, 2.5]).dtype)      # float64   one decimal makes it float
print(np.zeros(3).dtype)             # float64   the default for zeros, ones, full
```

`int64` stores whole numbers. `float64` stores decimal numbers with about 16 correct digits, using 8 bytes each.

This course trains models in **float32**. A float32 number uses 4 bytes and keeps about 7 correct digits. That is enough for training, it halves the memory, and the computer multiplies float32 matrices about twice as fast. A 1000×1000 matrix takes 4,000,000 bytes in float32 and 8,000,000 bytes in float64.

You choose the type in two ways:

```python
x = np.zeros(3, dtype=np.float32)          # ask for float32 when you create the array
print(x.dtype)                             # float32

y = np.array([1, 2, 3]).astype(np.float32) # convert an existing array
print(y)                                   # [1. 2. 3.]
```

`.astype` always makes a new array. The original keeps its type.

### Arithmetic works entry by entry

`+`, `-`, `*`, `/` and `**` act on matching entries:

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])
print(a + b)      # [11. 22. 33.]
print(a * b)      # [10. 40. 90.]
print(a * 2)      # [2. 4. 6.]     the 2 is used for every entry
print(a ** 2)     # [1. 4. 9.]
```

No loop is needed. NumPy runs the loop for you, much faster than a Python `for` loop would.

### `*` is not matrix multiplication; `@` is

This is the most common beginner mistake. For two matrices, `*` multiplies matching entries. `@` computes the matrix product you know from linear algebra, $(AB)_{ij} = \sum_k A_{ik} B_{kj}$:

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[10, 0],
              [ 0, 1]])
print(A * B)
# [[10  0]
#  [ 0  4]]      each entry: A[i, j] * B[i, j]
print(A @ B)
# [[10  2]
#  [30  4]]      the matrix product
```

Check one entry of `A @ B` by hand: row 0 of $A$ is $[1, 2]$ and column 0 of $B$ is $[10, 0]$, so the entry is $1 \cdot 10 + 2 \cdot 0 = 10$.

The shapes follow the math rule: `(n, m) @ (m, p)` gives `(n, p)`, and the two inner numbers must match. `*` needs shapes that match entry by entry. Lesson n-l04 shows the exceptions to that rule.
:::

::: think
Q: What are the shape, `ndim` and `size` of `np.zeros((4, 5))`? What is its dtype?
A: Shape `(4, 5)`, `ndim` 2, `size` 20. The dtype is `float64`, because `zeros` uses float64 unless you pass `dtype=`.
:::

::: think
Q: `A` has shape `(2, 3)` and `B` has shape `(3, 2)`. Which of `A @ B` and `A * B` works, and what shape does it give?
A: `A @ B` works and gives shape `(2, 2)`: the inner numbers (3 and 3) match. `A * B` fails with `ValueError: operands could not be broadcast together with shapes (2,3) (3,2)`, because the two grids have different shapes.
:::

::: think
Q: You store 1,000,000 numbers. How many bytes do you need in float32, and in float64?
A: 4,000,000 bytes in float32 (4 bytes each) and 8,000,000 bytes in float64 (8 bytes each).
:::

::: hint level=1
Each function needs one or two NumPy calls from this lesson. Look at the table "Making arrays" and the section on number types.
:::

::: hint level=2
`zeros_grid`: `np.zeros` with a shape tuple and `dtype=np.float32`. `describe`: read `.shape`, `.ndim`, `.size` and `.dtype.name` from the array. `to_float32`: `np.array(..., dtype=np.float32)` works for lists and arrays. `multiply_both_ways`: `*` and `@`.
:::

::: hint level=3
```pseudo
zeros_grid(rows, cols):      return a float32 array of zeros with shape (rows, cols)
describe(a):                 return (a's shape, a's ndim, a's size, a's dtype name)
to_float32(values):          return values as a float32 array
multiply_both_ways(A, B):    return (A times B entry by entry, A matrix-times B)
```
:::

::: attempt
The editor below opens `workspace/numpy_basics/n01_arrays.py`: fill in the four functions.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/numpy_basics/n01_arrays.py#zeros_grid

- `(rows, cols)` is the shape tuple. `dtype=np.float32` sets the type when the array is made, so no conversion is needed.

@include solutions/numpy_basics/n01_arrays.py#describe

- `a.dtype` is a type object; `.name` gives its name as text, such as `"float32"`.

@include solutions/numpy_basics/n01_arrays.py#to_float32

- `np.array` accepts a Python list, a number or another array, and `dtype=` converts on the way.

@include solutions/numpy_basics/n01_arrays.py#multiply_both_ways

- `*` multiplies matching entries; `@` is the matrix product.
:::

::: breakit
Run `np.ones((2, 3)) @ np.ones((2, 3))`. NumPy refuses with `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ...`, because the inner numbers are 3 and 2. Now run `np.ones((2, 3)) @ np.ones((3, 2))`. It works and gives a `(2, 2)` matrix full of 3s. Each entry adds three products of $1 \cdot 1$.
:::

::: recap
- An array is a grid of numbers of one type. `shape`, `ndim`, `size` and `dtype` describe it.
- NumPy counts from 0: the math entry $A_{11}$ is `A[0, 0]`.
- `np.zeros`, `np.ones`, `np.full` and `np.arange` make arrays; `dtype=np.float32` or `.astype(np.float32)` sets the type.
- `+ - * / **` work entry by entry. `@` is the matrix product.
:::
