---
id: s0-l00
title: float32 traps and your machine's speed
stage: s0
order: 0
prereqs: [n-l04]
files: [workspace/stage0/numpy_basics.py]
functions: [to_f32, matmul_gflops]
tests: tests/s0/test_l00_setup.py
est_minutes: 30
---

::: goal
You will learn where float64 sneaks into float32 code, and you will measure how fast your computer multiplies matrices. You write two functions: `to_f32`, which converts arrays to float32, and `matmul_gflops`, which measures the speed.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| dtype | the number type of an array, e.g. `float32` (4 bytes per number) or `float64` (8 bytes) |
| $\varepsilon$ | machine epsilon: the gap between 1 and the next number the type can store |
| $n$ | matrix size: we multiply two $n \times n$ matrices |
| $A, B, C$ | the matrices in $C = AB$; $A_{il}$ is the entry in row $i$, column $l$ |
| $i, j, l$ | indices: $i$ a row of $C$, $j$ a column of $C$, $l$ runs along the shared side |
| FLOP | one floating-point operation: one multiplication or one addition |
| GFLOPS | a billion ($10^9$) FLOPs per second |
| $t$ | measured time in seconds |
| reps | how many times we repeat the measurement |
:::

::: numpy
### `np.asarray` converts only when it has to

`.astype` always makes a new array. `np.asarray(x, dtype=...)` returns `x` itself when `x` already has that type, and converts otherwise. `np.shares_memory(a, b)` tells you whether two arrays use the same memory:

```python
x = np.ones(3, dtype=np.float32)
y = np.asarray(x, dtype=np.float32)   # already float32: no copy
z = x.astype(np.float32)              # always a copy
print(np.shares_memory(x, y))         # True
print(np.shares_memory(x, z))         # False
print(np.asarray([1, 2], dtype=np.float32).dtype)   # float32   a list gets converted
```

### Random numbers with a generator

The course never uses NumPy's global random functions. Every function receives a generator `rng` instead, so a run can be repeated exactly:

```python
rng = np.random.default_rng(0)        # 0 is the seed
M = rng.standard_normal((2, 2), dtype=np.float32)   # 2x2 matrix of random float32 numbers
print(M.dtype)                        # float32
```

### Timing code

`time.perf_counter()` returns the current time in seconds, measured very precisely. Read it before and after, and subtract:

```python
import time
t0 = time.perf_counter()
# ... the work you want to time ...
seconds = time.perf_counter() - t0
```

### A tuple from a loop

`tuple(f(a) for a in arrays)` applies `f` to every item and collects the results in a tuple. `def to_f32(*arrays)` lets the function take any number of arguments; inside, `arrays` is a tuple of them.
:::

::: concept
### Why float32, and its limits

Lesson n-l01 showed that float32 uses half the memory of float64 and runs matrix products about twice as fast. Here is the price. A float stores a fixed number of binary digits:

| dtype | bytes | binary digits after the point | machine epsilon $\varepsilon$ | about |
| --- | --- | --- | --- | --- |
| `float32` | 4 | 23 | $2^{-23} \approx 1.19 \times 10^{-7}$ | 7 correct digits |
| `float64` | 8 | 52 | $2^{-52} \approx 2.22 \times 10^{-16}$ | 16 correct digits |

Example: in float32, `1 + 1e-8` is exactly `1`. The number $10^{-8}$ is smaller than half the gap $\varepsilon$ above 1, so the sum rounds back to 1. In float64 the sum is stored correctly.

Seven digits are enough for training. Gradients from a batch of examples are noisy by whole percents, far more than $10^{-7}$. Only one task needs float64: **gradient checking** in s1-l08, which subtracts two nearly equal numbers.

### Where float64 sneaks in

Two things turn float32 arrays into float64 without any warning.

**1. Creating an array without `dtype=`.** `np.zeros(3)`, `np.ones(3)`, `np.array([0.1, 0.2])` and `rng.standard_normal(3)` all give float64.

**2. Mixing in a NumPy float64 value.** NumPy 2 follows these rules, with `x` a float32 array:

| expression | result dtype | why |
| --- | --- | --- |
| `x * 0.5` | float32 | a plain Python number adopts the array's type |
| `x / math.sqrt(2)` | float32 | `math.sqrt` returns a plain Python number |
| `x / np.sqrt(2)` | **float64** | `np.sqrt(2)` returns a NumPy float64 value, which wins |
| `x + np.array(0.5)` | **float64** | a float64 array, even with one number, wins |
| `x + np.float32(0.5)` | float32 | same type on both sides |

After one such step, every later array is float64 too: double the memory, half the speed. The habit that prevents this: give `dtype=` whenever you create an array, and check `.dtype` when a function returns.

### Views, copies and contiguous memory

Lessons n-l02 and n-l03 showed that slices, `reshape` and `.T` return views, while index arrays and `.astype` return copies. One more detail matters later.

Memory is one long line of bytes. A float32 matrix of shape `(2, 3)` is stored row by row: `a[0,0], a[0,1], a[0,2], a[1,0], ...`. Such an array is called **contiguous**. `a.T` is a view that reads the same memory column by column, so it is not contiguous (`a.T.flags['C_CONTIGUOUS']` is `False`). Most functions handle this fine. One exception: `reshape` of a non-contiguous array has to copy the numbers. Stage 2 runs into this when it reshapes attention heads.

### Counting the work in a matrix product

Let $A$ and $B$ be $n \times n$ matrices and $C = AB$. One entry of $C$ is

$$
C_{ij} = \sum_{l=1}^{n} A_{il} B_{lj}
$$

This sum needs $n$ multiplications. Adding $n$ numbers together needs $n - 1$ additions. So one entry costs

$$
\text{FLOPs per entry} = n + (n - 1) = 2n - 1
$$

$C$ has $n^2$ entries, so the whole product costs

$$
\begin{aligned}
\text{FLOPs per product} &= n^2 (2n - 1) \\
&= 2n^3 - n^2 \\
&\approx 2n^3 \quad \text{(for large } n \text{, the } n^2 \text{ is tiny next to } 2n^3\text{)}
\end{aligned}
$$

For $n = 1024$: $2 \cdot 1024^3 \approx 2.15 \times 10^9$ FLOPs. If you repeat the product `reps` times and it takes $t$ seconds in total, the speed is

$$
\text{GFLOPS} = \frac{2n^3 \cdot \text{reps}}{t \cdot 10^9}
$$

Example: 5 products with $n = 1024$ in $t = 0.1$ seconds give $\frac{2.15 \times 10^9 \cdot 5}{0.1 \cdot 10^9} \approx 107$ GFLOPS.

Do one untimed **warm-up** product first. The first product pays one-time costs, such as starting the helper threads, that you don't want in the measurement.

NumPy does not compute `A @ B` itself. It calls a fast linear-algebra library (usually OpenBLAS), which uses all your CPU cores. `np.show_config()` prints which library you have. The number you measure mostly describes that library and your CPU.

When the check for this lesson passes, it saves your number in `progress.json`. In Stage 3 you will use it to estimate how long a long training run takes.
:::

::: think
Q: You write `h = x @ W1 / np.sqrt(fan_in)`, where `x` and `W1` are float32 and `fan_in` is an int. What is `h.dtype`, and how do you fix it?
A: `float64`, because `np.sqrt(fan_in)` returns a NumPy float64 value. Fix it with `math.sqrt(fan_in)`, which returns a plain Python number, or with `np.float32(np.sqrt(fan_in))`.
:::

::: think
Q: Why do we count $2n^3$ FLOPs for a matrix product instead of $n^3$?
A: Each of the $n^3$ terms $A_{il}B_{lj}$ needs one multiplication and one addition into the running sum. Both count as FLOPs.
:::

::: think
Q: `to_f32` must return a float32 input without copying it. What is gained, and what is the risk?
A: Gained: no time or memory is wasted when nothing needs converting. Risk: the result shares memory with the input, so changing one changes the other.
:::

::: hint level=1
`to_f32`: there is a NumPy conversion that only does work when the type is wrong (see the box above). `matmul_gflops`: make the two matrices once, do a warm-up product, time all `reps` products together, then apply the formula.
:::

::: hint level=2
`np.asarray(a, dtype=np.float32)` inside `tuple(... for a in arrays)`. For the matrices: `rng.standard_normal((n, n), dtype=np.float32)`. Time with `time.perf_counter()` before and after the loop. Return `float(...)`.
:::

::: hint level=3
```pseudo
to_f32(*arrays):
    return a tuple: each argument converted to float32 (no copy if it already is)

matmul_gflops(n, reps, rng):
    A, B ← two (n, n) float32 random matrices from rng
    A @ B                                   # warm-up, not timed
    t0 ← perf_counter()
    repeat reps times: A @ B
    seconds ← perf_counter() − t0
    return float( 2 · n³ · reps / seconds / 1e9 )
```
:::

::: attempt
The editor below opens `workspace/stage0/numpy_basics.py`: fill in `to_f32` and `matmul_gflops`.
Check: press **Run check** in the Check section below.. When it passes, it also prints and saves your computer's GFLOPS.
:::

::: solution
@include solutions/stage0/numpy_basics.py#to_f32

- `np.asarray(a, dtype=np.float32)` returns `a` unchanged when it is already a float32 array. Lists, numbers and other types get converted.
- `tuple(...)` always returns a tuple, even for one argument, so `(x,) = to_f32(x)` works.

@include solutions/stage0/numpy_basics.py#matmul_gflops

- The matrices are created as float32 directly, so no conversion is timed.
- The warm-up product is outside the timed part.
- `2.0 * n**3 * reps` is the total FLOP count; dividing by seconds and by $10^9$ gives GFLOPS.
- `float(...)` turns the NumPy number into a plain Python number, as the docstring asks.
:::

::: breakit
Make `x = np.ones(3, dtype=np.float32)` and print `(x + np.array(0.5)).dtype`, `(x / np.sqrt(2)).dtype` and `(x * 0.5).dtype`. You get `float64`, `float64` and `float32`. The first two are the silent upcasts from the table.
:::

::: breakit
Run your `matmul_gflops` once with float32 matrices and once with float64 matrices (change the `dtype`). The float64 number comes out at roughly half the float32 one.
:::

::: recap
- float32 keeps about 7 digits; that is enough for training but not for gradient checks.
- float64 sneaks in through creation without `dtype=` and through NumPy float64 values such as `np.sqrt(2)`.
- A product of two $n \times n$ matrices costs about $2n^3$ FLOPs; you derived it and measured your machine's GFLOPS.
:::
