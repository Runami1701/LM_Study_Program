---
id: s0-l00
title: Setup & NumPy pitfalls
stage: s0
order: 0
prereqs: []
files: [workspace/stage0/numpy_basics.py]
functions: [to_f32, matmul_gflops]
tests: tests/s0/test_l00_setup.py
est_minutes: 30
---

::: goal
Get the course running, learn the three NumPy habits this course relies on (watch the dtype, know views from copies, know when memory is contiguous), and measure how fast your machine multiplies matrices: `to_f32(*arrays) → tuple of float32 arrays` and `matmul_gflops(n, reps, rng) → float`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| dtype | an array's element type, e.g. `float32` (4 bytes per number) or `float64` (8 bytes) |
| $\varepsilon$ | machine epsilon: the gap between 1 and the next representable number |
| $n$ | matrix size: we multiply two $(n, n)$ matrices |
| $A, B, C$ | the matrices in $C = AB$; $A_{il}$ is the entry in row $i$, column $l$ |
| $i, j, l$ | indices: $i$ a row of $C$, $j$ a column of $C$, $l$ runs along the shared dimension |
| FLOP | one floating-point operation (one multiply or one add) |
| GFLOPS | $10^9$ FLOPs per second |
| $t$ | measured wall-clock time in seconds |
| reps | number of timed repetitions |
| BLAS | the optimized linear-algebra library NumPy calls for `@` (OpenBLAS, MKL, ...) |
:::

::: concept
### 1. Get the course running

From the repository folder, once:

1. `pip install -e .` installs the `course` command-line tool and the dependencies.
2. `python -m course data` downloads Tiny Shakespeare (about 1.1 MB) into `data/`.
3. `python -m course serve` opens this site at `http://localhost:8000`.

Then, for every lesson: read, write the code in the file named under **Your attempt**, and run `python -m course check <lesson-id>`. A passing check unlocks the solution here within a few seconds.

### 2. float32 vs float64

A float stores a sign, an exponent and a fraction. The fraction width decides precision:

| dtype | bytes | fraction bits | machine epsilon $\varepsilon$ | about |
| --- | --- | --- | --- | --- |
| `float32` | 4 | 23 | $2^{-23} \approx 1.19 \times 10^{-7}$ | 7 significant digits |
| `float64` | 8 | 52 | $2^{-52} \approx 2.22 \times 10^{-16}$ | 16 significant digits |

Concrete consequence: in float32, `1 + 1e-8` equals exactly `1`, because $10^{-8}$ is smaller than half the gap $\varepsilon$ above 1, so the sum rounds back to 1. In float64 it does not.

Why we train in **float32** anyway:

- **Memory.** A `(1024, 1024)` matrix is 4 MiB in float32 and 8 MiB in float64.
- **Speed.** The CPU's vector instructions process twice as many float32 numbers per instruction, so matrix products run roughly twice as fast.
- **Precision is not the bottleneck.** Minibatch gradients are noisy at the level of percent, far above $10^{-7}$.

The one place we need float64 is **gradient checking** (s1-l08): there we subtract two nearly equal numbers, and float32's 7 digits are not enough.

### 3. Where float64 sneaks in

NumPy's default float type is float64, so these all give float64 unless you ask otherwise:

- `np.zeros(3)`, `np.ones(3)`, `np.empty(3)`, `np.array([0.1, 0.2])`
- `np.random.randn(3)`, and `rng.standard_normal(3)` without a `dtype=` argument

Mixing types follows NumPy 2's promotion rules. With `x` a float32 array:

| expression | result dtype | why |
| --- | --- | --- |
| `x * 0.5` | float32 | a plain Python float is "weak": it adopts the array's type |
| `x / math.sqrt(2)` | float32 | `math.sqrt` returns a plain Python float |
| `x / np.sqrt(2)` | **float64** | `np.sqrt(2)` returns a NumPy `float64` scalar, which is "strong" |
| `x + np.array(0.5)` | **float64** | a 0-d float64 array is still a float64 array |
| `x + np.float32(0.5)` | float32 | same type, nothing to promote |

Nothing warns you. One float64 constant in a forward pass silently turns every later array into float64: double the memory, half the speed. The habit: **create arrays with an explicit dtype, and check `.dtype` at function boundaries.**

### 4. BLAS: who actually multiplies

`a @ b` is not computed by NumPy's own loops: it calls BLAS, a heavily tuned, multi-threaded library. `np.show_config()` prints which one your NumPy uses (usually OpenBLAS on pip installs). This is why a NumPy matmul is fast enough to train real models on a CPU, and why the speed you measure below is mostly a property of BLAS and your CPU.

### 5. Views vs copies

Some operations return a **view**: a new array object that looks at the same memory. Others return a **copy**.

Worked example. Start with `a = [0, 1, 2, 3, 4, 5]`.

- `v = a[1:4]` (basic slicing) is a **view**. After `v[0] = 100`, `a` is `[0, 100, 2, 3, 4, 5]`.
- `c = a[[1, 2]]` (indexing with an integer list, "fancy indexing") is a **copy**. After `c[0] = -1`, `a` is unchanged.

| operation | view or copy |
| --- | --- |
| basic slicing `a[1:4]`, `a[:, 0]` | view |
| `a.reshape(...)`, `a.T` | view (when the memory layout allows it) |
| fancy indexing `a[idx]` with an integer array | copy |
| `a.astype(t)` | copy, always (by default) |
| `np.asarray(a, dtype=t)` | no copy if `a` is already an ndarray of type `t`; otherwise converts |

`np.shares_memory(a, b)` tells you which you have. Views are fast and cheap; the danger is writing through a view and changing the original by accident.

### 6. Contiguity

Memory is one long line of bytes. A float32 array of shape $(2, 3)$ is stored row by row ("C order"): `a[0,0], a[0,1], a[0,2], a[1,0], ...`. To step one row down you skip 3 numbers = 12 bytes; one column right, 4 bytes. These step sizes are the **strides**: `(12, 4)`.

`a.T` is a view with strides `(4, 12)`: the same memory read in a different order. It is no longer **C-contiguous** (`a.T.flags['C_CONTIGUOUS']` is `False`). Most NumPy functions handle that fine, but `reshape` of a non-contiguous array must copy, and some operations get slower. This matters in stage 2, when multi-head attention reshapes and transposes `(B, T, d)` arrays.

### 7. Counting the work in a matrix product

Let $A$ and $B$ be $(n, n)$ and $C = AB$. One entry of $C$ is

$$
C_{ij} = \sum_{l=1}^{n} A_{il} B_{lj}
$$

That sum has $n$ products, and adding $n$ numbers together takes $n - 1$ additions:

$$
\text{FLOPs per entry} = n + (n - 1) = 2n - 1
$$

$C$ has $n \cdot n = n^2$ entries, so

$$
\begin{aligned}
\text{FLOPs per product} &= n^2 (2n - 1) \\
&= 2n^3 - n^2 \\
&\approx 2n^3 \quad \text{(for large } n \text{, } n^2 \text{ is tiny next to } 2n^3\text{)}
\end{aligned}
$$

For $n = 1024$: $2 \cdot 1024^3 = 2{,}147{,}483{,}648 \approx 2.15 \times 10^9$ FLOPs. If `reps` products take $t$ seconds in total:

$$
\text{GFLOPS} = \frac{2n^3 \cdot \text{reps}}{t \cdot 10^9}
$$

Example: 5 products of size 1024 in $t = 0.1$ s give $\frac{2.15 \times 10^9 \cdot 5}{0.1 \cdot 10^9} \approx 107$ GFLOPS.

Two measuring habits: use `time.perf_counter()` (a high-resolution clock meant for timing), and do one **warm-up** product first, because the very first call pays one-off costs (starting BLAS threads, touching fresh memory) that are not part of the steady-state speed.

`python -m course check s0-l00` stores your number in `progress.json`; in stage 3 you will use it to estimate how long an overnight training run takes.
:::

::: think
Q: You write `h = x @ W1 / np.sqrt(fan_in)`, with `x` and `W1` both float32 and `fan_in` an int. What is `h.dtype`, and what are two ways to fix it?
A: `float64`. `np.sqrt(fan_in)` returns a NumPy `float64` scalar, which is a strong type in NumPy 2 and promotes the float32 product. Fixes: use `math.sqrt(fan_in)` (a Python float is weak), or `np.float32(np.sqrt(fan_in))`. Even better: fold the scale into `W1` once at initialization, so the forward pass has no constant at all.
:::

::: think
Q: Why count $2n^3$ FLOPs for a matrix product rather than $n^3$?
A: Each of the $n^3$ terms $A_{il}B_{lj}$ needs one multiply and one add into the running sum. FLOPs count both. (CPUs often do the pair as one "fused multiply-add" instruction, but by convention it still counts as 2 FLOPs, which is also how the $6ND$ training-cost rule in stage 3 is counted.)
:::

::: think
Q: `to_f32` must return an input that is already float32 *without copying*. Why is that useful, and what is the risk?
A: Useful: no wasted time or memory when nothing needs converting (inside a training loop, a needless copy of every parameter each step adds up). Risk: the result shares memory with the input, so writing into the result in place also changes the caller's array. Know which of the two you have; `np.shares_memory` tells you.
:::

::: hint level=1
`to_f32`: there is a NumPy conversion that only does work when the dtype is wrong. `matmul_gflops`: make the matrices once, warm up, time all `reps` products together, then apply the FLOP formula.
:::

::: hint level=2
`np.asarray(a, dtype=...)` converts only when needed (unlike `.astype`, which always copies). A generator inside `tuple(...)` handles any number of arguments. For the matrices, `rng.standard_normal(shape, dtype=np.float32)` creates float32 directly; time with `time.perf_counter()` before and after the loop.
:::

::: hint level=3
```pseudo
to_f32(*arrays):
    return a tuple: for each a in arrays, a converted to float32 (no copy if already float32)

matmul_gflops(n, reps, rng):
    A, B ← two (n, n) float32 standard-normal matrices from rng
    A @ B                                   # warm-up, not timed
    t0 ← perf_counter()
    repeat reps times: A @ B
    seconds ← perf_counter() − t0
    return float( 2 · n³ · reps / seconds / 1e9 )
```
:::

::: attempt
File: `workspace/stage0/numpy_basics.py` — implement `to_f32` and `matmul_gflops`.
Check: `python -m course check s0-l00` (on a pass it also prints and saves your machine's GFLOPS).
:::

::: solution
@include solutions/stage0/numpy_basics.py#to_f32

- `np.asarray(a, dtype=np.float32)` returns `a` itself when it is already a float32 ndarray, so no copy. Otherwise (float64 array, int array, Python list, Python scalar) it builds a new float32 array.
- `tuple(... for a in arrays)` works for any number of arguments and always returns a tuple, so `(x,) = to_f32(x)` and `x, y = to_f32(x, y)` both unpack cleanly.

@include solutions/stage0/numpy_basics.py#matmul_gflops

- `rng.standard_normal(..., dtype=np.float32)` creates float32 directly. Creating float64 and converting would measure the same thing, but costs a pointless copy.
- The warm-up product absorbs one-off costs so they do not pollute the timing.
- Timing all `reps` products together (rather than each one) keeps the clock's own overhead negligible.
- `2.0 * n**3`: the `2.0` makes the arithmetic float, and the operation count is $2n^3$ per product, derived above.
- `float(...)` turns the NumPy scalar into a plain Python float, which is what the docstring promises.
:::

::: breakit
**Silent upcast.** In a Python shell: make `x = np.ones(3, dtype=np.float32)` and inspect `(x + np.array(0.5)).dtype`, `(x / np.sqrt(2)).dtype` and `(x * 0.5).dtype`. Expect `float64`, `float64`, `float32`. The first two are the bugs this course's dtype checks will catch.
:::

::: breakit
**Half precision, double speed.** Call your `matmul_gflops` once as written and once with the matrices created as float64 (`dtype=np.float64`). Expect the float64 number to be roughly half the float32 one: the CPU's vector registers hold half as many float64 numbers.
:::

::: recap
- You can install, serve and check lessons with `python -m course`.
- float32 is the model dtype; float64 creeps in through default constructors and NumPy float64 scalars, and you know the promotion table.
- You can tell views from copies (`np.shares_memory`) and know why contiguity matters for `reshape`.
- You derived the $2n^3$ FLOP count and measured your machine's GFLOPS.
:::
