---
id: a-l01
title: The Tensor class
stage: a
order: 1
prereqs: [s1-boss]
files: [workspace/autograd/tensor.py]
functions: [Tensor]
tests: tests/a/test_l01_tensor.py
est_minutes: 45
---

::: goal
You will start an autograd engine: a tool that computes backward passes for you. Its core is a `Tensor`: an array that also stores its gradient, the tensors it was computed from, and a small function that sends its gradient back to them. You write `+`, `*` and `zero_grad`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| node | one Tensor in the graph of computations |
| `data` | the node's value (a NumPy array) |
| `grad` | $\frac{\partial L}{\partial \text{data}}$, with the same shape as `data` |
| `_prev` | the node's parents: the tensors it was computed from |
| `_backward` | a function that adds this node's share of the gradient into its parents' `grad` |
| $L$ | the final number (the loss) that `grad` refers to |
| $x, a, b, c, f$ | example tensors |
| $g$ | the gradient $\frac{\partial L}{\partial c}$ arriving at an output $c$ |
| leaf | a node with no parents: an input or a parameter |
:::

::: numpy
### A function defined inside a function (a closure)

A function defined inside another function can use the outer function's variables, even after the outer function has finished:

```python
def make_adder(n):
    def add(x):
        return x + n          # uses n from make_adder
    return add

f = make_adder(10)
print(f(5))                   # 15
```

Each Tensor operation uses this. `__add__` defines a small function `_backward` that remembers the two inputs and the output, and stores it on the output. Later, when the output's gradient is known, calling `out._backward()` uses those remembered tensors.

### Adding into an array

`+=` adds into the existing array. Two additions add up:

```python
g = np.zeros(2)
g += np.array([1., 2.])
g += np.array([1., 2.])
print(g)                      # [2. 4.]
```

### Methods with special names

Python calls `a.__add__(b)` when you write `a + b`, and `a.__mul__(b)` when you write `a * b`. Defining these two methods makes `+` and `*` work on Tensors.
:::

::: concept
### Why an engine

In Stage 1, every forward function had a backward twin, written by hand, and `backward` had to call them in exactly the reverse order. A transformer has dozens of steps; one wrong order or forgotten cache is a silent bug. An autograd engine records the forward computation while it runs, and replays it backwards automatically. You still derive each operation's derivative once, but you never wire a backward pass by hand again.

### The graph of computations

Every operation creates a new node whose parents are its inputs. For $c = a \cdot b + a$:

| node | value | parents | operation |
| --- | --- | --- | --- |
| `a`, `b` | leaves | none | none |
| `t` | $a \cdot b$ | `a`, `b` | `*` |
| `c` | $t + a$ | `t`, `a` | `+` |

The arrows point from inputs to outputs and never form a loop.

### Each node knows only its own derivative

When `c = a + b` is created, its `_backward` function remembers `a`, `b` and `c`. Later, once `c.grad` holds $g = \frac{\partial L}{\partial c}$, calling `c._backward()` applies the chain rule for this one step:

$$
\frac{\partial L}{\partial a} \mathrel{+}= g \cdot \frac{\partial c}{\partial a} = g \cdot 1, \qquad
\frac{\partial L}{\partial b} \mathrel{+}= g \cdot 1
$$

For `c = a * b` (entry by entry), $\frac{\partial c}{\partial a} = b$ and $\frac{\partial c}{\partial b} = a$:

$$
\frac{\partial L}{\partial a} \mathrel{+}= g \cdot b, \qquad \frac{\partial L}{\partial b} \mathrel{+}= g \cdot a
$$

Numbers: with $a = [2, 3]$, $b = [5, -1]$ and $g = [1, 2]$, the call adds $[5, -2]$ to `a.grad` and $[2, 6]$ to `b.grad`.

### Why `+=`: a node used twice

Take $f = x \cdot x$ at $x = 3$. Then $\frac{df}{dx} = 2x = 6$. In the graph, `f` is a `*` node whose two parents are **the same object** `x`. Its `_backward` runs `self.grad += g * other.data` and `other.grad += g * self.data`, and both lines add into `x.grad`:

$$
\text{x.grad} = 0 + 1 \cdot 3 + 1 \cdot 3 = 6 \quad \checkmark
$$

With `=` instead of `+=`, the second line would overwrite the first, and the result would be 3. The rule from s1-l06, "contributions from several paths add up", becomes one character of code. The same holds when `x` feeds two different nodes, as `a` does in $c = a \cdot b + a$.

### What a Tensor stores

- `data`: float32, except that a float64 array stays float64 (gradient checks need it). Lists, ints and Python numbers become float32.
- `grad`: zeros with the same shape and dtype.
- `_prev`: a tuple of parent Tensors (empty for a leaf).
- `_backward`: a function with no arguments. For a leaf it does nothing.

Plain numbers, as in `a * 3.0`, are wrapped into constant Tensors by the given helper `_wrap`, so every operation sees two Tensors.

In this lesson both inputs of `+` and `*` have the same shape. Lesson a-l02 adds broadcasting (`(B, m) + (m,)`), and a-l03 writes `backward()`, which calls every `_backward` in the right order. Until then, the checks set `out.grad` themselves and call `out._backward()`.
:::

::: think
Q: For $c = a \cdot b + a$, what is $\frac{\partial c}{\partial a}$, and which two `_backward` calls add into `a.grad`?
A: $\frac{\partial c}{\partial a} = b + 1$. The `+` node adds $g \cdot 1$ (through its parent `a`). The `*` node, after receiving $g$ through `t`, adds $g \cdot b$. Both land in `a.grad` because of `+=`.
:::

::: think
Q: Why does each node store a function, instead of a label like `"*"` that a big `if` statement in the engine would look up?
A: The function remembers exactly the values its derivative needs, at the moment the node is created. The engine never needs to know which operations exist: adding a new operation means writing one function, not editing a central `if` statement.
:::

::: think
Q: Why keep float64 data as float64 instead of always converting to float32?
A: Gradient checks (s1-l08) run in float64; converting would bring back the 1% rounding errors. Parameters and other model arrays are created as float32 on purpose.
:::

::: hint level=1
`__init__` stores four things. `__add__` and `__mul__` compute the output value with NumPy, create the output Tensor with both inputs as parents, define a small inner function that updates both parents' gradients from the output's gradient, and store that function on the output.
:::

::: hint level=2
`np.asarray(data)`, then `.astype(np.float32)` unless the dtype is already float32 or float64; `np.zeros_like` for the gradient. Inside `__add__`, the inner `def _backward(): ...` can read `out.grad`: `out` exists by the time the function runs. Wrap `other` first with `self._wrap(other)`.
:::

::: hint level=3
```pseudo
__init__(data, _prev, _op):
    data ← data as an array (float32 unless it is already float32 or float64)
    grad ← zeros like data;  _prev ← tuple(_prev);  _backward ← a function that does nothing;  _op ← _op

__mul__(other):
    other ← other wrapped as a Tensor
    out ← Tensor(self.data · other.data, parents (self, other), "*")
    define _backward():
        self.grad  += out.grad · other.data
        other.grad += out.grad · self.data
    out._backward ← that function
    return out

zero_grad(): grad ← zeros like data
```
:::

::: attempt
File: `workspace/autograd/tensor.py` — fill in `Tensor.__init__`, `__add__`, `__mul__` and `zero_grad`.
Check: `python -m course check a-l01`
:::

::: solution
@include solutions/autograd/tensor.py#Tensor

- `__init__` keeps float32 and float64 arrays as they are and converts everything else to float32. `grad` copies the data's shape and dtype.
- In `__add__` and `__mul__`, the inner function remembers `self`, `other` and `out`. It runs later, when `out.grad` is known.
- Every update uses `+=`, so a tensor used twice (or by two nodes) collects both contributions.
- The `unbroadcast(...)` calls belong to a-l02. With equal shapes they return the gradient unchanged, so they do no harm now.
- `backward` hands the work to the module-level function you write in a-l03.
:::

::: breakit
Change the `+=` in `__mul__` to `=`. Then evaluate $f = x \cdot x$ at $x = 3$ by calling `f.grad = np.array(1.0); f._backward()` yourself. You get `x.grad == 3.0` instead of 6: the second assignment overwrote the first contribution.
:::

::: recap
- A Tensor stores `data`, `grad`, its parents, and a `_backward` function that knows its local derivative.
- `+` passes the gradient through unchanged; `*` multiplies it by the other input.
- Gradients add up with `+=`, which handles every reused node automatically.
:::
