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
Start an autograd engine: a `Tensor` that holds an array, a gradient of the same shape, the tensors it was computed from, and a small function that pushes its gradient back to them; implement `+`, `*` and `zero_grad`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| node | one Tensor in the computational graph |
| `data` | the node's value (a NumPy array) |
| `grad` | $\frac{\partial L}{\partial \text{data}}$, same shape as `data` |
| `_prev` | the node's parents: the tensors it was computed from |
| `_backward` | a closure that adds this node's contribution into its parents' `grad` |
| $L$ | the final scalar (the loss) that `grad` refers to |
| $x, a, b, c, f$ | example tensors |
| $g$ | the upstream gradient $\frac{\partial L}{\partial c}$ arriving at an output $c$ |
| leaf | a node with no parents: an input or a parameter |
:::

::: concept
### Why an engine

In stage 1 every forward function had a hand-written backward twin, and the model's `backward` called them in exactly the reverse order. For a transformer that is dozens of steps; one wrong order or forgotten cache is a silent bug. An autograd engine records the forward computation as it happens and replays it backwards automatically. You still derive every local derivative (once, per operation), but never again wire up a backward pass by hand.

### The computational graph

Every operation creates a new node whose parents are its inputs. For $c = a \cdot b + a$:

| node | value | parents | op |
| --- | --- | --- | --- |
| `a`, `b` | leaves | — | — |
| `t` | $a \cdot b$ | `a`, `b` | `*` |
| `c` | $t + a$ | `t`, `a` | `+` |

The graph is a DAG (directed acyclic graph): arrows point from inputs to outputs and never loop.

### Each node knows only its local derivative

When `c = a + b` is created, its `_backward` closure captures `a`, `b` and `c` itself. Later, once `c.grad` holds $g = \frac{\partial L}{\partial c}$, calling `c._backward()` applies the chain rule for this one step:

$$
\frac{\partial L}{\partial a} \mathrel{+}= g \cdot \frac{\partial c}{\partial a} = g \cdot 1, \qquad
\frac{\partial L}{\partial b} \mathrel{+}= g \cdot 1
$$

For `c = a * b` (elementwise), $\frac{\partial c}{\partial a} = b$ and $\frac{\partial c}{\partial b} = a$:

$$
\frac{\partial L}{\partial a} \mathrel{+}= g \cdot b, \qquad \frac{\partial L}{\partial b} \mathrel{+}= g \cdot a
$$

Numbers: $a = [2, 3]$, $b = [5, -1]$, $g = [1, 2]$ gives `a.grad += [5, -2]` and `b.grad += [2, 6]`.

### Why `+=`: a node used twice

Take $f = x \cdot x$ with $x = 3$. Then $\frac{df}{dx} = 2x = 6$. In the graph, `f` is a `*` node whose two parents are **the same object** `x`. Its `_backward` runs `self.grad += g * other.data` and `other.grad += g * self.data`, and both lines target `x.grad`:

$$
\text{x.grad} = 0 + 1 \cdot 3 + 1 \cdot 3 = 6 \quad \checkmark
$$

With `=` instead of `+=`, the second line would overwrite the first and give 3. The general rule from s1-l06, "contributions from several paths add", becomes one character in code. The same holds when `x` feeds two different nodes, as `a` does in $c = a \cdot b + a$.

### What a Tensor stores

- `data`: float32, except that a float64 array stays float64 (gradient checks need it). Lists, ints and Python floats become float32.
- `grad`: zeros of the same shape and dtype.
- `_prev`: a tuple of parent Tensors (empty for leaves).
- `_backward`: a function of no arguments; for a leaf it does nothing.

Plain numbers such as `a * 3.0` are wrapped into constant Tensors (the given `_wrap` helper) so every operation sees two Tensors.

In this lesson both operands have the same shape. Lesson a-l02 adds broadcasting (`(B, m) + (m,)`), and a-l03 writes `backward()`, which calls every `_backward` in the right order. Until then, the checks set `out.grad` and call `out._backward()` themselves.
:::

::: think
Q: For $c = a \cdot b + a$, what is $\frac{\partial c}{\partial a}$, and which two `_backward` calls contribute to `a.grad`?
A: $\frac{\partial c}{\partial a} = b + 1$. The `+` node adds $g \cdot 1$ (from its parent `a`), and the `*` node, after receiving $g$ through `t`, adds $g \cdot b$. Both land in `a.grad` because of `+=`.
:::

::: think
Q: Why does each node store a closure instead of, say, a string like `"*"` and a big `if` in the engine?
A: The closure captures exactly the values its derivative needs (the other operand, the output), at the moment the node is created. The engine then never needs to know what operations exist: adding a new op means writing one function, not editing a central switch.
:::

::: think
Q: Why keep float64 data as float64 instead of always converting to float32?
A: Gradient checks (s1-l08) run in float64; converting would bring back the ~1% rounding errors. Everything else (parameters, activations) is created as float32 on purpose.
:::

::: hint level=1
`__init__` stores four things. `__add__` and `__mul__` compute the output value with NumPy, create the output Tensor with both inputs as parents, define a small inner function that updates both parents' grads from the output's grad, and attach it to the output.
:::

::: hint level=2
`np.asarray(data)` then `.astype(np.float32)` unless the dtype is already float32 or float64; `np.zeros_like` for the grad. Inside `__add__`, `def _backward(): ...` can read `out.grad` because `out` exists by the time it runs (closures look variables up when called). Wrap `other` first with `self._wrap(other)`.
:::

::: hint level=3
```pseudo
__init__(data, _prev, _op):
    data ← data as an array (float32 unless already float32/float64)
    grad ← zeros like data;  _prev ← tuple(_prev);  _backward ← do nothing;  _op ← _op

__mul__(other):
    other ← wrap other as a Tensor
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
File: `workspace/autograd/tensor.py` — implement `Tensor.__init__`, `__add__`, `__mul__`, `zero_grad`.
Check: `python -m course check a-l01`
:::

::: solution
@include solutions/autograd/tensor.py#Tensor

- `__init__` keeps float32/float64 arrays as they are and converts everything else to float32; `grad` mirrors the data's shape and dtype.
- In `__add__` and `__mul__` the closure captures `self`, `other` and `out`; it runs later, when `out.grad` is known.
- Every update is `+=`, so a tensor used twice (or by two nodes) collects both contributions.
- The `unbroadcast(...)` calls are the a-l02 extension: with equal shapes they return the gradient unchanged, so they are harmless now.
- `backward` delegates to the module-level function you write in a-l03.
:::

::: breakit
Change the `+=` in `__mul__` to `=` and evaluate $f = x \cdot x$ at $x = 3$ by hand-calling `f.grad = np.array(1.0); f._backward()`. Expect `x.grad == 3.0` instead of the correct 6: the second assignment overwrote the first contribution.
:::

::: recap
- A Tensor stores `data`, `grad`, its parents and a `_backward` closure with its local derivative.
- `+` passes the gradient through unchanged; `*` multiplies it by the other operand.
- Gradients accumulate with `+=`, which handles every reused node automatically.
:::
