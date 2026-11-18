---
id: s2-l03
title: GELU
stage: s2
order: 3
prereqs: [s2-l02]
files: [workspace/stage2/layers.py]
functions: [gelu]
tests: tests/s2/test_l03_gelu.py
est_minutes: 35
---

::: goal
You will write GELU, the nonlinear function inside the transformer's MLP (the tanh version), as one operation with a derived backward pass. It must keep float32.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | an input value (the function works entry by entry) |
| $u$ | the inner part $c\,(x + 0.044715\,x^3)$ |
| $c$ | the constant $\sqrt{2/\pi} \approx 0.7979$ |
| $t$ | $\tanh u$ |
| $\Phi$ | the standard normal CDF: $\Phi(x) = P(Z \le x)$ for $Z \sim N(0,1)$ |
| $\pi$ | 3.14159… |
| $\mathrm{ReLU}$ | $\max(0, x)$ |
| $g$ | the gradient arriving from above |
| $\frac{d}{dx}$ | derivative with respect to $x$ |
:::

::: numpy
### A NumPy constant can change the dtype

`np.sqrt` returns a NumPy float64 value, which turns a float32 array into float64 (s0-l00). `math.sqrt` returns a plain Python number, which does not:

```python
import math
a = np.ones(3, dtype=np.float32)
print((a * np.sqrt(2 / np.pi)).dtype)       # float64   silently upcast
print((a * math.sqrt(2 / math.pi)).dtype)   # float32
```

### Cubes

`a * a * a` gives the same result as `a ** 3` and is faster: NumPy's general power function does more work than two multiplications.
:::

::: concept
### What GELU is

GELU multiplies each input by the probability that a standard normal variable falls below it:

$$
\mathrm{GELU}(x) = x\,\Phi(x)
$$

For large positive $x$, $\Phi(x) \approx 1$, so the output is about $x$, as with ReLU. For large negative $x$, $\Phi(x) \approx 0$, so the output is about 0. In between the curve is smooth, with a small dip below zero: its minimum is about $-0.170$, at $x \approx -0.75$.

$\Phi$ has no simple formula, so GPT-2 (and we) use a tanh approximation:

$$
\mathrm{GELU}(x) \approx \frac{1}{2} x \left(1 + \tanh u\right), \qquad u = c\left(x + 0.044715\, x^3\right), \quad c = \sqrt{2/\pi}
$$

| $x$ | $-3$ | $-1$ | $-0.5$ | $0$ | $0.5$ | $1$ | $3$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GELU | $-0.004$ | $-0.159$ | $-0.154$ | $0$ | $0.346$ | $0.841$ | $2.996$ |
| ReLU | $0$ | $0$ | $0$ | $0$ | $0.5$ | $1$ | $3$ |

Unlike ReLU, GELU has a nonzero gradient for small negative inputs (so no unit "dies" with a gradient of exactly zero), and it has no corner at 0.

### The derivative

Write $\mathrm{GELU}(x) = \frac{1}{2}x(1 + t)$ with $t = \tanh u$. By the product rule:

$$
\frac{d}{dx}\mathrm{GELU}(x) = \frac{1}{2}(1 + t) + \frac{1}{2}x\,\frac{dt}{dx}
$$

By the chain rule, with $\frac{d}{du}\tanh u = 1 - t^2$ (s1-l06) and $\frac{du}{dx} = c\,(1 + 3 \cdot 0.044715\,x^2)$:

$$
\frac{dt}{dx} = (1 - t^2)\cdot c\,(1 + 0.134145\,x^2)
$$

$$
\frac{d}{dx}\mathrm{GELU}(x) = \frac{1}{2}(1 + t) + \frac{1}{2}\,x\,(1 - t^2)\,c\,(1 + 0.134145\,x^2)
$$

Check at $x = 0$: $t = 0$, so the slope is $\frac{1}{2}$, as for $x\,\Phi(x)$, since $\Phi(0) = \frac{1}{2}$. The backward pass multiplies this slope by $g$, entry by entry. Save $t$ in the forward pass.

### A float32 trap that really happened

The obvious constant `np.sqrt(2 / np.pi)` is a NumPy **float64** value, and it turns the whole float32 array into float64 (see the box above). Nothing crashes; it is just slow. While this course was being built, exactly this line made the reference GELU take 0.6 seconds of a 1.7-second training step. Use `math.sqrt(2 / math.pi)`. The check tests the output's dtype.
:::

::: think
Q: Why might a smooth function like GELU train better than ReLU?
A: ReLU's gradient is exactly 0 for every negative input, so a unit whose inputs are all negative stops learning ("dies"). GELU still passes a small gradient there, and its smooth shape makes the loss smoother too. In practice, transformers train a little better with it.
:::

::: think
Q: Where is GELU's derivative negative, and why is that not a problem?
A: For $x$ below about $-0.75$. There the curve falls from about 0 (at very negative $x$) down to its minimum $-0.170$, so it decreases as $x$ grows. Those slopes are small, and a negative derivative only means that the gradient pushes the input the other way; gradient descent works with either sign.
:::

::: hint level=1
Compute $u$, $t = \tanh u$ and the output in the forward pass. The backward function applies the derivative formula using the saved $t$.
:::

::: hint level=2
`c = math.sqrt(2.0 / math.pi)`; `u = c * (a + 0.044715 * a * a * a)` with `a = x.data`; `np.tanh`. The derivative needs `a`, `t` and `du = c * (1 + 3 * 0.044715 * a * a)`.
:::

::: hint level=3
```pseudo
gelu(x):
    a ← x.data;  u ← c · (a + 0.044715 · a³);  t ← tanh(u)
    out ← Tensor(0.5 · a · (1 + t), parents (x,))
    backward:
        du ← c · (1 + 3 · 0.044715 · a²)
        x.grad += g · (0.5 · (1 + t) + 0.5 · a · (1 − t²) · du)
```
:::

::: attempt
The editor below opens `workspace/stage2/layers.py`: fill in `gelu`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage2/layers.py#gelu

- `_C` is a Python number (see the top of the module), so the output stays float32.
- The backward function remembers `t` and `a`; the backward itself is the derived formula, in one line.
:::

::: breakit
Replace `_C` with `np.sqrt(2.0 / np.pi)` and time a forward pass on a `(32, 128, 512)` float32 input (`time.perf_counter()` around `gelu(x)`). The output dtype becomes float64, and the call takes several times longer.
:::

::: recap
- GELU$(x) = x\Phi(x)$, computed with a tanh approximation; it is smooth and has no dead units.
- Its derivative follows from the product and chain rules: $\frac{1}{2}(1 + t) + \frac{1}{2}x(1 - t^2)c(1 + 0.134145x^2)$.
- NumPy float64 constants silently upcast; use Python numbers for constants.
:::
