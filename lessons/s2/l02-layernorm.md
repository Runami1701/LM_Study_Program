---
id: s2-l02
title: LayerNorm (forward + backward)
stage: s2
order: 2
prereqs: [s2-l01]
files: [workspace/stage2/layers.py]
functions: [layernorm, LayerNorm]
tests: tests/s2/test_l02_layernorm.py
est_minutes: 75
---

::: goal
You will rescale each position's vector to mean 0 and variance 1, then stretch and shift it with learned $\gamma$ and $\beta$: `(..., d)` → `(..., d)`. You write it as one operation and derive its backward pass slowly, one step at a time.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | one vector (one position), $x = (x_1, \dots, x_d)$ |
| $d$ | its length (the model width) |
| $\mu$ | the mean of its entries, $\mu = \frac{1}{d}\sum_j x_j$ |
| $v$ | the variance, $v = \frac{1}{d}\sum_j (x_j - \mu)^2$ |
| $\epsilon$ | a small number, $10^{-5}$, that prevents division by zero |
| $\sigma$ | $\sqrt{v + \epsilon}$ |
| $\hat x$ | the normalized vector, $\hat x_i = (x_i - \mu)/\sigma$ |
| $\gamma, \beta$ | learned stretch and shift, shape $(d,)$ |
| $y$ | the output, $y_i = \gamma_i \hat x_i + \beta_i$ |
| $L$ | the loss |
| $i, j, k$ | positions inside the vector |
| $\delta_{ij}$ | 1 if $i = j$, otherwise 0 |
| $\overline{a}$ | the mean of a vector's entries, $\overline{a} = \frac{1}{d}\sum_i a_i$ |
| $\odot$ | entry-by-entry product |
| $d\hat x$, $dx$, $dy$ | gradients of $L$ with respect to $\hat x$, $x$, $y$ |
| $g$ | the gradient arriving from above |
:::

::: numpy
### Mean and variance of the last axis, kept as columns (n-l04)

```python
x = np.array([[1., 2., 6.]])
print(x.mean(axis=-1, keepdims=True))      # [[3.]]
print(x.var(axis=-1, keepdims=True))       # [[4.66666667]]   divides by d, as in the formula
print(1 / np.sqrt(x.var(axis=-1, keepdims=True)))   # [[0.46291005]]   this is 1 / sigma (without epsilon)
```

`axis=-1` means the last axis, so the same code works for `(d,)`, `(B, d)` and `(B, T, d)` arrays: each position is normalized on its own.
:::

::: concept
### Why normalize

Deep networks suffer when their numbers drift in size from layer to layer: numbers that are too large saturate or explode, numbers that are too small fade away. This is the variance argument of s1-l09, repeated through dozens of steps. LayerNorm resets every position's vector to a standard size before each part of a block. Then $\gamma$ and $\beta$ let the model choose the size it actually wants.

### Forward

For one vector $x$ of length $d$:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j, \qquad v = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2, \qquad \sigma = \sqrt{v + \epsilon}
$$

$$
\hat x_i = \frac{x_i - \mu}{\sigma}, \qquad y_i = \gamma_i \hat x_i + \beta_i
$$

**Worked example** ($d = 3$, $x = [1, 2, 6]$, ignoring $\epsilon$): $\mu = 3$; the differences from the mean are $[-2, -1, 3]$; $v = \frac{4 + 1 + 9}{3} = 4.667$; $\sigma = 2.160$;

$$
\hat x = \left[\tfrac{-2}{2.160}, \tfrac{-1}{2.160}, \tfrac{3}{2.160}\right] = [-0.926, -0.463, 1.389]
$$

which has mean 0 and variance 1. In a batch, this happens separately for each of the $B \cdot T$ positions, over the last axis.

### Backward through $\gamma$ and $\beta$

$y_i = \gamma_i \hat x_i + \beta_i$, and every position shares the same $\gamma_i$ and $\beta_i$. So, as for every shared parameter, sum over the positions:

$$
d\gamma_i = \sum_{\text{positions}} dy_i\, \hat x_i, \qquad d\beta_i = \sum_{\text{positions}} dy_i
$$

The gradient that reaches $\hat x$ is $d\hat x_i = dy_i\, \gamma_i$.

### Backward through the normalization: the hard part

Every $\hat x_i$ depends on **every** $x_j$, through $\mu$ and $\sigma$. So we need $\frac{\partial \hat x_i}{\partial x_j}$. First, three small derivatives:

$$
\frac{\partial \mu}{\partial x_j} = \frac{1}{d}
$$

$$
\frac{\partial v}{\partial x_j} = \frac{1}{d}\sum_k 2(x_k - \mu)\left(\delta_{kj} - \frac{1}{d}\right) = \frac{2}{d}(x_j - \mu) - \frac{2}{d^2}\underbrace{\sum_k (x_k - \mu)}_{=\,0} = \frac{2}{d}(x_j - \mu)
$$

$$
\frac{\partial \sigma}{\partial x_j} = \frac{1}{2\sigma}\frac{\partial v}{\partial x_j} = \frac{x_j - \mu}{d\,\sigma} = \frac{\hat x_j}{d}
$$

Now the quotient $\hat x_i = \frac{x_i - \mu}{\sigma}$, using $\frac{\partial (x_i - \mu)}{\partial x_j} = \delta_{ij} - \frac{1}{d}$:

$$
\begin{aligned}
\frac{\partial \hat x_i}{\partial x_j}
&= \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d}\right) - \frac{x_i - \mu}{\sigma^2}\cdot\frac{\hat x_j}{d} \\
&= \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d} - \frac{\hat x_i \hat x_j}{d}\right)
\end{aligned}
$$

Chain rule, summing over all $i$, because every output depends on $x_j$:

$$
\begin{aligned}
dx_j &= \sum_i d\hat x_i \, \frac{\partial \hat x_i}{\partial x_j} \\
&= \frac{1}{\sigma}\left(\sum_i d\hat x_i\,\delta_{ij} - \frac{1}{d}\sum_i d\hat x_i - \frac{\hat x_j}{d}\sum_i d\hat x_i\,\hat x_i\right) \\
&= \frac{1}{\sigma}\left(d\hat x_j - \overline{d\hat x} - \hat x_j\,\overline{d\hat x \odot \hat x}\right)
\end{aligned}
$$

That gives the short form, now derived:

$$
dx = \frac{1}{\sigma}\left(d\hat x - \overline{d\hat x} - \hat x\;\overline{d\hat x \odot \hat x}\right)
$$

How to read it: take the incoming gradient and remove its mean, because shifting all $x_j$ by the same amount changes nothing after normalization. Then remove its part along $\hat x$, because scaling $x$ changes nothing either. Finally divide by $\sigma$.

**Numeric check** with $x = [1, 2, 6]$ and $d\hat x = [1, 0, 0]$: $\overline{d\hat x} = \frac{1}{3}$ and $\overline{d\hat x \odot \hat x} = \frac{-0.926}{3} = -0.309$, so

$$
dx = \frac{1}{2.160}\Big([1, 0, 0] - 0.333 + 0.309 \cdot [-0.926, -0.463, 1.389]\Big) = [0.176, -0.220, 0.044]
$$

Numerical gradients give $[0.1763, -0.2204, 0.0441]$. The entries add up to 0, as the "remove the mean" reading predicts.

### In code

All the means are over the last axis with `keepdims=True`, so one formula handles `(B, T, d)` at once. Save $\hat x$ and $\frac{1}{\sigma}$ in the forward pass; the backward needs nothing else except $\gamma$. Use `unbroadcast` for $d\gamma$ and $d\beta$.
:::

::: think
Q: Why is $\sum_k (x_k - \mu) = 0$, and where did that save work in the derivation?
A: $\sum_k x_k - d\mu = d\mu - d\mu = 0$. It removed the second term of $\frac{\partial v}{\partial x_j}$: moving $x_j$ also moves $\mu$, but that effect on the variance cancels out.
:::

::: think
Q: Explain without algebra why each row of $dx$ adds up to zero.
A: Adding the same number to every $x_j$ shifts $\mu$ by that number and leaves $\hat x$, and so the loss, unchanged. The sum of $dx$ is the loss's rate of change in exactly that direction, so it is 0.
:::

::: think
Q: LayerNorm forces mean 0 and variance 1. Doesn't that limit what the model can learn?
A: $\gamma$ and $\beta$ can undo it when that helps: with $\gamma_i = \sigma$ and $\beta_i = \mu$ the input comes back (roughly, since they are the same for every position). The normalization fixes the size first, which keeps training stable, and the learned stretch and shift keep the flexibility.
:::

::: hint level=1
Forward: two means over the last axis, then normalize, stretch and shift. Backward: three gradients. $\gamma$ and $\beta$ take one line each, plus `unbroadcast`; $x$ uses the boxed formula.
:::

::: hint level=2
`x.data.mean(axis=-1, keepdims=True)` and `.var(axis=-1, keepdims=True)`. Keep `rstd = 1 / np.sqrt(var + eps)` and `xhat` for the backward function. In the backward: `dxhat = g * gamma.data`, then the means of `dxhat` and of `dxhat * xhat` over the last axis, with `keepdims=True`.
:::

::: hint level=3
```pseudo
layernorm(x, gamma, beta, eps):
    mu, var ← mean and variance of x over the last axis (kept)
    rstd ← 1 / sqrt(var + eps);  xhat ← (x − mu) · rstd
    out ← Tensor(xhat · gamma + beta, parents (x, gamma, beta))
    backward:
        gamma.grad += unbroadcast(g · xhat, gamma's shape)
        beta.grad  += unbroadcast(g, beta's shape)
        dxhat ← g · gamma
        x.grad += rstd · (dxhat − mean(dxhat) − xhat · mean(dxhat · xhat))   # means over the last axis
```
:::

::: attempt
The editor below opens `workspace/stage2/layers.py`: fill in `layernorm` and `LayerNorm`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage2/layers.py#layernorm

- `rstd` and `xhat` are computed once in the forward pass and remembered by the backward function, so nothing is recomputed.
- The `x.grad` line is the boxed formula, applied to every position at once through means with `keepdims=True`.
- `unbroadcast` adds $d\gamma$ and $d\beta$ up over all positions, because every position shares $\gamma$ and $\beta$.

@include solutions/stage2/layers.py#LayerNorm
:::

::: breakit
Replace the backward of $x$ with the naive `x.grad += g * gamma.data * rstd`, which treats $\mu$ and $\sigma$ as constants, and run the check. The gradient check fails with relative errors of about 0.1 to 1: you dropped the two terms that remove the mean and the part along $\hat x$.
:::

::: recap
- LayerNorm rescales each position to mean 0 and variance 1, then stretches by $\gamma$ and shifts by $\beta$.
- You derived $\frac{\partial \hat x_i}{\partial x_j} = \frac{1}{\sigma}(\delta_{ij} - \frac{1}{d} - \frac{\hat x_i \hat x_j}{d})$ and from it $dx = \frac{1}{\sigma}(d\hat x - \overline{d\hat x} - \hat x\,\overline{d\hat x \odot \hat x})$.
- You checked it on $x = [1, 2, 6]$ by hand and with numerical gradients.
:::
