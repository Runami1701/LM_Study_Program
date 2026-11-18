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
Normalize each position's vector to mean 0 and variance 1, then rescale with learned $\gamma, \beta$: `(..., d)` → `(..., d)`. Implement it as one op, and derive its backward pass slowly, step by step.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | one vector (one position), $x = (x_1, \dots, x_d)$ |
| $d$ | its length (model width) |
| $\mu$ | mean of the entries, $\mu = \frac{1}{d}\sum_j x_j$ |
| $v$ | (biased) variance, $v = \frac{1}{d}\sum_j (x_j - \mu)^2$ |
| $\epsilon$ | small constant, $10^{-5}$, avoids division by zero |
| $\sigma$ | $\sqrt{v + \epsilon}$ |
| $\hat x$ | normalized vector, $\hat x_i = (x_i - \mu)/\sigma$ |
| $\gamma, \beta$ | learned scale and shift, shape $(d,)$ |
| $y$ | output, $y_i = \gamma_i \hat x_i + \beta_i$ |
| $L$ | loss |
| $i, j, k$ | indices into the vector |
| $\delta_{ij}$ | Kronecker delta: 1 if $i = j$, else 0 |
| $\overline{a}$ | the mean over the vector's entries, $\overline{a} = \frac{1}{d}\sum_i a_i$ |
| $\odot$ | elementwise product |
| $d\hat x$, $dx$, $dy$ | gradients of $L$ with respect to $\hat x$, $x$, $y$ |
| $g$ | upstream gradient |
:::

::: concept
### Why normalize

Deep networks suffer when activations drift in scale from layer to layer: too large saturates or explodes, too small vanishes (s1-l09's variance argument, now repeated through dozens of operations). LayerNorm resets every position's vector to a standard scale before each sub-layer, and then lets the model choose the scale it wants through $\gamma$ and $\beta$.

### Forward

For one vector $x$ of length $d$:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j, \qquad v = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2, \qquad \sigma = \sqrt{v + \epsilon}
$$

$$
\hat x_i = \frac{x_i - \mu}{\sigma}, \qquad y_i = \gamma_i \hat x_i + \beta_i
$$

**Worked example** ($d = 3$, $x = [1, 2, 6]$, ignoring $\epsilon$): $\mu = 3$; deviations $[-2, -1, 3]$; $v = \frac{4 + 1 + 9}{3} = 4.667$; $\sigma = 2.160$;

$$
\hat x = \left[\tfrac{-2}{2.160}, \tfrac{-1}{2.160}, \tfrac{3}{2.160}\right] = [-0.926, -0.463, 1.389]
$$

with mean 0 and variance 1. In a batch, this happens independently for each of the $B \cdot T$ positions, over the last axis.

### Backward through $\gamma$ and $\beta$

$y_i = \gamma_i \hat x_i + \beta_i$, and $\gamma_i, \beta_i$ are shared by every position, so (sum over positions, as for every shared parameter):

$$
d\gamma_i = \sum_{\text{positions}} dy_i\, \hat x_i, \qquad d\beta_i = \sum_{\text{positions}} dy_i
$$

and the gradient arriving at $\hat x$ is $d\hat x_i = dy_i\, \gamma_i$.

### Backward through the normalization: the hard part

Every $\hat x_i$ depends on **every** $x_j$, through $\mu$ and $\sigma$. We need $\frac{\partial \hat x_i}{\partial x_j}$. Three small derivatives first:

$$
\frac{\partial \mu}{\partial x_j} = \frac{1}{d}
$$

$$
\frac{\partial v}{\partial x_j} = \frac{1}{d}\sum_k 2(x_k - \mu)\left(\delta_{kj} - \frac{1}{d}\right) = \frac{2}{d}(x_j - \mu) - \frac{2}{d^2}\underbrace{\sum_k (x_k - \mu)}_{=\,0} = \frac{2}{d}(x_j - \mu)
$$

$$
\frac{\partial \sigma}{\partial x_j} = \frac{1}{2\sigma}\frac{\partial v}{\partial x_j} = \frac{x_j - \mu}{d\,\sigma} = \frac{\hat x_j}{d}
$$

Now the quotient $\hat x_i = \frac{x_i - \mu}{\sigma}$, with $\frac{\partial (x_i - \mu)}{\partial x_j} = \delta_{ij} - \frac{1}{d}$:

$$
\begin{aligned}
\frac{\partial \hat x_i}{\partial x_j}
&= \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d}\right) - \frac{x_i - \mu}{\sigma^2}\cdot\frac{\hat x_j}{d} \\
&= \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d} - \frac{\hat x_i \hat x_j}{d}\right)
\end{aligned}
$$

Chain rule, summing over all $i$ (every output depends on $x_j$):

$$
\begin{aligned}
dx_j &= \sum_i d\hat x_i \, \frac{\partial \hat x_i}{\partial x_j} \\
&= \frac{1}{\sigma}\left(\sum_i d\hat x_i\,\delta_{ij} - \frac{1}{d}\sum_i d\hat x_i - \frac{\hat x_j}{d}\sum_i d\hat x_i\,\hat x_i\right) \\
&= \frac{1}{\sigma}\left(d\hat x_j - \overline{d\hat x} - \hat x_j\,\overline{d\hat x \odot \hat x}\right)
\end{aligned}
$$

That is the compact form, now derived:

$$
dx = \frac{1}{\sigma}\left(d\hat x - \overline{d\hat x} - \hat x\;\overline{d\hat x \odot \hat x}\right)
$$

Reading it: take the incoming gradient, remove its mean (shifting all $x_j$ equally changes nothing after normalization) and remove its component along $\hat x$ (scaling $x$ changes nothing either), then divide by $\sigma$.

**Numeric check** with $x = [1, 2, 6]$ and $d\hat x = [1, 0, 0]$: $\overline{d\hat x} = \frac{1}{3}$, $\overline{d\hat x \odot \hat x} = \frac{-0.926}{3} = -0.309$:

$$
dx = \frac{1}{2.160}\Big([1, 0, 0] - 0.333 + 0.309 \cdot [-0.926, -0.463, 1.389]\Big) = [0.176, -0.220, 0.044]
$$

Finite differences give $[0.1763, -0.2204, 0.0441]$. The entries sum to 0, as the "remove the mean" reading predicts.

### In code

All the means are over the last axis with `keepdims=True`, so one formula handles `(B, T, d)` at once. Save $\hat x$ and $\frac{1}{\sigma}$ in the forward pass; the backward needs nothing else (plus $\gamma$). Use `unbroadcast` for $d\gamma$ and $d\beta$.
:::

::: think
Q: Why is $\sum_k (x_k - \mu) = 0$, and where did that save work in the derivation?
A: $\sum_k x_k - d\mu = d\mu - d\mu = 0$. It killed the second term of $\frac{\partial v}{\partial x_j}$: moving $x_j$ also moves $\mu$, but the effect of that on the variance cancels out.
:::

::: think
Q: Explain without algebra why each row of $dx$ sums to zero.
A: Adding the same constant to every $x_j$ shifts $\mu$ by that constant and leaves $\hat x$, and hence the loss, unchanged. The sum of $dx$ is the loss's rate of change along that direction, so it is 0.
:::

::: think
Q: LayerNorm forces mean 0 and variance 1. Doesn't that limit what the model can represent?
A: $\gamma$ and $\beta$ undo it if useful: $\gamma_i = \sigma$, $\beta_i = \mu$ would recover the input (approximately, since they are per-feature, not per-position). The normalization fixes the scale *before* the learnable rescaling, so training is stable while expressiveness is kept.
:::

::: hint level=1
Forward: two means over the last axis, then normalize, scale and shift. Backward: three gradients; $\gamma$ and $\beta$ are one-liners plus `unbroadcast`; $x$ uses the boxed formula.
:::

::: hint level=2
`x.data.mean(axis=-1, keepdims=True)` and `.var(axis=-1, keepdims=True)` (NumPy's `var` is the biased one, dividing by $d$). Keep `rstd = 1 / np.sqrt(var + eps)` and `xhat` for the closure. In the backward, `dxhat = g * gamma.data`, then means of `dxhat` and `dxhat * xhat` over the last axis with `keepdims=True`.
:::

::: hint level=3
```pseudo
layernorm(x, gamma, beta, eps):
    mu, var ← mean and variance of x over the last axis (kept)
    rstd ← 1 / sqrt(var + eps);  xhat ← (x − mu) · rstd
    out ← Tensor(xhat · gamma + beta, parents (x, gamma, beta))
    backward:
        gamma.grad += unbroadcast(g · xhat, gamma shape)
        beta.grad  += unbroadcast(g, beta shape)
        dxhat ← g · gamma
        x.grad += rstd · (dxhat − mean(dxhat) − xhat · mean(dxhat · xhat))   # means over last axis
```
:::

::: attempt
File: `workspace/stage2/layers.py` — implement `layernorm` and `LayerNorm`.
Check: `python -m course check s2-l02`
:::

::: solution
@include solutions/stage2/layers.py#layernorm

- `rstd` and `xhat` are computed once in the forward pass and captured by the closure: the backward needs no recomputation.
- The `x.grad` line is the boxed formula, applied to every position at once through `keepdims=True` means.
- `unbroadcast` sums $d\gamma$ and $d\beta$ over all positions, since $\gamma$ and $\beta$ are shared.

@include solutions/stage2/layers.py#LayerNorm
:::

::: breakit
Replace the backward of $x$ with the naive `x.grad += g * gamma.data * rstd` (treating $\mu$ and $\sigma$ as constants) and run the check. Expect the gradcheck to fail with relative errors around 0.1 to 1: you dropped the two "remove the mean / remove the $\hat x$ component" terms.
:::

::: recap
- LayerNorm: normalize each position to mean 0, variance 1, then scale by $\gamma$ and shift by $\beta$.
- You derived $\frac{\partial \hat x_i}{\partial x_j} = \frac{1}{\sigma}(\delta_{ij} - \frac{1}{d} - \frac{\hat x_i \hat x_j}{d})$ and from it $dx = \frac{1}{\sigma}(d\hat x - \overline{d\hat x} - \hat x\,\overline{d\hat x \odot \hat x})$.
- Checked on $x = [1, 2, 6]$ by hand and by finite differences.
:::
