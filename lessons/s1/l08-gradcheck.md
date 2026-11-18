---
id: s1-l08
title: Gradient checking
stage: s1
order: 8
prereqs: [s1-l07]
files: [workspace/stage1/gradcheck.py]
functions: [gradcheck]
tests: tests/s1/test_l08_gradcheck.py
est_minutes: 40
---

::: goal
Build the tool that tells you whether a backward pass is right: compare each analytic gradient with a numerical estimate from two loss evaluations, on random entries, and report the worst relative error per parameter.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $f$ | the loss as a function of one parameter entry $\theta$ (all others held fixed) |
| $\theta$ | the parameter entry being checked |
| $\epsilon$ | a small step, e.g. $10^{-5}$ |
| $f'$, $f''$, $f'''$ | first, second, third derivatives of $f$ at $\theta$ |
| $O(\epsilon^2)$ | "a term no bigger than a constant times $\epsilon^2$" for small $\epsilon$ |
| $a$ | the analytic gradient (from your backward pass) |
| $n$ | the numerical gradient (from finite differences) |
| $\delta$ | a tiny floor, $10^{-8}$, that avoids dividing by zero |
| $u$ | machine epsilon of the float type (s0-l00) |
:::

::: concept
### The idea

A derivative is a limit: $f'(\theta) = \lim_{\epsilon \to 0} \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$. With a small but finite $\epsilon$ we can **estimate** any gradient by running the forward pass twice, no calculus needed. It is far too slow to train with (two forward passes per parameter entry), but perfect for testing a backward pass that is fast but easy to get wrong.

### One-sided vs centered differences: derive the error

Taylor's theorem expands $f$ around $\theta$:

$$
f(\theta + \epsilon) = f + \epsilon f' + \frac{\epsilon^2}{2} f'' + \frac{\epsilon^3}{6} f''' + O(\epsilon^4)
$$

$$
f(\theta - \epsilon) = f - \epsilon f' + \frac{\epsilon^2}{2} f'' - \frac{\epsilon^3}{6} f''' + O(\epsilon^4)
$$

**One-sided** difference, from the first line:

$$
\frac{f(\theta + \epsilon) - f(\theta)}{\epsilon} = f' + \frac{\epsilon}{2} f'' + O(\epsilon^2)
$$

The error is proportional to $\epsilon$. **Centered** difference: subtract the second line from the first. The even terms ($f$ and $\frac{\epsilon^2}{2}f''$) cancel and the odd ones double:

$$
f(\theta + \epsilon) - f(\theta - \epsilon) = 2\epsilon f' + \frac{\epsilon^3}{3} f''' + O(\epsilon^5)
$$

$$
\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon} = f' + \frac{\epsilon^2}{6} f''' + O(\epsilon^4)
$$

The error is now proportional to $\epsilon^2$. With $\epsilon = 10^{-5}$: about $10^{-5}$ for one-sided, about $10^{-10}$ for centered. Same cost, two loss evaluations, five more correct digits.

### Why float64

Each loss value carries rounding error of about $u \cdot |f|$, where $u$ is machine epsilon. Dividing a difference of two such values by $2\epsilon$ amplifies it to roughly $\frac{u |f|}{\epsilon}$:

- float64: $\frac{2.2 \times 10^{-16}}{10^{-5}} \approx 2 \times 10^{-11}$, negligible;
- float32: $\frac{1.2 \times 10^{-7}}{10^{-5}} \approx 10^{-2}$, a 1% error from rounding alone.

So gradient checks always run on float64 copies of the parameters. That is the one place this course uses float64.

### Comparing two numbers: relative error

Is $a = 0.0012$ vs $n = 0.0011$ close? Is $a = 1200$ vs $n = 1201$? Absolute differences can't answer both. Use the relative error

$$
\text{rel} = \frac{|a - n|}{\max(|a|, |n|, \delta)}
$$

$0.0012$ vs $0.0011$ gives $\frac{0.0001}{0.0012} = 0.083$ (bad); $1200$ vs $1201$ gives $\frac{1}{1201} = 8 \times 10^{-4}$ (suspicious). The floor $\delta = 10^{-8}$ handles $a = n = 0$, which would otherwise be $\frac{0}{0}$.

Rules of thumb for float64:

| relative error | verdict |
| --- | --- |
| $< 10^{-7}$ | correct |
| $10^{-7}$ to $10^{-5}$ | probably fine (kinks like ReLU, or tiny gradients, can do this) |
| $> 10^{-3}$ | a bug |

### Spot checks

A real model has hundreds of thousands of entries: checking all means hundreds of thousands of forward passes. A bug in a backward formula almost always affects many entries, so checking ~20 random entries per parameter array catches it. For arrays with fewer than 20 entries, check them all.

### The interface

`f` is a function of no arguments that recomputes the loss from the current contents of `params`. You change one entry **in place**, call `f()`, change it again, call `f()`, then **put the original value back**: forgetting to restore it quietly changes the model for every later check.
:::

::: think
Q: With the same two function evaluations, why is the centered difference so much more accurate than $\frac{f(\theta+\epsilon) - f(\theta)}{\epsilon}$?
A: In the centered difference the $\frac{\epsilon^2}{2}f''$ terms of $f(\theta+\epsilon)$ and $f(\theta-\epsilon)$ cancel, so the leading error is $\frac{\epsilon^2}{6}f'''$ instead of $\frac{\epsilon}{2}f''$: one extra power of $\epsilon$, i.e. about $10^{-10}$ instead of $10^{-5}$ for $\epsilon = 10^{-5}$.
:::

::: think
Q: Why not make $\epsilon$ as small as possible, say $10^{-12}$?
A: The truncation error shrinks like $\epsilon^2$, but the rounding error grows like $\frac{u}{\epsilon}$. At $\epsilon = 10^{-12}$ in float64 the rounding term is about $\frac{2 \times 10^{-16}}{10^{-12}} = 2 \times 10^{-4}$: far worse. $\epsilon \approx 10^{-5}$ balances the two.
:::

::: think
Q: Your check reports $3 \times 10^{-2}$ for `W1` but $10^{-9}$ for every other parameter. Where do you look?
A: At the parts of the backward pass that produce `dW1` only: `dW1 = flat.T @ da` and the inputs it uses (the cached `flat`, and `da` from the tanh backward). Since `b1` passes, `da` is probably right, which points at `flat` or the transpose.
:::

::: hint level=1
For each parameter array, choose up to `n_checks` random entries. For each entry: nudge up, evaluate, nudge down, evaluate, restore, and compare the centered difference with the analytic value.
:::

::: hint level=2
`rng.choice(p.size, size=n, replace=False)` picks distinct flat positions; `np.unravel_index(flat, p.shape)` turns one into a multi-dimensional index. Save the old value, write `old + eps`, call `f()`, write `old - eps`, call `f()`, write `old` back. Keep the running maximum of the relative error.
:::

::: hint level=3
```pseudo
gradcheck(f, params, grads, eps, n_checks, rng):
    result ← empty dict
    for each name, p in params (skip names not in grads):
        worst ← 0
        for each of min(n_checks, size of p) distinct random positions idx:
            old ← p[idx]
            p[idx] ← old + eps;  f_plus  ← f()
            p[idx] ← old − eps;  f_minus ← f()
            p[idx] ← old
            n ← (f_plus − f_minus) / (2 eps);  a ← grads[name][idx]
            worst ← max(worst, |a − n| / max(|a|, |n|, 1e-8))
        result[name] ← worst
    return result
```
:::

::: attempt
File: `workspace/stage1/gradcheck.py` — implement `gradcheck`.
Check: `python -m course check s1-l08`
:::

::: solution
@include solutions/stage1/gradcheck.py#gradcheck

- `rng.choice(p.size, size=n, replace=False)` samples distinct flat positions; `n = min(n_checks, p.size)` checks every entry of small arrays.
- `np.unravel_index` converts a flat position into an index tuple, so the code works for 1-D biases and 2-D weights alike.
- The value is restored right after the two evaluations, so later checks (and the caller) see the original parameters.
- `max(abs(numeric), abs(analytic), 1e-8)` is the relative-error denominator with the floor $\delta$.
:::

::: breakit
Run your `gradcheck` on the s1-l09 model with the parameters left in **float32** (and `eps = 1e-5`). Expect "errors" around $10^{-2}$ to $10^{-1}$ for a perfectly correct backward pass: pure rounding noise, as derived above. Then cast the parameters to float64 and watch the errors drop below $10^{-5}$.
:::

::: recap
- Centered differences have $O(\epsilon^2)$ error; one-sided only $O(\epsilon)$. You derived both from Taylor expansions.
- Check in float64: float32 rounding alone produces ~1% errors at $\epsilon = 10^{-5}$.
- Relative error $\frac{|a - n|}{\max(|a|, |n|, \delta)}$; below $10^{-7}$ is correct, above $10^{-3}$ is a bug.
- Spot-check ~20 random entries per array, and always restore what you perturb.
:::
