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
You will build the tool that tells you whether a backward pass is correct. It estimates each gradient numerically from two loss values, compares the estimate with your backward pass on random entries, and reports the worst relative error for each parameter.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $f$ | the loss as a function of one parameter entry $\theta$ (all others fixed) |
| $\theta$ | the parameter entry being checked |
| $\epsilon$ | a small step, e.g. $10^{-5}$ |
| $f'$, $f''$, $f'''$ | first, second and third derivatives of $f$ at $\theta$ |
| $O(\epsilon^2)$ | "at most a constant times $\epsilon^2$" when $\epsilon$ is small |
| $a$ | the analytic gradient (from your backward pass) |
| $n$ | the numerical gradient (from the two loss values) |
| $\delta$ | a tiny floor, $10^{-8}$, that avoids dividing by zero |
| $u$ | machine epsilon of the number type (s0-l00) |
:::

::: numpy
### Pick distinct random positions

`rng.choice(N, size=k, replace=False)` draws `k` different whole numbers from `0 .. N-1`:

```python
rng = np.random.default_rng(0)
print(rng.choice(10, size=3, replace=False))   # [5 9 6]
```

### Turn a flat position into a row and column

`np.unravel_index(position, shape)` converts a position in the flattened array into an index for each axis. It works for any number of axes:

```python
print(np.unravel_index(7, (3, 4)))   # (np.int64(1), np.int64(3))   row 1, column 3
W = np.zeros((3, 4))
i = np.unravel_index(7, (3, 4))
W[i] = 5.0                           # the tuple works as an index
print(W[1, 3])                       # 5.0
```

### Functions as arguments

`gradcheck` receives a function `f` and calls it as `f()`. The test builds `f` so that it reads the current contents of `params`. When you change `params["W"][i]` in place, the next `f()` uses the new value.
:::

::: concept
### The idea

A derivative is a limit: $f'(\theta) = \lim_{\epsilon \to 0} \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$. With a small but finite $\epsilon$, two loss computations **estimate** any gradient, without calculus. This is far too slow for training (two forward passes per parameter entry). But it is the perfect test for a backward pass, which is fast and easy to get wrong.

### One-sided or centered: derive the error

Taylor's theorem expands $f$ around $\theta$:

$$
f(\theta + \epsilon) = f + \epsilon f' + \frac{\epsilon^2}{2} f'' + \frac{\epsilon^3}{6} f''' + O(\epsilon^4)
$$

$$
f(\theta - \epsilon) = f - \epsilon f' + \frac{\epsilon^2}{2} f'' - \frac{\epsilon^3}{6} f''' + O(\epsilon^4)
$$

The **one-sided** estimate follows from the first line:

$$
\frac{f(\theta + \epsilon) - f(\theta)}{\epsilon} = f' + \frac{\epsilon}{2} f'' + O(\epsilon^2)
$$

Its error is proportional to $\epsilon$. For the **centered** estimate, subtract the second line from the first. The even terms ($f$ and $\frac{\epsilon^2}{2}f''$) cancel, and the odd terms double:

$$
f(\theta + \epsilon) - f(\theta - \epsilon) = 2\epsilon f' + \frac{\epsilon^3}{3} f''' + O(\epsilon^5)
$$

$$
\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon} = f' + \frac{\epsilon^2}{6} f''' + O(\epsilon^4)
$$

Its error is proportional to $\epsilon^2$. With $\epsilon = 10^{-5}$, the one-sided error is about $10^{-5}$ and the centered error about $10^{-10}$. Both cost two loss computations; the centered one gives five more correct digits.

### Why float64

Each loss value carries a rounding error of about $u \cdot |f|$, where $u$ is machine epsilon. Dividing the difference of two such values by $2\epsilon$ blows this error up to about $\frac{u |f|}{\epsilon}$:

- float64: $\frac{2.2 \times 10^{-16}}{10^{-5}} \approx 2 \times 10^{-11}$, which is negligible;
- float32: $\frac{1.2 \times 10^{-7}}{10^{-5}} \approx 10^{-2}$, a 1% error from rounding alone.

So gradient checks always run on float64 copies of the parameters.

### Comparing two numbers: relative error

Are $a = 0.0012$ and $n = 0.0011$ close? Are $a = 1200$ and $n = 1201$? The plain difference can't answer both questions. Use the relative error:

$$
\text{rel} = \frac{|a - n|}{\max(|a|, |n|, \delta)}
$$

$0.0012$ against $0.0011$ gives $\frac{0.0001}{0.0012} = 0.083$ (bad). $1200$ against $1201$ gives $\frac{1}{1201} = 8 \times 10^{-4}$ (suspicious). The floor $\delta = 10^{-8}$ handles $a = n = 0$, which would otherwise be $\frac{0}{0}$.

Rules of thumb in float64:

| relative error | verdict |
| --- | --- |
| below $10^{-7}$ | correct |
| $10^{-7}$ to $10^{-5}$ | probably fine (tiny gradients can cause this) |
| above $10^{-3}$ | a bug |

### Spot checks

A real model has hundreds of thousands of entries, and checking all of them would take hundreds of thousands of forward passes. A wrong backward formula almost always affects many entries, so about 20 random entries per parameter array are enough to catch it. Arrays with fewer than 20 entries are checked completely.

### How the function is used

`f` takes no arguments and recomputes the loss from the current contents of `params`. You change one entry **in place**, call `f()`, change it again, call `f()`, and then **put the original value back**. If you forget to restore it, the model changes for every later check.
:::

::: think
Q: The centered and the one-sided estimate both need two loss values. Why is the centered one so much more accurate?
A: In the centered difference, the $\frac{\epsilon^2}{2}f''$ terms of $f(\theta+\epsilon)$ and $f(\theta-\epsilon)$ cancel. The leading error becomes $\frac{\epsilon^2}{6}f'''$ instead of $\frac{\epsilon}{2}f''$: one extra power of $\epsilon$, about $10^{-10}$ instead of $10^{-5}$.
:::

::: think
Q: Why not make $\epsilon$ as small as possible, say $10^{-12}$?
A: The Taylor error shrinks like $\epsilon^2$, but the rounding error grows like $\frac{u}{\epsilon}$. At $\epsilon = 10^{-12}$ in float64 the rounding error is about $\frac{2 \times 10^{-16}}{10^{-12}} = 2 \times 10^{-4}$, which is far worse. $\epsilon \approx 10^{-5}$ balances the two errors.
:::

::: think
Q: Your check reports $3 \times 10^{-2}$ for `W1` and $10^{-9}$ for every other parameter. Where do you look?
A: At the parts of the backward pass that produce only `dW1`: the line `dW1 = flat.T @ da` and its inputs. `b1` passes, so `da` is probably right; that points at `flat` or the transpose.
:::

::: hint level=1
For each parameter array, choose up to `n_checks` random entries. For each entry: nudge it up and compute the loss, nudge it down and compute the loss, restore it, and compare the centered estimate with the analytic value.
:::

::: hint level=2
`rng.choice(p.size, size=n, replace=False)` picks distinct flat positions, and `np.unravel_index(flat, p.shape)` turns each into an index (see the box above). Save the old value, write `old + eps`, call `f()`, write `old - eps`, call `f()`, write `old` back. Keep the largest relative error.
:::

::: hint level=3
```pseudo
gradcheck(f, params, grads, eps, n_checks, rng):
    result ← empty dict
    for each name, p in params (skip names that are not in grads):
        worst ← 0
        for each of min(n_checks, size of p) different random positions idx:
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
File: `workspace/stage1/gradcheck.py` — fill in `gradcheck`.
Check: `python -m course check s1-l08`
:::

::: solution
@include solutions/stage1/gradcheck.py#gradcheck

- `n = min(n_checks, p.size)` checks every entry of small arrays.
- `np.unravel_index` makes the code work for 1-D biases and 2-D weights alike.
- The value is restored right after the two loss computations, so later checks, and the caller, see the original parameters.
- `max(abs(numeric), abs(analytic), 1e-8)` is the denominator of the relative error, with the floor $\delta$.
:::

::: breakit
Run your `gradcheck` on the s1-l09 model with the parameters left in **float32** (`eps = 1e-5`). A perfectly correct backward pass shows "errors" of about $10^{-2}$ to $10^{-1}$: pure rounding noise, as derived above. Convert the parameters to float64 and the errors drop below $10^{-5}$.
:::

::: recap
- Centered differences have $O(\epsilon^2)$ error; one-sided differences only $O(\epsilon)$. You derived both from Taylor expansions.
- Check in float64: float32 rounding alone causes about 1% error at $\epsilon = 10^{-5}$.
- Relative error is $\frac{|a - n|}{\max(|a|, |n|, \delta)}$; below $10^{-7}$ is correct, above $10^{-3}$ is a bug.
- Check about 20 random entries per array, and always restore what you change.
:::
