---
id: s1-l13
title: Sampling
stage: s1
order: 13
prereqs: [s1-l12]
files: [workspace/stage1/sample.py]
functions: [sample_next, sample]
tests: tests/s1/test_l13_sample.py
est_minutes: 40
---

::: goal
Make the trained model write: draw the next character from $\mathrm{softmax}(z / \tau)$, append it, slide the $k$-character window, and repeat.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $z$, $z_i$ | logits for the next character, shape $(V,)$ |
| $p_i$ | the probability of character $i$ |
| $\tau$ | temperature, $\tau > 0$ |
| $V$ | vocabulary size |
| $c_i$ | cumulative probability $c_i = p_0 + p_1 + \dots + p_i$ |
| $u$ | a uniform random number in $[0, 1)$ |
| $P$ | probability of an event |
| $k$ | context length |
| $i, j$ | class indices |
| $M$ | the index of the largest logit, $z_M = \max_j z_j$ |
| $e$ | Euler's number, as in $e^{z_i/\tau}$ |
:::

::: concept
### Drawing from a categorical distribution

The model gives probabilities $p_0, \dots, p_{V-1}$. To draw a character with exactly those probabilities, lay the probabilities end to end on $[0, 1)$:

$$
c_i = \sum_{j=0}^{i} p_j, \qquad 0 \le c_0 \le c_1 \le \dots \le c_{V-1} = 1
$$

Draw $u$ uniformly from $[0, 1)$ and return the first $i$ with $u < c_i$. Character $i$ is chosen exactly when $c_{i-1} \le u < c_i$ (with $c_{-1} = 0$), an interval of length $c_i - c_{i-1} = p_i$, so

$$
P(\text{choose } i) = c_i - c_{i-1} = p_i
$$

This "inverse CDF" method is what `rng.choice(V, p=p)` does for you. Example: $p = [0.659, 0.242, 0.099]$ gives $c = [0.659, 0.901, 1.0]$; $u = 0.70$ falls in $[0.659, 0.901)$, so character 1 is chosen.

### Temperature

Divide the logits by $\tau$ before the softmax:

$$
p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}
$$

With $z = [2.0, 1.0, 0.1]$:

| $\tau$ | $z / \tau$ | $p$ |
| --- | --- | --- |
| 0.5 | $[4, 2, 0.2]$ | $[0.864, 0.117, 0.019]$ |
| 1 | $[2, 1, 0.1]$ | $[0.659, 0.242, 0.099]$ |
| 2 | $[1, 0.5, 0.05]$ | $[0.502, 0.304, 0.194]$ |

**$\tau \to 0$.** Let $M$ be the largest logit. Divide top and bottom by $e^{z_M / \tau}$:

$$
p_i = \frac{e^{(z_i - z_M)/\tau}}{\sum_j e^{(z_j - z_M)/\tau}}
$$

For $j \ne M$, $z_j - z_M < 0$, so $\frac{z_j - z_M}{\tau} \to -\infty$ and $e^{(z_j - z_M)/\tau} \to 0$; the $j = M$ term is $e^0 = 1$. Hence $p_M \to 1$ and every other $p_i \to 0$: always the argmax ("greedy" decoding).

**$\tau \to \infty$.** Every $z_i / \tau \to 0$, every $e^{z_i/\tau} \to 1$, so $p_i \to \frac{1}{V}$: uniform noise.

Low temperature gives safe, repetitive text; high temperature gives varied, increasingly garbled text. Numerically, a tiny $\tau$ makes $z / \tau$ huge, which is exactly the overflow case of s1-l04: subtract the max first, and do it in float64 so $\tau = 10^{-3}$ is no problem.

### Generating text

The model sees exactly $k$ previous characters. Start from a prompt: take its last $k$ ids. If the prompt is shorter than $k$, pad on the left with the id of `"\n"`; the start of a line is the most natural "beginning" in this text. Then repeat $n$ times:

1. logits = `predict(params, context)` for the `(1, k)` context, giving `(V,)`;
2. draw the next id;
3. append its character to the output, drop the oldest id from the context and append the new one.

Each character costs one forward pass of a single example. That is slow per character but fine for a few hundred.
:::

::: think
Q: Why pad a short prompt with the newline id, rather than with id 0 or a special token?
A: The model has never seen a special token, and id 0 here *is* the newline anyway (`'\n'` sorts first). In general, padding with something the model saw in that position during training (text after a line break) keeps the input in-distribution.
:::

::: think
Q: With logits $[2, 1, 0.1]$, what is the probability of picking the argmax at $\tau = 0.5$, 1 and 2? Which temperature would you use to write the most "Shakespeare-like" text?
A: 0.864, 0.659, 0.502. $\tau = 1$ samples from exactly the distribution the model learned, so its statistics match the training text best. Slightly lower values (0.7 to 0.9) usually read better to humans but repeat themselves more.
:::

::: think
Q: Why compute `z / τ` and the softmax in float64 inside `sample_next`?
A: At $\tau = 10^{-3}$, logits of size ~10 become ~$10^4$. After max-subtraction the exponentials are fine in either type, but `rng.choice` checks that `p` sums to 1 within a tight tolerance, and float32 rounding over $V$ entries can fail that check. float64 is cheap for one `(V,)` vector.
:::

::: hint level=1
`sample_next` is a temperature-scaled, stable softmax followed by one random draw. `sample` keeps a Python list of the last $k$ ids and loops $n$ times.
:::

::: hint level=2
`logits.astype(np.float64) / temperature`, subtract the max, `np.exp`, normalize; then `rng.choice(V, p=p)` and `int(...)`. For the context, a list works: `ctx[1:] + [new_id]` slides it. `predict` wants a `(1, k)` int array: `np.array([ctx])`. Read $k$ from the parameter shapes: `W1` has $k \cdot e$ rows and `C` has $e$ columns.
:::

::: hint level=3
```pseudo
sample_next(logits, τ, rng):
    z ← logits as float64 / τ;  z ← z − max(z)
    p ← exp(z) / sum(exp(z))
    return int(rng.choice(V, p=p))

sample(params, itos, stoi, prompt, n, τ, rng):
    k   ← rows of W1 / columns of C
    ctx ← last k of ([newline id] · k  +  ids of prompt)
    out ← []
    repeat n times:
        logits ← predict(params, [ctx])[0]
        i ← sample_next(logits, τ, rng)
        append itos[i] to out;  ctx ← ctx without its first id, plus i
    return out joined into a string
```
:::

::: attempt
File: `workspace/stage1/sample.py` — implement `sample_next` and `sample`.
Check: `python -m course check s1-l13`
After your s1-l12 training run, try it: `python -m workspace.stage1.sample 0.8` prints 500 characters at temperature 0.8 from `runs/s1/params.npz`.
:::

::: solution
@include solutions/stage1/sample.py#sample_next

- float64 and max-subtraction make tiny temperatures safe; renormalizing makes `p` sum to 1 to within float64 rounding, as `rng.choice` requires.
- `rng.choice(len(p), p=p)` is the inverse-CDF draw derived above.

@include solutions/stage1/sample.py#sample

- `k` is recovered from shapes, so `sample` works for any trained configuration.
- `([pad] * k + ids)[-k:]` handles prompts both shorter and longer than $k$ in one expression.
- The context is a list of Python ints; `np.array([context])` makes the `(1, k)` batch that `predict` expects.
:::

::: breakit
Run `python -m workspace.stage1.sample 0.3`, then `1.0`, then `2.0` (same seed each time). Expect: at 0.3, the same common words and names in loops ("the the", "I will"); at 1.0, Shakespeare-shaped lines with plausible names, colons and line breaks, but mostly invented words; at 2.0, near-random letters and punctuation.
:::

::: recap
- The inverse-CDF method draws category $i$ with probability exactly $p_i$; you derived why.
- Temperature reshapes the distribution: $\tau \to 0$ gives the argmax, $\tau \to \infty$ gives uniform.
- Generation loops predict → draw → slide the $k$-character window.
:::
