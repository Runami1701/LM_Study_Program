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
You will make the trained model write text. It draws the next character from $\mathrm{softmax}(z / \tau)$, adds it to the text, slides the $k$-character window forward by one, and repeats.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $z$, $z_i$ | the scores for the next character, shape $(V,)$ |
| $p_i$ | the probability of character $i$ |
| $\tau$ | temperature, $\tau > 0$ |
| $V$ | vocabulary size |
| $c_i$ | running total of the probabilities, $c_i = p_0 + p_1 + \dots + p_i$ |
| $u$ | a random number drawn evenly from $[0, 1)$ |
| $P$ | the probability of an event |
| $k$ | context length |
| $i, j$ | character indices |
| $M$ | the index of the largest score, $z_M = \max_j z_j$ |
| $e$ | Euler's number, as in $e^{z_i/\tau}$ |
:::

::: numpy
### Draw from given probabilities

`rng.choice(V, p=p)` returns one whole number from `0 .. V-1`, picking `i` with probability `p[i]`. `p` must add up to 1:

```python
rng = np.random.default_rng(0)
print([int(rng.choice(3, p=[0.7, 0.2, 0.1])) for _ in range(8)])   # [0, 0, 0, 0, 1, 2, 0, 1]
```

### Running totals

```python
print(np.cumsum([0.659, 0.242, 0.099]))   # [0.659 0.901 1.   ]
```

### Sliding a window with Python lists

```python
ctx = [1, 0, 2]
print(ctx[1:] + [3])                  # [0, 2, 3]   drop the oldest id, add the new one
print(([0] * 3 + [5, 6])[-3:])        # [0, 5, 6]   pad on the left, keep the last 3
print(np.array([ctx]).shape)          # (1, 3)      a batch with one example
```
:::

::: concept
### Drawing from a list of probabilities

The model gives probabilities $p_0, \dots, p_{V-1}$. To draw a character with exactly these probabilities, lay the probabilities end to end on the interval $[0, 1)$:

$$
c_i = \sum_{j=0}^{i} p_j, \qquad 0 \le c_0 \le c_1 \le \dots \le c_{V-1} = 1
$$

Draw $u$ evenly from $[0, 1)$ and return the first $i$ with $u < c_i$. Character $i$ is chosen exactly when $c_{i-1} \le u < c_i$ (with $c_{-1} = 0$). That stretch has length $c_i - c_{i-1} = p_i$, so

$$
P(\text{choose } i) = c_i - c_{i-1} = p_i
$$

`rng.choice(V, p=p)` does this for you. Example: $p = [0.659, 0.242, 0.099]$ gives $c = [0.659, 0.901, 1.0]$. The draw $u = 0.70$ falls between 0.659 and 0.901, so character 1 is chosen.

### Temperature

Divide the scores by $\tau$ before the softmax:

$$
p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}
$$

With $z = [2.0, 1.0, 0.1]$:

| $\tau$ | $z / \tau$ | $p$ |
| --- | --- | --- |
| 0.5 | $[4, 2, 0.2]$ | $[0.864, 0.117, 0.019]$ |
| 1 | $[2, 1, 0.1]$ | $[0.659, 0.242, 0.099]$ |
| 2 | $[1, 0.5, 0.05]$ | $[0.502, 0.304, 0.194]$ |

**$\tau \to 0$.** Let $M$ be the position of the largest score. Divide the top and bottom by $e^{z_M / \tau}$:

$$
p_i = \frac{e^{(z_i - z_M)/\tau}}{\sum_j e^{(z_j - z_M)/\tau}}
$$

For $j \ne M$, $z_j - z_M < 0$, so $\frac{z_j - z_M}{\tau} \to -\infty$ and $e^{(z_j - z_M)/\tau} \to 0$. The $j = M$ term is $e^0 = 1$. So $p_M \to 1$ and every other $p_i \to 0$: the model always picks its top choice ("greedy" decoding).

**$\tau \to \infty$.** Every $z_i / \tau \to 0$ and every $e^{z_i/\tau} \to 1$, so $p_i \to \frac{1}{V}$: pure noise.

Low temperatures give safe, repetitive text; high temperatures give varied and increasingly garbled text. A tiny $\tau$ makes $z / \tau$ huge, which is the overflow case of s1-l04. So subtract the maximum first, and compute in float64, so that $\tau = 10^{-3}$ causes no trouble.

### Writing text

The model always sees exactly $k$ previous characters. Start from a prompt and take its last $k$ ids. If the prompt is shorter than $k$, pad it on the left with the id of `"\n"`: the start of a line is the most natural "beginning" in this text. Then repeat $n$ times:

1. compute the scores with `predict(params, context)` for the `(1, k)` context, giving `(V,)`;
2. draw the next id;
3. add its character to the output, drop the oldest id from the context, and append the new one.

Each character costs one forward pass of one example. That is slow per character, but fine for a few hundred characters.
:::

::: think
Q: Why pad a short prompt with the newline id, and not with id 0 or a special symbol?
A: The model has never seen a special symbol. Here id 0 *is* the newline anyway, because `'\n'` sorts first. In general, padding with something the model saw in that place during training, such as text after a line break, keeps the input familiar.
:::

::: think
Q: With scores $[2, 1, 0.1]$, how likely is the top choice at $\tau = 0.5$, 1 and 2? Which temperature writes text most like Shakespeare's?
A: 0.864, 0.659 and 0.502. $\tau = 1$ draws from exactly the distribution the model learned, so its statistics match the training text best. Slightly lower values (0.7 to 0.9) often read better to people, but repeat themselves more.
:::

::: think
Q: Why compute `z / τ` and the softmax in float64 inside `sample_next`?
A: At $\tau = 10^{-3}$, scores of about 10 become about $10^4$. After subtracting the maximum, the exponentials are fine in either type. But `rng.choice` checks that `p` adds up to 1 very precisely, and float32 rounding over $V$ entries can fail that check. float64 costs almost nothing for one `(V,)` vector.
:::

::: hint level=1
`sample_next` is a safe softmax with temperature, followed by one random draw. `sample` keeps a Python list of the last $k$ ids and loops $n$ times.
:::

::: hint level=2
`logits.astype(np.float64) / temperature`, subtract the maximum, `np.exp`, and divide by the sum. Then `rng.choice(V, p=p)` and `int(...)`. For the context, `ctx[1:] + [new_id]` slides the window. `predict` needs a `(1, k)` array: `np.array([ctx])`. Get $k$ from the parameter shapes: `W1` has $k \cdot e$ rows and `C` has $e$ columns.
:::

::: hint level=3
```pseudo
sample_next(logits, τ, rng):
    z ← logits as float64 / τ;  z ← z − max(z)
    p ← exp(z) / sum(exp(z))
    return int(rng.choice(V, p=p))

sample(params, itos, stoi, prompt, n, τ, rng):
    k   ← rows of W1 / columns of C
    ctx ← the last k of ([newline id] · k  +  ids of prompt)
    out ← []
    repeat n times:
        logits ← predict(params, [ctx])[0]
        i ← sample_next(logits, τ, rng)
        add itos[i] to out;  ctx ← ctx without its first id, plus i
    return out joined into one string
```
:::

::: attempt
File: `workspace/stage1/sample.py` — fill in `sample_next` and `sample`.
Check: `python -m course check s1-l13`
After your s1-l12 training run, try it: `python -m workspace.stage1.sample 0.8` prints 500 characters at temperature 0.8 from `runs/s1/params.npz`.
:::

::: solution
@include solutions/stage1/sample.py#sample_next

- float64 and subtracting the maximum make tiny temperatures safe. Dividing by the sum makes `p` add up to 1 as precisely as `rng.choice` requires.
- `rng.choice(len(p), p=p)` is the running-total draw derived above.

@include solutions/stage1/sample.py#sample

- `k` comes from the shapes, so `sample` works for any trained configuration.
- `([pad] * k + ids)[-k:]` handles prompts shorter and longer than $k$ in one expression.
- The context is a list of Python ints; `np.array([context])` makes the `(1, k)` batch that `predict` expects.
:::

::: breakit
Run `python -m workspace.stage1.sample 0.3`, then `1.0`, then `2.0` (the same seed each time). At 0.3 the same common words and names come back in loops. At 1.0 you get lines shaped like Shakespeare, with plausible names, colons and line breaks, but many invented words. At 2.0 you get almost random letters and punctuation.
:::

::: recap
- The running-total method draws character $i$ with probability exactly $p_i$; you derived why.
- Temperature reshapes the distribution: $\tau \to 0$ always picks the top choice, $\tau \to \infty$ gives uniform noise.
- Writing text loops: predict → draw → slide the $k$-character window.
:::
