---
id: s1-l09
title: Assemble the MLP, initialize, overfit one batch
stage: s1
order: 9
prereqs: [s1-l08]
files: [workspace/stage1/model.py]
functions: [init_params, predict, forward, backward, count_params]
tests: tests/s1/test_l09_model.py
est_minutes: 60
---

::: goal
You will connect your layers into the full model: `x` `(B, k)` → look up → flatten → linear → tanh → linear → `logits` `(B, V)` → loss, plus its backward pass. You will choose starting values so that the first loss is about $\ln V$, and prove the gradients correct by letting the model memorize one batch.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C$ | embedding table $(V, e)$ |
| $W_1, b_1$ | first layer: $(k e, h)$ and $(h,)$ |
| $W_2, b_2$ | output layer: $(h, V)$ and $(V,)$ |
| $V, k, e, h$ | vocabulary size, context length, embedding size, hidden units |
| $B$ | batch size |
| $a$ | the hidden layer before tanh, $(B, h)$; $a_j$ is one unit |
| $x_i$ | the $i$-th of the $n = ke$ inputs of the first layer |
| $n$ | fan-in: the number of inputs that feed one unit ($n = ke$ for $W_1$) |
| $\mathrm{Var}$ | variance: the average squared distance from the mean |
| $\sigma$ | standard deviation (the square root of the variance) |
| $\mathbb{E}$ | expected value (the average over the random draws) |
| $N(0, 1)$ | the standard normal distribution: mean 0, variance 1 |
| $W$ | a weight matrix in general ($W_{ij}$ its entries) |
| $j$ | a hidden unit |
| $p_i$ | predicted probability of character $i$ |
| $z$ | logits |
:::

::: numpy
### A dictionary of arrays

The model keeps its parameters in a dictionary. Two loops you will need:

```python
p = {"W": np.ones((2, 3)), "b": np.zeros(3)}
print(sum(a.size for a in p.values()))        # 9    total number of entries
q = {name: a.astype(np.float32) for name, a in p.items()}
print(q["W"].dtype)                           # float32
```

### Square root and scaling

```python
print(np.sqrt(192))                           # 13.856406460551018
```

Dividing a random matrix by `np.sqrt(192)` makes its entries about 13.9 times smaller. Do this while the numbers are still float64, then convert the whole dictionary to float32 once, as in the loop above.

### Unpacking a saved tuple

`forward` returns everything `backward` needs in one tuple. `backward` unpacks it in the same order:

```python
cache = (x, emb.shape, V, c1, c2, c3, c4)        # in forward
x, emb_shape, V, c1, c2, c3, c4 = cache          # in backward
```
:::

::: concept
### The architecture (Bengio et al., 2003, in miniature)

| step | operation | output shape |
| --- | --- | --- |
| 1 | `emb = C[x]` | `(B, k, e)` |
| 2 | flatten | `(B, k·e)` |
| 3 | `a = flat @ W1 + b1` | `(B, h)` |
| 4 | `t = tanh(a)` | `(B, h)` |
| 5 | `logits = t @ W2 + b2` | `(B, V)` |
| 6 | mean cross-entropy | one number |

The backward pass visits the same steps in reverse, each with a function you already wrote: `cross_entropy_backward`, `linear_backward` (gives `dW2`, `db2`, `dt`), `tanh_backward`, `linear_backward` (gives `dW1`, `db1`, `dflat`), reshape back to `(B, k, e)`, and `embedding_backward` (gives `dC`). So the forward pass must keep every cache.

### How many parameters?

$$
\begin{aligned}
\text{parameters} &= \underbrace{V e}_{C} + \underbrace{k e \cdot h}_{W_1} + \underbrace{h}_{b_1} + \underbrace{h V}_{W_2} + \underbrace{V}_{b_2}
\end{aligned}
$$

For $V = 65$, $k = 8$, $e = 24$, $h = 512$:

$$
1{,}560 + 98{,}304 + 512 + 33{,}280 + 65 = 133{,}721
$$

Most of them are in $W_1$: each of the $h$ hidden units looks at all $ke = 192$ inputs.

### Starting values for $W_1$: keep the hidden layer at unit size

Hidden unit $j$ computes $a_j = \sum_{i=1}^{n} x_i W_{ij}$ (the bias starts at zero), with $n = ke$ inputs. Assume the inputs and weights are independent with mean 0, $\mathrm{Var}(x_i) = 1$ (the embeddings start as $N(0,1)$), and $\mathrm{Var}(W_{ij}) = \sigma^2$. For independent factors with mean 0, $\mathrm{Var}(x_i W_{ij}) = \mathbb{E}[x_i^2]\,\mathbb{E}[W_{ij}^2] = 1 \cdot \sigma^2$. Variances of independent terms add:

$$
\begin{aligned}
\mathrm{Var}(a_j) &= \sum_{i=1}^{n} \mathrm{Var}(x_i W_{ij}) \\
&= n \sigma^2
\end{aligned}
$$

With $\sigma = 1$ and $n = 192$, $a_j$ has standard deviation $\sqrt{192} \approx 13.9$. Almost every tanh unit would be saturated at $\pm 1$ and pass almost no gradient (s1-l06). With $\sigma = \frac{1}{\sqrt{n}}$, $\mathrm{Var}(a_j) = n \cdot \frac{1}{n} = 1$. So $W_1 = \frac{N(0, 1)}{\sqrt{ke}}$: this is the **$1/\sqrt{\text{fan\_in}}$** rule.

You can check it on the model. The fraction of hidden outputs with $|t| > 0.97$ (that is, $|a| > 2.1$) is about 3–4% with the rule, and about 88% with $\sigma = 1$.

### Starting values for $W_2$: start honest

Make $W_2$ tiny ($0.01 \cdot N(0,1)$) and $b_2 = 0$. Then every score is close to 0, softmax is close to uniform, and the first loss is close to $\ln V = 4.17$ (s1-l04). A large $W_2$ makes the untrained model confidently wrong, with a first loss of 30 or more. The first few hundred training steps would only undo that.

For the check to reproduce your starting values exactly, draw them in a fixed order from the same `rng`: `C`, then `W1`, then `W2` (see the docstring).

### Memorize one batch: the best end-to-end test

Take 32 examples and train on them alone. A correct model with 130K parameters can memorize 32 examples, so the loss must fall close to 0 within a few hundred steps of plain gradient descent. If the loss gets stuck, some gradient is wrong, or the update has the wrong sign. This test catches bugs that small gradient checks can miss, and it takes about a second.
:::

::: think
Q: Why should the first loss be close to $\ln V$, and what do first losses of 4.17 and 12 tell you about $W_2$?
A: A model that knows nothing should predict uniformly, which scores $\ln V = 4.17$. A loss of 12 means the starting scores are large and confidently wrong ($p_y \approx e^{-12}$), so $W_2$ is too big. Training first has to shrink those scores, which shows up as a sharp drop at the start.
:::

::: think
Q: If $W_1$ were $N(0, 1)$ with $n = 192$, what would the standard deviation of a hidden unit be, and what would tanh do?
A: $\mathrm{Var}(a_j) = n \cdot 1 \cdot 1 = 192$, so $\sigma_a \approx 13.9$. Most $|a_j| > 3$, where tanh is flat ($1 - t^2 < 0.01$). The units are saturated, gradients through them vanish, and $W_1$ barely learns.
:::

::: think
Q: Your model memorizes one batch down to a loss of 0.02, but the gradient check reports a relative error of 0.3 for `C`. How can both be true?
A: Memorizing only shows that the updates lower the loss on that batch. The model can memorize 32 examples using $W_1$, $W_2$ and the biases even if the gradient of `C` is wrong (for example, with repeated ids dropped, or zero). You need both tests.
:::

::: hint level=1
`forward` calls your five forward functions in order and returns all their caches together. `backward` unpacks the caches and calls the backward functions in reverse order. `predict` is `forward` without the loss.
:::

::: hint level=2
Put `x`, the `(B, k, e)` embedding shape and `V` into the cache, because the embedding backward needs them. Un-flatten `dflat` with `.reshape(emb_shape)`. For the parameters: `rng.standard_normal(shape)`, `/ np.sqrt(k * e)`, `* 0.01`, `np.zeros(h)`, then `.astype(np.float32)` for each. `count_params`: the sum of `.size`.
:::

::: hint level=3
```pseudo
forward(params, x, y):
    emb ← embedding_forward(C, x)            # (B, k, e)
    flat ← flatten_context(emb)              # (B, k·e)
    a, c1 ← linear_forward(flat, W1, b1)     # (B, h)
    t, c2 ← tanh_forward(a)
    logits, c3 ← linear_forward(t, W2, b2)   # (B, V)
    loss, c4 ← cross_entropy(logits, y)
    return loss, (x, shape of emb, V, c1, c2, c3, c4)

backward(cache):
    dlogits ← cross_entropy_backward(c4)
    dt, dW2, db2 ← linear_backward(dlogits, c3)
    da ← tanh_backward(dt, c2)
    dflat, dW1, db1 ← linear_backward(da, c1)
    dC ← embedding_backward(dflat reshaped to emb's shape, x, V)
    return {C: dC, W1: dW1, b1: db1, W2: dW2, b2: db2}
```
:::

::: attempt
File: `workspace/stage1/model.py` — fill in `init_params`, `predict`, `forward`, `backward` and `count_params`.
Check: `python -m course check s1-l09`
:::

::: solution
@include solutions/stage1/model.py#init_params

- Drawing `C`, `W1`, `W2` in this order from one `rng` makes runs repeatable; the biases need no randomness.
- `/ np.sqrt(k * e)` is the $1/\sqrt{\text{fan\_in}}$ rule; `* 0.01` keeps the first scores near 0. Drawing in float64 and converting once at the end avoids stray float64 parameters.

@include solutions/stage1/model.py#predict

@include solutions/stage1/model.py#forward

- Each forward function returns its cache; `forward` bundles them with what the embedding backward needs (`x`, the embedding shape, `V`).

@include solutions/stage1/model.py#backward

- The calls mirror `forward` in reverse; each gradient has its parameter's shape.
- `dflat.reshape(emb_shape)` is the backward of flatten.

@include solutions/stage1/model.py#count_params

- `p.size` is the number of entries of an array; `int(...)` turns the NumPy sum into a Python int.
:::

::: breakit
Start `W2` with standard deviation 1 instead of 0.01, and print the first loss and the first 300 training losses (once the s1-l10 loop exists). The first loss is about 30 instead of 4.17, and the curve looks like a hockey stick: a steep fall over about 100 steps that only gets back to where good starting values would have begun.
:::

::: breakit
Start `W1` without the $1/\sqrt{ke}$ scale and compute `np.mean(np.abs(np.tanh(a)) > 0.97)` on a batch. You get about 0.88 (88% of units saturated) instead of about 0.035.
:::

::: recap
- The full forward and backward pass is a chain of your earlier functions, with the caches passed along.
- The model has 133,721 parameters for $V = 65, k = 8, e = 24, h = 512$; most are in $W_1$.
- $1/\sqrt{\text{fan\_in}}$ keeps the hidden layer at unit variance (you derived it); a tiny $W_2$ starts the loss at $\ln V$.
- Memorizing one batch is a fast test of the whole model.
:::
