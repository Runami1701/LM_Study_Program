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
Wire your layers into the full model, `x` `(B, k)` → embed → flatten → linear → tanh → linear → `logits` `(B, V)` → loss, with its backward pass. Initialize it so the first loss is about $\ln V$, and prove the gradients are right by overfitting one batch.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C$ | embedding table $(V, e)$ |
| $W_1, b_1$ | first layer: $(k e, h)$ and $(h,)$ |
| $W_2, b_2$ | output layer: $(h, V)$ and $(V,)$ |
| $V, k, e, h$ | vocabulary size, context length, embedding size, hidden units |
| $B$ | batch size |
| $a$ | pre-activations of the hidden layer, $(B, h)$; $a_j$ one unit |
| $x_i$ | the $i$-th of the $n = ke$ inputs to the first layer |
| $n$ | fan-in: the number of inputs feeding one unit ($n = ke$ for $W_1$) |
| $\mathrm{Var}$ | variance: the mean squared distance from the mean |
| $\sigma$ | standard deviation (square root of the variance) |
| $\mathbb{E}$ | expected value (average over the random draws) |
| $N(0, 1)$ | a standard normal distribution: mean 0, variance 1 |
| $W$ | a generic weight matrix ($W_{ij}$ its entries) |
| $j$ | a hidden unit index |
| $p_i$ | predicted probability of class $i$ |
| $z$ | logits |
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
| 6 | mean cross-entropy | scalar |

The backward pass visits the same steps in reverse, each using the function you already wrote: `cross_entropy_backward`, `linear_backward` (gives `dW2`, `db2`, `dt`), `tanh_backward`, `linear_backward` (gives `dW1`, `db1`, `dflat`), reshape back to `(B, k, e)`, `embedding_backward` (gives `dC`). The forward pass must keep every cache for this.

### How many parameters?

$$
\begin{aligned}
\text{params} &= \underbrace{V e}_{C} + \underbrace{k e \cdot h}_{W_1} + \underbrace{h}_{b_1} + \underbrace{h V}_{W_2} + \underbrace{V}_{b_2}
\end{aligned}
$$

For $V = 65$, $k = 8$, $e = 24$, $h = 512$:

$$
1{,}560 + 98{,}304 + 512 + 33{,}280 + 65 = 133{,}721
$$

Most of them sit in $W_1$: each of the $h$ hidden units looks at all $ke = 192$ inputs.

### Initializing $W_1$: keep pre-activations at unit size

Hidden unit $j$ computes $a_j = \sum_{i=1}^{n} x_i W_{ij}$ (ignore the zero bias), with $n = ke$ inputs. Suppose the inputs and weights are independent with mean 0, $\mathrm{Var}(x_i) = 1$ (our embeddings start as $N(0,1)$), and $\mathrm{Var}(W_{ij}) = \sigma^2$. For independent zero-mean factors, $\mathrm{Var}(x_i W_{ij}) = \mathbb{E}[x_i^2]\,\mathbb{E}[W_{ij}^2] = 1 \cdot \sigma^2$, and variances of independent terms add:

$$
\begin{aligned}
\mathrm{Var}(a_j) &= \sum_{i=1}^{n} \mathrm{Var}(x_i W_{ij}) \\
&= n \sigma^2
\end{aligned}
$$

With $\sigma = 1$ and $n = 192$, $a_j$ has standard deviation $\sqrt{192} \approx 13.9$: almost every tanh unit saturated at $\pm 1$, passing almost no gradient (s1-l06). Choosing $\sigma = \frac{1}{\sqrt{n}}$ gives $\mathrm{Var}(a_j) = n \cdot \frac{1}{n} = 1$. Hence $W_1 = \frac{N(0, 1)}{\sqrt{ke}}$, the **$1/\sqrt{\text{fan\_in}}$** rule.

Check it on the model: the fraction of hidden activations with $|t| > 0.97$ (i.e. $|a| > 2.1$) should be small, about 3–4% for unit-variance $a$. With $\sigma = 1$ it would be about 88%.

### Initializing $W_2$: start honest

Make $W_2$ tiny ($0.01 \cdot N(0,1)$) and $b_2 = 0$. Then every logit is nearly 0, softmax is nearly uniform, and the initial loss is nearly $\ln V = 4.17$ (s1-l04). A large $W_2$ would make the untrained model confidently wrong at random, starting at a loss of 10 or more; the first few hundred steps would only undo that ("hockey stick" curve), wasting time.

For the checks to reproduce your parameters exactly, draw them in a fixed order with the same `rng`: `C`, then `W1`, then `W2` (see the docstring).

### Overfit one batch: the best end-to-end test

Take 32 examples and train on only them. A correct model with 130K parameters can memorize 32 examples, so the loss must go to nearly 0 within a few hundred steps of plain gradient descent. If it plateaus, some gradient is wrong (or the update has the wrong sign). This test catches bugs that gradient checks on tiny models can miss, and it takes a second.
:::

::: think
Q: Why must the initial loss be close to $\ln V$, and what would a starting loss of 4.17 vs 12 tell you about $W_2$?
A: With no knowledge, the honest prediction is uniform, which scores $\ln V = 4.17$. A loss of 12 means the random initial logits are large and confidently wrong ($p_y \approx e^{-12}$): $W_2$ is too big. Training first has to shrink those logits, which shows as a sharp initial drop.
:::

::: think
Q: Derive the standard deviation of a pre-activation if $W_1$ were $N(0, 1)$ with $n = 192$, and say what that does to tanh.
A: $\mathrm{Var}(a_j) = n \cdot 1 \cdot 1 = 192$, so $\sigma_a \approx 13.9$. Most $|a_j| > 3$, where tanh is flat ($1 - t^2 < 0.01$): units are saturated, gradients through them vanish, and $W_1$ barely learns.
:::

::: think
Q: Your model overfits one batch to loss 0.02, but the full-model gradient check says `C` has relative error 0.3. How can both be true?
A: Overfitting only shows that the update lowers the loss on that batch; a model can still memorize 32 examples using $W_1$, $W_2$ and the biases even if the gradient for `C` is wrong (say, duplicates dropped, or zero). Both tests are needed.
:::

::: hint level=1
`forward` calls your five forward functions in order and returns all their caches together; `backward` unpacks them and calls the backward functions in reverse order. `predict` is `forward` without the loss.
:::

::: hint level=2
Keep `x`, the `(B, k, e)` embedding shape and `V` in the cache, because the embedding backward needs them. Un-flatten `dflat` with `.reshape` to that saved shape. For the parameters: `rng.standard_normal(shape)`, `/ np.sqrt(k * e)`, `* 0.01`, `np.zeros(h)`, then `.astype(np.float32)` for each. `count_params`: sum of `.size`.
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
File: `workspace/stage1/model.py` — implement `init_params`, `predict`, `forward`, `backward`, `count_params`.
Check: `python -m course check s1-l09`
:::

::: solution
@include solutions/stage1/model.py#init_params

- Draw order `C`, `W1`, `W2` from one `rng` makes runs reproducible; the biases need no randomness.
- `/ np.sqrt(k * e)` is the $1/\sqrt{\text{fan\_in}}$ rule; `* 0.01` keeps the first logits near 0. Drawing in float64 and casting once with `.astype(np.float32)` avoids accidental float64 parameters.

@include solutions/stage1/model.py#predict

@include solutions/stage1/model.py#forward

- Each forward function returns its cache; `forward` bundles them with what the embedding backward needs (`x`, the embedding shape, `V`).

@include solutions/stage1/model.py#backward

- The calls mirror `forward` in reverse; shapes are noted on each line, and each gradient matches its parameter's shape.
- `dflat.reshape(emb_shape)` is the backward of flatten.

@include solutions/stage1/model.py#count_params

- `p.size` is the number of entries of an array; `int(...)` turns the NumPy sum into a Python int.
:::

::: breakit
Initialize `W2` with standard deviation 1 instead of 0.01, and print the initial loss and the first 300 training losses (once s1-l10's loop exists). Expect an initial loss around 30 instead of 4.17, and a "hockey stick": a steep fall over the first ~100 steps that only gets back to where a good initialization started.
:::

::: breakit
Initialize `W1` without the $1/\sqrt{ke}$ scale and measure `np.mean(np.abs(np.tanh(a)) > 0.97)` on a batch. Expect about 0.88 (88% of units saturated) instead of about 0.035.
:::

::: recap
- The full forward and backward pass is a composition of your earlier functions, with caches threaded through.
- 133,721 parameters for $V = 65, k = 8, e = 24, h = 512$; most live in $W_1$.
- $1/\sqrt{\text{fan\_in}}$ keeps pre-activations at unit variance (you derived it); a tiny $W_2$ starts the loss at $\ln V$.
- Overfitting one batch is a fast end-to-end correctness test.
:::
