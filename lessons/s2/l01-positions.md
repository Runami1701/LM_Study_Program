---
id: s2-l01
title: Token + positional embeddings; targets at every position
stage: s2
order: 1
prereqs: [a-l04]
files: [workspace/stage2/layers.py]
functions: [get_batch_seq, Linear, Embeddings]
tests: tests/s2/test_l01_embeddings.py
est_minutes: 45
---

::: goal
You will build the input side of the transformer. A batch is now made of whole sequences: inputs `x` `(B, T)` and targets `y` `(B, T)`, shifted by one position. You also write a `Linear` module and an `Embeddings` module, which turns `(B, T)` ids into `(B, T, d)` vectors by adding a learned position vector to each character vector.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $T$ | context length: how many positions the model sees at once (128 here) |
| $t$ | a position, $0 \dots T-1$; also the length of a shorter input |
| $d$ | model width: the length of every position's vector (128 here) |
| $B$ | batch size |
| $V$ | vocabulary size |
| $W_{te}$ | character embedding table `wte`, shape $(V, d)$ |
| $W_{pe}$ | position embedding table `wpe`, shape $(T, d)$ |
| $h_{b,t}$ | the input vector of position $t$ in sequence $b$ |
| $s$ | the start of a window |
| $x_{b,t}, y_{b,t}$ | input id and target id at position $t$ of row $b$ |
| `Module` | a given base class that finds a model's parameters by their attribute names |
:::

::: numpy
### Inputs and targets from the same positions

The targets are the inputs moved one position to the right, so the same position table works with `+ 1`:

```python
data = np.arange(10)
idx = np.array([[2, 3, 4]])     # one window: positions 2, 3, 4
print(data[idx])                # [[2 3 4]]   inputs
print(data[idx + 1])            # [[3 4 5]]   targets: the next character at every position
```

### Attributes that hold modules

A module stores its parameters as attributes: `self.W = Tensor(...)`. A bigger module stores smaller modules the same way: `self.emb = Embeddings(...)`. The given `Module` class walks through these attributes and names every parameter, for example `blocks.0.attn.qkv.W`. Optimizers and checkpoints use these names.
:::

::: concept
### Every position is a training example

In Stage 1, one example was $k$ characters and one target. A transformer reads a whole window of $T$ characters and predicts the next character **at every position**. For a window that starts at $s$:

$$
x_{b,t} = \text{data}[s + t], \qquad y_{b,t} = \text{data}[s + t + 1], \qquad t = 0, \dots, T-1
$$

So `y` is `x` shifted one step to the left. With the text `hello!` and $T = 5$, starting at $s = 0$:

| position $t$ | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| input $x$ | h | e | l | l | o |
| target $y$ | e | l | l | o | ! |

Position 0 predicts `e` from `h`; position 4 predicts `!` from `hello`. One window gives $T$ training signals, with contexts of length 1 up to $T$. (The causal mask of s2-l05 stops position $t$ from looking at $x_{t+1}$, which is its own answer.) A window needs $T + 1$ characters, so starts run up to $n - T - 1$: draw them with `rng.integers(0, n - T)`, by the same argument as in s1-l01. `x` uses the positions table of s1-l01; `y` uses the same table plus one.

### Character embeddings, as before

`wte` `(V, d)` maps each id to a vector. Looking up `(B, T)` ids gives `(B, T, d)`. On your engine this is `wte[idx]`, the `getitem` operation, whose backward adds up repeated ids.

### Why positions need their own vectors

Attention (s2-l04) compares every position with every other position through dot products, and nothing in that computation knows *where* a vector sits. If you shuffle the positions, each output is shuffled the same way. To the model, "dog bites man" and "man bites dog" would differ only by chance. So we add a **learned position vector**, one per position:

$$
h_{b,t} = W_{te}[x_{b,t}] + W_{pe}[t]
$$

Shapes: the character part is `(B, T, d)` and the position part `(T, d)`. Broadcasting adds the position part to every sequence of the batch: all sequences use the same vector for position 3. In the backward pass, `unbroadcast` therefore **adds up** the gradient of `wpe` over the $B$ sequences.

If an input is shorter than $T$ (during text generation, say $t = 5$), use only the first $t$ position vectors: `wpe[0..t-1]`.

### The Linear module

`Linear(n_in, n_out)` computes $xW + b$ as a module, with small random weights: $W \sim N(0, 0.02^2)$, as in GPT-2. Your engine's `matmul` handles batches, so the same module works on `(B, T, n_in)` inputs.
:::

::: think
Q: A window of $T = 128$ positions gives how many training signals? Why are they worth less than 128 independent examples?
A: 128 predictions, one per position. They share most of their context (positions 50 and 51 see almost the same characters), so their errors are related. They are still very cheap: one forward pass computes all of them.
:::

::: think
Q: Why must the gradient of `wpe` be added up over the batch, and which part of your engine does it?
A: The vector of position $t$ was added to all $B$ sequences: one parameter, $B$ uses, so $B$ contributions. The `+` operation broadcasts `(T, d)` against `(B, T, d)`, and its `unbroadcast` sums over the extra front axis.
:::

::: think
Q: Without position vectors, could the model still tell "ab" from "ba"?
A: Not through attention: attention treats the context as an unordered collection of vectors (s2-l04). The causal mask leaks a little order information, because each position sees a different number of earlier positions, but far too little. The position vectors make order explicit.
:::

::: hint level=1
`get_batch_seq` is `get_batch` from s1-l01 with $T$ targets instead of one. `Linear` and `Embeddings` create Tensors in `__init__` and use engine operations in `__call__`.
:::

::: hint level=2
Targets: read `data` at `idx + 1`. Parameters: `Tensor((rng.standard_normal(shape) * 0.02).astype(np.float32))`. In `Embeddings.__call__`: `self.wte[idx]` plus `self.wpe[np.arange(t)]`. The `+` broadcasts, and its backward un-broadcasts.
:::

::: hint level=3
```pseudo
get_batch_seq(data, B, T, rng):
    starts ← rng.integers(0, n − T, size=B)
    idx ← starts as a column + [0 .. T−1] as a row            # (B, T)
    return data[idx] as int64, data[idx + 1] as int64

Embeddings.__call__(idx):
    t ← number of columns of idx
    return wte[idx] + wpe[0 .. t−1]                           # (B, t, d) + (t, d)
```
:::

::: attempt
File: `workspace/stage2/layers.py` — fill in `get_batch_seq`, `Linear` and `Embeddings`. The `Module` class is given.
Check: `python -m course check s2-l01`
:::

::: solution
@include solutions/stage2/layers.py#get_batch_seq

@include solutions/stage2/layers.py#Linear

- One module serves 2-D and batched 3-D inputs, because `matmul` treats the front axes as batch axes.

@include solutions/stage2/layers.py#Embeddings

- `self.wte[idx]` is the engine's `getitem`, whose backward adds up repeats. `self.wpe[np.arange(t)]` picks the first $t$ positions, so shorter inputs work.
- The `+` broadcasts `(t, d)` over the batch, and its backward adds `wpe`'s gradient up over the batch.
:::

::: breakit
Remove the position vectors (return only `self.wte[idx]`) and train for 500 steps once s2-l09 is done. The loss stays clearly worse than with position vectors at the same step: the model sees only a bag of characters, plus the little order information the mask leaks.
:::

::: recap
- One window of $T + 1$ characters gives $T$ next-character predictions; the targets are the inputs shifted by one.
- Each position gets a learned vector, added to the character vector; without it, attention cannot see order.
- The position table is shared by the whole batch, so its gradient is added up over the batch.
:::
