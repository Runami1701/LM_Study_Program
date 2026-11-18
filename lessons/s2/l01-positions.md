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
Set up the transformer's input side: batches of whole sequences `x` `(B, T)` with targets `y` `(B, T)` shifted by one, a `Linear` module, and an `Embeddings` module that adds a learned position vector to each token vector, `(B, T)` → `(B, T, d)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $T$ | context length (block size): the number of positions the model sees, 128 here |
| $t$ | a position, $0 \dots T-1$; also the actual length of a shorter input |
| $d$ | model width: the size of every position's vector (128 here) |
| $B$ | batch size |
| $V$ | vocabulary size |
| $W_{te}$ | token embedding table `wte`, shape $(V, d)$ |
| $W_{pe}$ | position embedding table `wpe`, shape $(T, d)$ |
| $h_{b,t}$ | the input vector of position $t$ in sequence $b$ |
| $s$ | a window's start position |
| $x_{b,t}, y_{b,t}$ | input id and target id at position $t$ of row $b$ |
| `Module` | given base class: finds a model's parameters by attribute name |
:::

::: concept
### Every position is a training example

In stage 1, one `(x, y)` pair was $k$ characters and one target. A transformer reads a whole window of $T$ characters at once and predicts the next character **at every position**. For a window starting at $s$:

$$
x_{b,t} = \text{data}[s + t], \qquad y_{b,t} = \text{data}[s + t + 1], \qquad t = 0, \dots, T-1
$$

so `y` is `x` shifted one step left. With the text `hello!` and $T = 5$ from $s = 0$:

| position $t$ | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| input $x$ | h | e | l | l | o |
| target $y$ | e | l | l | o | ! |

Position 0 predicts `e` from `h`; position 4 predicts `!` from `hello`. One window gives $T$ training signals with context lengths 1 to $T$. (The causal mask of s2-l05 ensures position $t$ cannot peek at $x_{t+1}$, which *is* its answer.) A window needs $T + 1$ characters, so starts run up to $n - T - 1$: `rng.integers(0, n - T)`, by the same argument as s1-l01. Building `x` is the s1-l01 broadcast, `(B, 1) + (1, T)`; `y` uses the same indices plus one.

### Token embeddings: as before

`wte` `(V, d)` maps each id to a vector; looking up `(B, T)` ids gives `(B, T, d)`. On your engine this is `wte[idx]`, the `getitem` op, whose backward accumulates repeated ids.

### Why positions need their own vectors

Attention (s2-l04) compares every position with every other through dot products, and nothing in that computation knows *where* a vector sits: shuffle the positions and each output is just shuffled the same way. To the model, "dog bites man" and "man bites dog" would differ only by luck. So we add a **learned position vector**, one per position:

$$
h_{b,t} = W_{te}[x_{b,t}] + W_{pe}[t]
$$

Shapes: token part `(B, T, d)` plus position part `(T, d)`. The position part broadcasts over the batch: every sequence uses the same vector for position 3. In the backward pass, `unbroadcast` therefore **sums** the gradient of `wpe` over the $B$ sequences.

If an input is shorter than $T$ (during generation, say $t = 5$), use only the first $t$ position vectors: `wpe[0..t-1]`.

### The given Module class and Linear

`Module` (given in `layers.py`) finds every Tensor attribute of a module, recursing into sub-modules and lists of them, and names them by path: `blocks.0.attn.qkv.W`. Optimizers and checkpoints use those names.

`Linear(n_in, n_out)` is $xW + b$ as a module, with $W \sim N(0, 0.02^2)$: small random weights, as GPT-2 does. (A $1/\sqrt{\text{fan\_in}}$ scale, s1-l09, would also work; LayerNorm in s2-l02 makes the transformer less sensitive to this choice.) Because your engine's `matmul` handles batches, the same module works on `(B, T, n_in)`.
:::

::: think
Q: A window of $T = 128$ positions gives how many training signals, and why aren't they 128 independent examples?
A: 128 predictions, one per position. They share most of their context (position 50 and 51 see almost the same characters) and their errors are correlated, so they are worth less than 128 independent samples. They are still far cheaper: one forward pass computes all of them.
:::

::: think
Q: Why must the gradient of `wpe` be summed over the batch, and which part of your engine does it?
A: Position $t$'s vector was added to all $B$ sequences: one parameter, $B$ uses, so $B$ contributions. The `+` op broadcasts `(T, d)` against `(B, T, d)`, and its `unbroadcast` sums the extra leading axis.
:::

::: think
Q: Without position embeddings, could the model still tell "ab" from "ba"?
A: Not through attention alone: attention treats the context as an unordered set of vectors (s2-l04). The causal mask leaks a little order information (each position sees a different number of predecessors), but nowhere near enough. The position vectors make order explicit.
:::

::: hint level=1
`get_batch_seq` is s1-l01's `get_batch` with `T` targets instead of one. `Linear` and `Embeddings` create Tensors in `__init__` and use engine operations in `__call__`.
:::

::: hint level=2
Targets: index `data` with `idx + 1`. Parameters: `Tensor((rng.standard_normal(shape) * 0.02).astype(np.float32))`. In `Embeddings.__call__`, `self.wte[idx]` and `self.wpe[np.arange(t)]`, then `+` (which broadcasts and unbroadcasts).
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
File: `workspace/stage2/layers.py` — implement `get_batch_seq`, `Linear`, `Embeddings` (the `Module` class is given).
Check: `python -m course check s2-l01`
:::

::: solution
@include solutions/stage2/layers.py#get_batch_seq

@include solutions/stage2/layers.py#Linear

- One module for 2-D and batched 3-D inputs, because `matmul` treats leading axes as batch axes.

@include solutions/stage2/layers.py#Embeddings

- `self.wte[idx]` is the engine's `getitem` (accumulating backward); `self.wpe[np.arange(t)]` selects the first $t$ positions, so shorter inputs work.
- The `+` broadcasts `(t, d)` over the batch, and its backward sums `wpe`'s gradient over the batch.
:::

::: breakit
Remove the position embedding (return only `self.wte[idx]`) and train for 500 steps after s2-l09. Expect a clearly worse loss than with positions at the same step, because the model only sees a bag of characters plus whatever order the causal mask leaks.
:::

::: recap
- One window of $T + 1$ characters gives $T$ next-character predictions; targets are inputs shifted by one.
- Positions get learned vectors added to the token vectors; without them attention can't see order.
- The position table is shared across the batch, so its gradient is summed over the batch.
:::
