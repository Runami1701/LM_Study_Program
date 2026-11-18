---
id: s2-l07
title: Transformer block + full model
stage: s2
order: 7
prereqs: [s2-l06]
files: [workspace/stage2/block.py, workspace/stage2/model.py]
functions: [MLP, Block, GPT, count_params]
tests: tests/s2/test_l07_model.py
est_minutes: 60
---

::: goal
You will put the pieces together. A **block** is attention plus a small MLP, each with a LayerNorm before it and a shortcut ("residual") around it. The full model stacks $L$ blocks between the embeddings and an output layer: `idx` `(B, t)` → `logits` `(B, t, V)` → loss.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | the vectors flowing through the model, $(B, T, d)$ |
| $d$ | model width |
| $h$ | number of heads |
| $L$ | number of blocks (layers) |
| $V$ | vocabulary size |
| $T$ | context length |
| $f$ | the part of a block that a shortcut goes around (attention or MLP) |
| $\ln_1, \ln_2$ | the two LayerNorms of a block |
| $\mathrm{attn}$, $\mathrm{mlp}$ | the attention and MLP parts of a block |
| $I$ | the identity: $I(x) = x$ |
| $\frac{\partial f}{\partial x}$ | how the output of $f$ changes with its input |
:::

::: numpy
### A list of modules

A Python list can hold modules, and a loop calls them one after another:

```python
self.blocks = [Block(d, n_heads, rng) for _ in range(n_layers)]   # in __init__
for block in self.blocks:                                         # in __call__
    x = block(x)
```

The given `Module` class finds the parameters inside lists too, and names them `blocks.0...`, `blocks.1...`, and so on.

### Flattening positions for the loss

The loss needs one row per prediction. `(B, t, V)` logits become `(B·t, V)`, and `(B, t)` targets become `(B·t,)`:

```python
loss = ops.cross_entropy(ops.reshape(logits, (B * t, V)), targets.reshape(-1))
```

`targets` is a plain NumPy array, so its own `.reshape` works; `logits` is a Tensor, so it uses `ops.reshape`.
:::

::: concept
### The block

A block does two steps:

1. Normalize $x$ with a LayerNorm, run attention on the result, and **add** the attention output to $x$.
2. Normalize the new $x$ with a second LayerNorm, run the MLP, and **add** its output to $x$.

In formulas:

$$
x \leftarrow x + \mathrm{attn}(\ln_1(x)), \qquad x \leftarrow x + \mathrm{mlp}(\ln_2(x))
$$

The MLP works on each position separately: `Linear(d, 4d)`, then GELU, then `Linear(4d, d)`. Attention moves information **between** positions; the MLP processes it **within** each position. The width $4d$ is the usual choice from GPT-2.

The LayerNorm sits **before** each part ("pre-LN"), not after it. GPT-2 and most modern models do this, because it trains more stably.

### Why the shortcuts help

A block computes $x + f(x)$, not $f(x)$. Its derivative is

$$
\frac{\partial (x + f(x))}{\partial x} = I + \frac{\partial f}{\partial x}
$$

The $I$ means: whatever $f$ does, the gradient from above passes through the block **unchanged**, plus whatever $f$ adds. With $L$ blocks stacked, the gradient reaches the first block directly through the chain of shortcuts. Without shortcuts it would be multiplied by $L$ local derivatives, and a product of many factors easily shrinks to nothing or explodes.

A second effect: a block whose $f$ outputs zeros is exactly the identity. So a fresh model with small weights starts close to "do nothing", which is an easy starting point. The check uses this: with the output projections set to zero, the block must return its input.

On your engine the shortcut is a plain `+`. Its backward passes the gradient to both inputs (a-l01), which is exactly the $I + \frac{\partial f}{\partial x}$ above.

### The full model

| step | operation | shape |
| --- | --- | --- |
| 1 | embeddings (s2-l01) | $(B, t, d)$ |
| 2 | $L$ blocks | $(B, t, d)$ |
| 3 | final LayerNorm `ln_f` | $(B, t, d)$ |
| 4 | output layer `head`, a `Linear(d, V)` without bias | $(B, t, V)$ |
| 5 | mean cross-entropy over all $B \cdot t$ positions | one number |

### Counting the parameters

One block:

| part | weights | biases and LayerNorm |
| --- | --- | --- |
| `qkv` | $d \cdot 3d = 3d^2$ | $3d$ |
| `proj` (attention) | $d^2$ | $d$ |
| MLP `fc` | $d \cdot 4d = 4d^2$ | $4d$ |
| MLP `proj` | $4d \cdot d = 4d^2$ | $d$ |
| two LayerNorms | | $2 \cdot 2d = 4d$ |
| **total** | $12d^2$ | $13d$ |

The whole model:

$$
\underbrace{L(12d^2 + 13d)}_{\text{blocks}} + \underbrace{Vd}_{\text{wte}} + \underbrace{Td}_{\text{wpe}} + \underbrace{2d}_{\text{ln\_f}} + \underbrace{dV}_{\text{head}}
$$

For $V = 65$, $T = 128$, $d = 128$, $L = 4$:

$$
4 \cdot 198{,}272 + 8{,}320 + 16{,}384 + 256 + 8{,}320 = 826{,}368
$$

The often-quoted short form $12Ld^2 + Vd + Td$ gives 811,136; it leaves out the biases, the LayerNorms and the separate `head`. Some models reuse `wte` as `head` ("weight tying"), which saves $dV$ parameters: only 8,320 here, but 2 million in Stage 3, where $V = 8000$ and $d = 256$. We keep them separate, for simplicity.

### Starting loss

All weights start at $N(0, 0.02^2)$, so the first logits are small and the first loss is close to $\ln 65 = 4.17$, as in s1-l09. The check tests this, and it also lets a tiny model memorize one batch, as in s1-l09.
:::

::: think
Q: A block's attention and MLP both output exactly zero. What does the block compute, and why is that a good starting point?
A: The identity: $x + 0 + 0 = x$. A model made of such blocks passes the embeddings straight to the output layer. Training then only has to learn useful changes, and the gradient flows through every block unchanged from the first step on.
:::

::: think
Q: How many parameters does one block have for $d = 128$, and what fraction of the whole model is that?
A: $12 \cdot 128^2 + 13 \cdot 128 = 196{,}608 + 1{,}664 = 198{,}272$. Four blocks make $793{,}088$ of the $826{,}368$ parameters: 96%.
:::

::: think
Q: Why does the MLP work on each position separately, while attention mixes positions?
A: Attention's matrix $P$ combines value vectors from different positions. The MLP's `Linear` layers multiply each position's vector by the same weights, independently, like a batch of separate vectors. Information moves between positions only in attention.
:::

::: hint level=1
Each class creates its parts in `__init__`, in the order given in the docstring, and chains them in `__call__`. The block is two lines with `+`. The model loops over its blocks and computes the loss only when targets are given.
:::

::: hint level=2
`MLP.__call__`: `self.proj(gelu(self.fc(x)))`. `Block.__call__`: `x = x + self.attn(self.ln1(x))`, then the same with `ln2` and `mlp`. `GPT.__call__`: embeddings, the loop over `self.blocks`, `self.ln_f`, `self.head`; then flatten logits and targets for `ops.cross_entropy`. `count_params`: sum `p.data.size` over `model.parameters()`.
:::

::: hint level=3
```pseudo
Block.__call__(x):
    x ← x + attn(ln1(x))
    x ← x + mlp(ln2(x))
    return x

GPT.__call__(idx, targets):
    x ← emb(idx)                              # (B, t, d)
    for each block: x ← block(x)
    logits ← head(ln_f(x))                    # (B, t, V)
    if targets is None: return logits, None
    loss ← cross_entropy(logits reshaped to (B·t, V), targets flattened)
    return logits, loss
```
:::

::: attempt
Files: `workspace/stage2/block.py` — fill in `MLP` and `Block`; `workspace/stage2/model.py` — fill in `GPT` and `count_params`.
Check: `python -m course check s2-l07`
:::

::: solution
@include solutions/stage2/block.py#MLP

@include solutions/stage2/block.py#Block

- The two `+` lines are the shortcuts; LayerNorm comes before each part.

@include solutions/stage2/model.py#GPT

- The loss averages over all $B \cdot t$ positions, so every position counts as one prediction.
- Without targets, the model returns only the logits; text generation (s2-l09) uses this.

@include solutions/stage2/model.py#count_params
:::

::: breakit
Remove both shortcuts (`x = self.attn(self.ln1(x))` and `x = self.mlp(self.ln2(x))`) and train a 4-block model for 300 steps (after s2-l09). The loss falls much more slowly than with shortcuts, and the gradients reaching the embeddings are far smaller. Print `np.linalg.norm(model.emb.wte.grad)` in both versions to see it.
:::

::: recap
- A pre-LN block computes $x + \mathrm{attn}(\ln_1(x))$, then $x + \mathrm{mlp}(\ln_2(x))$.
- Shortcuts give the derivative $I + \frac{\partial f}{\partial x}$, so gradients reach the first block unchanged.
- The model is embeddings → $L$ blocks → LayerNorm → output layer; it has $L(12d^2 + 13d) + 2Vd + Td + 2d = 826{,}368$ parameters here.
:::
