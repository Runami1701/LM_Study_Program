---
id: s3-l05
title: Scale up
stage: s3
order: 5
prereqs: [s3-l04]
files: [workspace/stage3/train_bpe_lm.py]
functions: [estimate_hours, tokens_for_hours]
tests: tests/s3/test_l05_scale.py
est_minutes: 45
---

::: goal
You will plan a long training run before you start it. You derive the cost of training, about $6ND$ FLOPs for $N$ parameters and $D$ tokens; you turn it into hours using your machine's measured speed; and you pick a model size and a token count that fit into one night.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $N$ | number of parameters that take part in matrix products |
| $D$ | number of training tokens |
| $C$ | total training compute, in FLOPs |
| FLOPS | FLOPs per second that training actually achieves |
| $d, L, T, V$ | width, number of blocks, context length, vocabulary size |
| $W$ | a weight matrix of shape $(n, m)$ |
| $x$ | one token's input vector to $W$, length $n$ |
:::

::: numpy
### Units and big numbers

Python writes large numbers with `e`: `50e9` is $50 \times 10^9$, and `2.5e7` is 25 million. Underscores make long numbers readable and change nothing: `26_600_000 == 26600000`.

```python
N, D = 5_800_000, 24_600_000
print(6 * N * D)              # 856080000000000   (8.6e14 FLOPs)
print(6 * N * D / 50e9 / 3600)   # 4.756   hours at 50 GFLOPS
```

### Whole numbers

`int(x)` cuts off the decimals: `int(13.7)` is `13`. `tokens_for_hours` returns a whole number of tokens.
:::

::: concept
### The model for this stage

| setting | Stage 2 | Stage 3 |
| --- | --- | --- |
| vocabulary $V$ | 65 characters | 4,096 BPE tokens |
| width $d$ | 128 | 256 |
| heads | 4 | 8 |
| blocks $L$ | 4 | 6 |
| context $T$ | 128 characters | 256 tokens (about 1,000 characters) |
| parameters | 0.83 M | 6.90 M |

With the formula of s2-l07: $6 \cdot (12 \cdot 256^2 + 13 \cdot 256) + 2 \cdot 4096 \cdot 256 + 256 \cdot 256 + 2 \cdot 256 = 6{,}901{,}760$.

### Deriving $6ND$

Take one weight matrix $W$ of shape $(n, m)$ and one token. **Forward:** $y = xW$ computes $m$ outputs, each with $n$ multiplications and about $n$ additions: $2nm$ FLOPs, that is, **2 FLOPs per weight per token**. **Backward:** two products of the same size, $dx = dy\,W^\top$ and $dW = x^\top dy$ (s1-l06): **4 FLOPs per weight per token**. Together:

$$
C \approx 6 N D
$$

Two notes on $N$:

- Embedding lookups (`wte`, `wpe`) copy rows and multiply nothing, so they don't count. Here $N = 6{,}901{,}760 - 4096 \cdot 256 - 256 \cdot 256 = 5{,}787{,}648$.
- Attention also multiplies queries with keys and weights with values: about $4 L T d$ extra FLOPs per token in the forward pass, which adds roughly 15% here. The formula is an estimate; that is fine for planning.

### From FLOPs to hours

$$
\text{hours} = \frac{6 N D}{\text{FLOPS} \cdot 3600}
$$

Which FLOPS? In s0-l00 you measured the speed of one big matrix product, about 450 GFLOPS on the reference machine. Training reaches only a fraction of that: it runs many smaller products, plus softmax, LayerNorm, GELU and the Python overhead of the engine. **Measure** it instead: run the training script for a few minutes, read the tokens per second from its log, and compute

$$
\text{effective FLOPS} = 6 N \times \text{tokens per second}
$$

On the reference machine the Stage 3 model trained at **S3_TOKS** tokens per second, which is **EFF_GFLOPS** GFLOPS effective, about EFF_PCT of the matrix-product speed.

### How many tokens?

Research on language models (the "Chinchilla" study) found that for a fixed compute budget, a model learns best from about **20 tokens per parameter**. For $N = 5.8$ M that is $D \approx 116$ M tokens, which would take about **CHIN_HOURS** hours here. Too long for one night.

So we choose the other way round: fix the time, and derive $D$:

$$
D = \frac{\text{hours} \cdot 3600 \cdot \text{FLOPS}}{6N}
$$

The default config trains for 6,000 steps of $16 \times 256 = 4{,}096$ tokens: $D = 24.6$ M tokens, about 4 tokens per parameter, and about **RUN_HOURS** hours at the reference speed. Our tokenized data has 26.6 M tokens, so the run sees almost every story once. A smaller model ($d = 192$) would get more tokens per parameter in the same time; a larger one would get fewer. Both are reasonable experiments after the boss.

### Running it overnight

`python -m workspace.stage3.train_bpe_lm`. The first start learns the merges and tokenizes the data (a few minutes, once). Then it trains, saving a checkpoint every 500 steps. If the run stops, set `CONFIG["resume"] = True` and start it again. At the end it measures bits per character on 300,000 held-out characters and writes `runs/s3/results.json` with a sample story.

Use your Stage 2 code: `train_bpe_lm.py` calls your `train` from `workspace/stage2/train_tf.py`, with your GPT, AdamW and schedule. Nothing new has to be written for the model itself.
:::

::: think
Q: Why does the backward pass cost about twice the forward pass?
A: For each weight matrix, the forward pass does one product ($xW$). The backward pass does two of the same size: one for the input gradient ($dy\,W^\top$) and one for the weight gradient ($x^\top dy$).
:::

::: think
Q: Your machine trains at 1,500 tokens per second with $N = 5.8$ M. How long do 24.6 M tokens take? What is the effective FLOPS?
A: $\frac{24.6\text{M}}{1500} = 16{,}400$ s, about 4.6 hours. Effective FLOPS $= 6 \cdot 5.8\text{M} \cdot 1500 \approx 5.2 \times 10^{10}$, about 52 GFLOPS.
:::

::: think
Q: Why use the measured tokens per second instead of the GFLOPS from s0-l00?
A: s0-l00 measured one large, ideal matrix product. Training also runs smaller products, element-wise work (softmax, LayerNorm, GELU) and Python overhead, so it reaches only a fraction of that speed. The measured tokens per second include all of it.
:::

::: hint level=1
Both functions rearrange the same equation: hours $= \frac{6ND}{\text{FLOPS} \cdot 3600}$.
:::

::: hint level=2
`estimate_hours`: `6.0 * N * D / flops_per_s / 3600.0`. `tokens_for_hours`: solve for $D$, and wrap the result in `int(...)`.
:::

::: hint level=3
```pseudo
estimate_hours(N, D, flops_per_s):   return 6 · N · D / flops_per_s / 3600
tokens_for_hours(N, hours, flops_per_s):  return int(hours · 3600 · flops_per_s / (6 · N))
```
:::

::: attempt
File: `workspace/stage3/train_bpe_lm.py` — fill in `estimate_hours` and `tokens_for_hours`.
Check: `python -m course check s3-l05`
Then start the long run: `python -m workspace.stage3.train_bpe_lm`, and take on the boss when it finishes.
:::

::: solution
@include solutions/stage3/train_bpe_lm.py#estimate_hours

@include solutions/stage3/train_bpe_lm.py#tokens_for_hours

- The two functions undo each other, which the check tests.
:::

::: breakit
Before the long run, start the script and stop it (Ctrl+C) after the first few log lines. Read the tokens per second, compute your effective FLOPS, and compare it with your GFLOPS from s0-l00. Then use `estimate_hours` to predict the full run's time, and check your prediction against the time in `results.json` the next morning.
:::

::: recap
- Training costs about $6ND$ FLOPs: 2 per weight per token forward, 4 backward.
- Plan with the measured speed: effective FLOPS $= 6N \times$ tokens per second.
- About 20 tokens per parameter is ideal; on a CPU, fit $D$ to the hours you have.
:::
