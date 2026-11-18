---
id: s3-l04
title: Bits per character
stage: s3
order: 4
prereqs: [s3-l03]
files: [workspace/stage3/metrics.py]
functions: [bits_per_char, evaluate_bpc]
tests: tests/s3/test_l04_bpc.py
est_minutes: 40
---

::: goal
You will measure models with **bits per character** (bpc), a number that you can compare across tokenizers. A per-token loss can't be compared between a character model and a subword model. You will derive the conversion and evaluate a model on held-out stories.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $L_i$ | the loss (in nats) of the $i$-th prediction |
| $N_{\text{tok}}$ | number of predicted tokens |
| $N_{\text{char}}$ | number of characters those tokens stand for |
| $\bar L$ | mean loss per token, $\frac{1}{N_{\text{tok}}}\sum_i L_i$ |
| bpc | bits per character |
| $\ln, \log_2$ | natural log and log base 2; $\log_2 x = \frac{\ln x}{\ln 2}$ |
| $V$ | vocabulary size |
| $T$ | context length |
| $r$ | a window number |
:::

::: numpy
### Logs in two bases

```python
import math
print(math.log(2))                     # 0.6931471805599453
print(math.log2(65), math.log(65) / math.log(2))   # 6.022367813028454 6.022367813028454
```

### From a mean back to a sum

The model returns the **mean** loss over its predictions. To add up losses over many batches, multiply each mean by its number of predictions:

```python
_, loss = model(x, y)                  # y has shape (batch, T)
total_nats += float(loss.data) * y.size
```

### Bytes per token, read with an index array

`n_bytes = np.array([len(b) for b in build_vocab(merges)])` holds the byte length of every token. `n_bytes[y].sum()` adds up the lengths of all target tokens at once (n-l03).
:::

::: concept
### Why per-token loss doesn't compare

Stage 2 reached a loss of about 1.5 nats **per character**. A BPE model might reach 3.5 nats **per token**. Which is better? You can't tell: one BPE token stands for about 4 characters, so predicting it is a much bigger task. We need the loss per unit of **text**, not per token.

### Deriving bits per character

The model's total loss over a text is the sum of its per-token losses, $\sum_i L_i = N_{\text{tok}} \bar L$ nats. That is the total "surprise" for the whole text, however it was cut into tokens. Divide by the number of characters to get nats per character, and by $\ln 2$ to turn nats into bits:

$$
\text{bpc} = \frac{\sum_i L_i}{\ln 2 \cdot N_{\text{char}}} = \frac{N_{\text{tok}}\,\bar L}{\ln 2 \cdot N_{\text{char}}}
$$

For a character model, $N_{\text{tok}} = N_{\text{char}}$, and this becomes $\frac{\bar L}{\ln 2}$ (s2-l09).

**Worked example.** A BPE model has a mean loss of 3.6 nats per token, and its tokens average 3.94 characters:

$$
\text{bpc} = \frac{3.6}{0.693 \times 3.94} = 1.32
$$

A character model on the same text would need a loss of $1.32 \times 0.693 = 0.91$ nats per character to match it.

We count characters as **UTF-8 bytes**. TinyStories is almost all English, so one character is one byte. For Korean text this measure is "bits per byte", and it stays fair across tokenizers, because every tokenizer covers the same bytes.

**A check.** A model that guesses uniformly over the 256 byte tokens has loss $\ln 256$ per token, and each token is one byte:

$$
\text{bpc} = \frac{\ln 256}{\ln 2} = \log_2 256 = 8
$$

exactly the 8 bits that a byte holds. The check tests this with a fake uniform model.

### Evaluating on held-out text

1. Encode the held-out text.
2. Cut the ids into consecutive windows of $T + 1$ tokens: window $r$ is `ids[r*T : r*T + T + 1]`. Its first $T$ tokens are the input, its last $T$ tokens the targets.
3. For a batch of windows, add `loss × number of targets` to the total nats, and the bytes of all target tokens to the total characters.
4. Return the ratio, converted with `bits_per_char`.

Windows step by $T$, so every token except the first is predicted once. The first prediction in each window has little context, which makes the result slightly pessimistic; with $T = 256$ the effect is small.
:::

::: think
Q: Model A (characters) has 1.40 nats per character. Model B (BPE, 3.94 characters per token) has 3.70 nats per token. Which is better?
A: A: $\frac{1.40}{0.693} = 2.02$ bpc. B: $\frac{3.70}{0.693 \times 3.94} = 1.36$ bpc. B is better, even though its per-token loss is much larger.
:::

::: think
Q: Why count the characters of the **target** tokens, and not of all the text?
A: The total loss covers only the predicted tokens. The characters must match them, otherwise the first token of every window would add characters without adding loss, and the result would look too good.
:::

::: hint level=1
`bits_per_char` is one division. `evaluate_bpc` encodes once, then loops over batches of windows, keeping two running totals: nats and characters.
:::

::: hint level=2
`total_nats / (math.log(2) * n_chars)`. In `evaluate_bpc`: `n_windows = (len(ids) - 1) // T`; build `x` and `y` with `np.stack` of the window slices; add `float(loss.data) * y.size` and `int(n_bytes[y].sum())`; call `gc.collect()` after each batch.
:::

::: hint level=3
```pseudo
evaluate_bpc(model, text, merges, T, batch_size):
    ids ← encode(text, merges);  n_bytes ← byte length of every token
    n_windows ← (number of ids − 1) // T
    for each group of batch_size windows:
        x ← for each window r: ids[r·T : r·T + T]
        y ← for each window r: ids[r·T + 1 : r·T + T + 1]
        _, loss ← model(x, y)
        nats  += loss · (number of entries of y)
        chars += sum of n_bytes over all of y
    return bits_per_char(nats, chars)
```
:::

::: attempt
File: `workspace/stage3/metrics.py` — fill in `bits_per_char` and `evaluate_bpc`.
Check: `python -m course check s3-l04`
:::

::: solution
@include solutions/stage3/metrics.py#bits_per_char

@include solutions/stage3/metrics.py#evaluate_bpc

- `float(loss.data) * y.size` turns the batch's mean loss back into a sum.
- `n_bytes[y].sum()` counts the characters of exactly the predicted tokens.
- `gc.collect()` frees each batch's graph, as in s2-l09.
:::

::: breakit
After your Stage 3 run, evaluate the trained model on the same 300,000 held-out characters twice: once with windows of $T = 256$ and once with $T = 16$ (the model accepts shorter inputs). The bpc with $T = 16$ comes out clearly higher, because each prediction sees on average only about 8 tokens, roughly 30 characters, of the story before it. Context is worth bits.
:::

::: recap
- Per-token losses can't be compared across tokenizers; bits per character can.
- bpc $= \frac{\sum_i L_i}{\ln 2 \cdot N_{\text{char}}}$; a uniform byte model scores exactly 8.
- Evaluate on consecutive windows of held-out text, counting the characters of the predicted tokens only.
:::
