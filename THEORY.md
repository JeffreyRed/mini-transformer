# Theory & Code Walkthrough — mini-transformer

> Step 3 of the mini-LLM series. Prerequisite: [mini-self-attention](https://github.com/JeffreyRed/mini-self-attention).

---

## Table of Contents

1. [What this step adds](#1-what-this-step-adds)
2. [Positional encoding — the problem and the solution](#2-positional-encoding--the-problem-and-the-solution)
3. [The causal mask](#3-the-causal-mask)
4. [Stacked blocks — why depth matters](#4-stacked-blocks--why-depth-matters)
5. [Pre-norm vs post-norm residuals](#5-pre-norm-vs-post-norm-residuals)
6. [GELU activation](#6-gelu-activation)
7. [Weight tying](#7-weight-tying)
8. [Training — LR warmup, label smoothing, perplexity](#8-training--lr-warmup-label-smoothing-perplexity)
9. [Text generation — greedy, temperature, top-k](#9-text-generation--greedy-temperature-top-k)
10. [Code walkthrough](#10-code-walkthrough)
11. [Full data flow](#11-full-data-flow)
12. [How this maps onto GPT-2](#12-how-this-maps-onto-gpt-2)

---

## 1. What this step adds

`mini-self-attention` built one encoder block. It had no sense of word order,
no stacking, no way to generate text.

`mini-transformer` adds four things:

| Addition | File | Why it matters |
|---|---|---|
| Positional encoding | `positional_encoding.py` | Attention is order-blind without it |
| Causal mask | `model.py` | Prevents the model from "cheating" by looking at future tokens |
| Stacked blocks | `model.py` | Each layer refines the previous layer's representations |
| `generate()` method | `model.py` | The model can now produce new text |

Everything else — multi-head attention, residuals, LayerNorm, feedforward, cross-entropy training — was already built in step 2.

---

## 2. Positional encoding — the problem and the solution

### The problem

Self-attention computes a weighted sum over all positions simultaneously.
The computation for position `i` uses the vectors at positions `j = 0..T`,
but the indices `i` and `j` never explicitly appear in the math.

This means the attention mechanism is **permutation-invariant**: if you
shuffle all the words, the set of attention scores is the same, just
rearranged. The model cannot distinguish:

```
"I like cats"   →   same scores as   →   "cats like I"
```

### The solution — sinusoidal encoding

Add a fixed, unique vector to each token's embedding before the first block.
The vector encodes the position using sine and cosine waves at different frequencies:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

Each dimension uses a different frequency. The first few dimensions oscillate
rapidly (high frequency — fine-grained position), while the last few
oscillate slowly (low frequency — coarse position). It is like a continuous
analogue of a binary counter.

```
position 0:  [ 0.000,  1.000,  0.000,  1.000, ... ]
position 1:  [ 0.841,  0.540,  0.100,  0.995, ... ]
position 2:  [ 0.909, -0.416,  0.200,  0.980, ... ]
position 3:  [ 0.141, -0.990,  0.296,  0.955, ... ]
```

**Key properties:**

- **Fixed, not learned** — one fewer thing to optimise; works on sequences longer than any seen during training.
- **Added, not concatenated** — dimensionality is preserved; no extra parameters.
- **Relative positions are learnable** — `PE(pos + k)` is a linear function of `PE(pos)` for any constant offset `k`. The attention weights can therefore learn to focus on "two positions back" without explicitly encoding this.

The `outputs/positional_encoding.png` plot shows this matrix directly —
each row is the vector added to the token at that position.

![attention_heatmap](outputs/positional_encoding.png)
---

## 3. The causal mask

### Why it is needed

In `mini-self-attention` every position could attend to every other position.
That is fine for an encoder (understanding a complete sentence), but breaks
language modelling.

When training a language model, the task at position `i` is to predict token
`i+1` using only tokens `0..i`. If position `i` can attend to token `i+1`
during training, it trivially reads the answer — and learns nothing.

At inference (generation) time there *is* no future token yet, so the model
would fail on inputs it appeared to handle during training.

### How it is built

An upper-triangular boolean matrix, `True` where attention should be blocked:

```
seq_len = 4:

         pos_0  pos_1  pos_2  pos_3
  pos_0 [ F,     T,     T,     T  ]   pos_0 sees only itself
  pos_1 [ F,     F,     T,     T  ]   pos_1 sees pos_0 and itself
  pos_2 [ F,     F,     F,     T  ]   pos_2 sees pos_0..2
  pos_3 [ F,     F,     F,     F  ]   pos_3 sees everything
```

`True` positions are filled with `-inf` before the softmax.
`softmax(-inf) = 0` — those positions get zero attention weight.

```python
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
# diagonal=1 means the main diagonal (attending to yourself) is allowed
```

---

## 4. Stacked blocks — why depth matters

A single attention block can learn which pairs of words are related.
A second block, operating on the output of the first, can learn
**relationships between relationships**.

Concretely: after block 1, the vector for `"like"` already blends some
information from `"cats"` and `"I"`. Block 2's attention can now ask
questions about those blended representations — it is operating on
*partially processed* context, not raw embeddings.

This is why GPT-2 uses 12 blocks, GPT-3 uses 96, and deeper models
consistently outperform shallower ones: each layer provides a more
abstract representation for the next layer to work with.

In `main.py`, `compare_layers()` prints the cosine similarity between
each word's representation at layer 1 vs layer 2 — you can observe that
some words change significantly between layers (they were "refined" by
the second block) while others are more stable.

---

## 5. Pre-norm vs post-norm residuals

The original transformer (Vaswani 2017) used **post-norm**:

```
x = LayerNorm(x + sublayer(x))
```

Modern models (GPT-2 onwards) use **pre-norm**:

```
x = x + sublayer(LayerNorm(x))
```

Pre-norm normalises the *input* to each sublayer rather than the output.
This keeps gradient magnitudes more stable early in training — the residual
stream `x` is never normalised, so gradients can flow back through the
skip connection without being rescaled at each block. Pre-norm models
train more reliably without learning rate tuning and rarely need warmup
as much as post-norm models.

This project uses pre-norm throughout.

---

## 6. GELU activation

`mini-self-attention` used ReLU in the feedforward block:

```
ReLU(x) = max(0, x)   — hard zero for any negative input
```

`mini-transformer` uses GELU (Gaussian Error Linear Unit):

```
GELU(x) ≈ x · σ(1.702 · x)
```

GELU smoothly gates negative inputs rather than hard-zeroing them. Small
negative values still contribute a little. This gives slightly smoother
gradients and has become the standard for transformer feedforward layers
(GPT-2, BERT, T5, LLaMA all use GELU or a variant).

In practice on small corpora the difference is minor, but using GELU
now means mini-gpt is one step closer to production-style code.

---

## 7. Weight tying

The model has two matrices that map between token space and embedding space:

- **Input embedding** `E`: `(vocab_size × emb_dim)` — maps token index → vector
- **Output projection** `W`: `(emb_dim × vocab_size)` — maps vector → logits

These are transpositions of each other conceptually — both describe a
correspondence between tokens and directions in embedding space.

Weight tying sets `W = E^T`, sharing the same underlying data:

```python
self.head.weight = self.embedding.weight
```

**Benefits:**
- Reduces parameter count by `vocab_size × emb_dim` (significant for large vocabularies)
- Forces the model to learn a single consistent representation: the direction in embedding space that represents a token is the same whether the token is an input or a predicted output
- Empirically improves perplexity, especially on small datasets

Used in GPT-2, GPT-J, LLaMA, and most modern language models.

---

## 8. Training — LR warmup, label smoothing, perplexity

### Learning rate warmup

Transformers are sensitive to the learning rate at initialisation.
If the LR starts high, the attention weights and layer norms receive large
gradient updates before they are calibrated, which can send the loss
to `nan` or lock the model into a bad local minimum.

The solution: start at LR ≈ 0 and ramp linearly up to the target LR
over `warmup_steps` gradient steps.

```
step 0:   lr = 0.0
step 25:  lr = target / 2
step 50:  lr = target       ← full learning rate from here
```

After warmup, LR holds constant. (GPT-3 uses cosine decay after warmup —
that is the only remaining piece for mini-gpt.)

### Label smoothing

Instead of training the model to output probability 1.0 for the correct
token, we train it to output `1 - ε` for the correct token and `ε / (V-1)`
for all others. Here `ε = 0.1`.

This prevents overconfident predictions and acts as a regulariser —
important on tiny corpora where the model can otherwise memorise the
training data perfectly without generalising.

### Perplexity

```
Perplexity = exp(cross-entropy loss)
```

Interpretation: a perplexity of `N` means the model is roughly as uncertain
as if it had to choose uniformly among `N` words at each step.

- At random initialisation: perplexity ≈ `vocab_size` (18 for this corpus)
- After training: should drop well below `vocab_size`
- Perfect memorisation: perplexity approaches 1

Perplexity is reported alongside loss in the training loop and plotted on
a twin-axis chart in `outputs/loss_perplexity.png`.

![loss_perplexity](outputs/loss_perplexity.png)
---

## 9. Text generation — greedy, temperature, top-k

After training, the model can generate text autoregressively:

```
1. Encode prompt as token indices
2. Feed to model → get logits for the next position
3. Sample (or argmax) one token from the logits
4. Append to sequence → go to 2
5. Stop when EOS is predicted or max_new_tokens is reached
```

Three sampling strategies are available:

### Greedy
Always pick the highest-probability token:
```
next_id = argmax(logits)
```
Deterministic. Fast. Tends to produce repetitive, "safe" output.

### Temperature sampling
```
probs = softmax(logits / T)
next_id = multinomial(probs)
```
`T > 1` flattens the distribution → more random, creative, sometimes incoherent.
`T < 1` sharpens it → more focused, less varied.
`T = 1` is neutral (unmodified softmax).

### Top-k sampling
```
keep only the top-k logits
zero out the rest
sample from the truncated distribution
```
Prevents the model from sampling very unlikely tokens even when temperature
is high. Combining `temperature + top_k` is the standard approach for
language model decoding (used in GPT-2 demo, ChatGPT, etc.).

---

## 10. Code walkthrough

### `tokenizer.py`

New vs `mini-self-attention`: three special tokens are added at the front
of the vocabulary before any corpus words:

```python
SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]
```

- `<PAD>` at index 0 is the padding token — excluded from the loss via
  `CrossEntropyLoss(ignore_index=0)`
- `<BOS>` is prepended to every sentence before encoding
- `<EOS>` is appended — the model learns to predict it when a sentence ends,
  which is what stops generation

---

### `positional_encoding.py`

```python
pe  = torch.zeros(max_len, emb_dim)
pos = torch.arange(0, max_len).unsqueeze(1).float()    # (T, 1)
div = torch.exp(
    torch.arange(0, emb_dim, 2).float()
    * -(math.log(10000.0) / emb_dim)
)                                                       # (D/2,)
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
self.register_buffer("pe", pe.unsqueeze(0))            # (1, T, D)
```

`register_buffer` saves the tensor in the model's `state_dict` (so it's
included when you call `torch.save`) but does NOT add it to `parameters()`
(so it is never updated by the optimizer).

Forward pass simply adds a slice of the buffer:

```python
x = x + self.pe[:, :x.size(1), :]
```

---

### `attention.py`

Identical to `mini-self-attention` with one addition:

```python
weights = torch.nan_to_num(weights, nan=0.0)
```

When an entire row of the attention score matrix is `-inf` (e.g. padding
positions), `softmax` produces `nan`. This line replaces those with zero
so the model does not propagate `nan` through the rest of the computation.

---

### `model.py`

**`TransformerBlock`** uses pre-norm layout (LayerNorm before the sublayer):

```python
# Pre-norm self-attention
attn_out, weights = self.attn(self.norm1(x), mask)
x = x + self.drop(attn_out)

# Pre-norm feedforward
x = x + self.drop(self.ff(self.norm2(x)))
```

Compare with mini-self-attention's post-norm:
```python
# Post-norm (mini-self-attention)
x = self.norm1(x + attn_out)
```

**`MiniTransformer._causal_mask()`** builds the upper-triangular mask:

```python
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool, device=device),
    diagonal=1,
)
return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T) broadcasts over batch and heads
```

**Weight tying:**

```python
self.head.weight = self.embedding.weight
```

This assigns the same underlying tensor to both. PyTorch's autograd tracks
the sharing correctly — gradients flowing back through the output projection
are automatically added to those flowing back through the embedding.

**`generate()`** implements the autoregressive loop:

```python
for _ in range(max_new_tokens):
    logits, _ = self(ids)              # forward pass on growing sequence
    next_logits = logits[:, -1, :]    # only the last position matters
    # ... sample or argmax ...
    ids = torch.cat([ids, next_id], dim=1)
```

---

### `dataset.py`

**`CausalDataset`** wraps BOS/EOS-encoded sentences as shifted pairs:

```
Sentence:   <BOS>  I  like  cats  <EOS>
  input:    <BOS>  I  like  cats
  target:      I  like  cats  <EOS>
```

At position 0, the model sees `<BOS>` and must predict `I`.
At position 1, it sees `<BOS> I` and must predict `like`.
...and so on until it predicts `<EOS>`.

**`make_collate(pad_idx)`** returns a closure rather than a bare function —
this is necessary because `DataLoader`'s `collate_fn` argument takes a
zero-argument callable per batch item, but we need to pass `pad_idx`.

---

### `train.py`

**LR warmup** is applied per optimizer step (not per epoch):

```python
def get_lr(current_step):
    if current_step < warmup_steps:
        return lr * current_step / warmup_steps
    return lr

for pg in optimizer.param_groups:
    pg["lr"] = get_lr(step)
```

**Gradient clipping** before the optimizer step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This rescales the entire gradient vector if its L2 norm exceeds 1.0,
preventing rare but large gradient updates from destabilising training.

---

### `utils.py`

**`generate_text()`** is the clean API wrapper:

```python
words      = prompt.strip().split()
prompt_ids = tokenizer.encode(words, add_special=True)
src        = torch.tensor([prompt_ids])
out_ids    = model.generate(src, ...)
return " ".join(tokenizer.decode(out_ids[0].tolist()))
```

**`compare_layers()`** manually steps through the model block by block to
capture intermediate representations:

```python
h = model.embedding(x)
h = model.pos_enc(h)
for block in model.blocks:
    h, _ = block(h)
    layer_outputs.append(h.clone())
```

This is not possible by just calling `model.forward()` — we need to
intercept the hidden state after each block.

---

## 11. Full data flow

Tracing one training step end-to-end:

```
corpus.txt
  "I like cats"
        │
        ▼  tokenizer.py
  Encoded: [1, 5, 11, 4, 2]    (1=BOS, 5=I, 11=like, 4=cats, 2=EOS)
  input:   [1, 5, 11,  4]
  target:  [5, 11,  4,  2]
        │
        ▼  model.forward(input)
  embedding:  (1, 4, 32)       each index → 32-dim vector
  + PE:       (1, 4, 32)       position 0,1,2,3 vectors added
        │
        ▼  TransformerBlock 1
  norm1 → MultiHeadAttention (causal mask applied)
        → residual add
  norm2 → FeedForward (GELU)
        → residual add
  output: (1, 4, 32)           context-aware, position-aware
        │
        ▼  TransformerBlock 2
  same structure, operating on block 1's output
  output: (1, 4, 32)           further refined
        │
        ▼  final LayerNorm
        ▼  Linear head  →  (1, 4, vocab_size)  logits
        │
        ▼  CrossEntropyLoss( logits, target )
  At position 0: model sees <BOS>, predicts "I" → loss
  At position 1: model sees <BOS> I, predicts "like" → loss
  At position 2: model sees <BOS> I like, predicts "cats" → loss
  At position 3: model sees <BOS> I like cats, predicts <EOS> → loss
  Mean over all 4 positions = batch loss
        │
        ▼  backward + clip + Adam step
  All weights updated (embedding, PE is fixed, attention proj, FF, norms, head)
```

---

## 12. How this maps onto GPT-2

| Component | mini-transformer | GPT-2 (small) |
|---|---|---|
| Vocabulary | 21 tokens | 50,257 (BPE) |
| `emb_dim` | 32 | 768 |
| `n_layers` | 2 | 12 |
| `n_heads` | 2 | 12 |
| `ff_dim` | 64 | 3,072 |
| Parameters | ~40K | 117M |
| Positional encoding | sinusoidal (fixed) | learned |
| Activation | GELU | GELU |
| Weight tying | ✓ | ✓ |
| Pre-norm | ✓ | ✓ |
| Training objective | next-token | next-token |
| Generation | greedy / temp / top-k | same |

The architecture is identical in structure. GPT-2 uses **learned** positional
embeddings instead of sinusoidal (a simple `nn.Embedding(max_len, emb_dim)`)
and trains on 40GB of text instead of 10 sentences. Everything else —
the block layout, weight tying, pre-norm, causal mask, GELU, generation loop
— is the same code, just scaled up.

`mini-gpt` will add: a larger corpus, learned positional embeddings,
cosine LR decay, and a proper training/validation split with perplexity
tracked on held-out data.

---

*Next: `mini-gpt` — scale to real text, add evaluation, and build a complete language model.*
