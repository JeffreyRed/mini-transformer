"""
utils.py — Generation helpers and attention inspection.

The big addition over mini-self-attention is the generate() wrapper,
which gives a clean text-in / text-out interface around model.generate().
This is the prototype for what mini-gpt will expose as its main API.
"""

import torch
from typing import List, Tuple, Optional

from src.model     import MiniTransformer
from src.tokenizer import Tokenizer


# ── Text generation ───────────────────────────────────────────────────────────

def generate_text(
    model       : MiniTransformer,
    tokenizer   : Tokenizer,
    prompt      : str,
    max_new     : int   = 10,
    temperature : float = 1.0,
    top_k       : int   = 0,
    greedy      : bool  = False,
) -> str:
    """
    Generates text from a string prompt.

    Encodes the prompt → feeds to model.generate() → decodes the result.

    Args:
        model:       trained MiniTransformer
        tokenizer:   Tokenizer used during training
        prompt:      seed string  (words must be in vocabulary)
        max_new:     maximum tokens to generate
        temperature: sampling temperature (1.0 = neutral)
        top_k:       if >0, restrict sampling to top-k tokens
        greedy:      always pick the argmax token

    Returns:
        Generated string (prompt + new tokens, special tokens stripped).
    """
    words      = prompt.strip().split()
    prompt_ids = tokenizer.encode(words, add_special=True)
    src        = torch.tensor([prompt_ids], dtype=torch.long)

    out_ids = model.generate(
        src,
        max_new_tokens = max_new,
        temperature    = temperature,
        top_k          = top_k,
        greedy         = greedy,
    )

    return " ".join(tokenizer.decode(out_ids[0].tolist(), strip_special=True))


def interactive_generate(
    model     : MiniTransformer,
    tokenizer : Tokenizer,
) -> None:
    """
    Interactive text generation loop.
    Type a seed word or phrase → model continues it.
    Type 'quit' to exit.
    """
    vocab          = list(tokenizer.word2idx.keys())
    lower_to_vocab = {w.lower(): w for w in vocab}

    print("── Text generation ─────────────────────────────────")
    print(f"  Vocabulary: {[w for w in vocab if not w.startswith('<')]}")
    print("  Type a seed word or phrase.")
    print("  Flags: --greedy  --temp=0.8  --topk=5")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            raw = input("  Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        # Parse optional flags
        parts   = raw.split()
        flags   = {p for p in parts if p.startswith("--")}
        words   = [p for p in parts if not p.startswith("--")]
        greedy  = "--greedy" in flags
        temp    = 1.0
        top_k   = 0
        for f in flags:
            if f.startswith("--temp="):
                try:    temp  = float(f.split("=")[1])
                except: pass
            if f.startswith("--topk="):
                try:    top_k = int(f.split("=")[1])
                except: pass

        # Normalise case
        resolved = [lower_to_vocab.get(w.lower(), w) for w in words]
        unknown  = [w for w in resolved if w not in tokenizer.word2idx
                    and w not in tokenizer.SPECIAL]
        if unknown:
            print(f"  ✗ Unknown words: {unknown}. Vocabulary: "
                  f"{[w for w in vocab if not w.startswith('<')]}\n")
            continue

        result = generate_text(
            model, tokenizer, " ".join(resolved),
            max_new=8, temperature=temp, top_k=top_k, greedy=greedy,
        )
        print(f"\n  → {result}\n")

    print("────────────────────────────────────────────────────\n")


# ── Attention inspection ──────────────────────────────────────────────────────

def get_all_attention_weights(
    model    : MiniTransformer,
    sentence : List[int],
) -> List[torch.Tensor]:
    """
    Returns attention weights for every layer for a single encoded sentence.

    Args:
        model:    trained MiniTransformer
        sentence: list of token indices

    Returns:
        List of (n_heads, seq_len, seq_len) tensors, one per layer.
    """
    model.eval()
    x = torch.tensor(sentence).unsqueeze(0)
    with torch.no_grad():
        _, all_w = model(x)
    return [w.squeeze(0) for w in all_w]


def print_attention_table(
    weights : torch.Tensor,
    words   : List[str],
    layer   : int = 0,
    head    : int = 0,
) -> None:
    """
    Prints a human-readable attention weight table.

    Args:
        weights: (n_heads, seq_len, seq_len)
        words:   token strings for this sentence
        layer:   layer index (for the title only)
        head:    which head to display
    """
    w       = weights[head].numpy()
    col_w   = max(len(wd) for wd in words) + 2

    print(f"\n  Attention — layer {layer}  head {head}")
    print(f"  {'':>{col_w}}", end="")
    for word in words:
        print(f"  {word:>{col_w}}", end="")
    print()

    for i, row_word in enumerate(words):
        print(f"  {row_word:>{col_w}}", end="")
        for j in range(len(words)):
            val = w[i, j]
            bar = "█" * int(val * 8)
            print(f"  {val:.3f}{bar:>3}", end="")
        print()
    print()


def compare_layers(
    model    : MiniTransformer,
    sentence : List[int],
    words    : List[str],
) -> None:
    """
    Shows how the representation of each word changes from layer to layer
    by measuring cosine similarity between layer outputs.

    This demonstrates the 'refinement' behaviour of stacked blocks:
    each layer builds on the previous one's representations.
    """
    import torch.nn.functional as F

    model.eval()
    x = torch.tensor(sentence).unsqueeze(0)

    layer_outputs = []
    with torch.no_grad():
        h = model.embedding(x)
        h = model.pos_enc(h)
        for block in model.blocks:
            h, _ = block(h)
            layer_outputs.append(h.squeeze(0).clone())   # (T, D)

    print("\n  Layer-by-layer cosine similarity (how much each layer changed each word)")
    print("  Low = big change, High = small change\n")

    for word_idx, word in enumerate(words):
        sims = []
        for l in range(1, len(layer_outputs)):
            sim = F.cosine_similarity(
                layer_outputs[l-1][word_idx].unsqueeze(0),
                layer_outputs[l][word_idx].unsqueeze(0),
            ).item()
            sims.append(f"L{l}→L{l+1}: {sim:.3f}")
        print(f"    {word:<12}  {',  '.join(sims)}")
    print()
