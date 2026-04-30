"""
main.py — End-to-end pipeline for mini-transformer.

Usage:
    python main.py

Pipeline:
    1.  Load & tokenise corpus
    2.  Show BOS/EOS-wrapped training pairs
    3.  Plot positional encoding grid  (new vs mini-self-attention)
    4.  Build model
    5.  Train with LR warmup + gradient clipping
    6.  Inspect attention weights per layer
    7.  Show layer-by-layer representation change
    8.  Interactive text generation
    9.  Save model
    10. Plots: all-layers attention, loss + perplexity, animation
"""

import torch
from pathlib import Path

from src.tokenizer           import Tokenizer
from src.dataset             import CausalDataset
from src.model               import MiniTransformer
from src.train               import train
from src.utils               import (
    generate_text,
    interactive_generate,
    get_all_attention_weights,
    print_attention_table,
    compare_layers,
)
from src.visualize           import (
    plot_positional_encoding,
    plot_all_layers,
    plot_loss,
    animate_attention,
)

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH      = "data/corpus.txt"
EMB_DIM          = 32      # larger than mini-self-attention — gives cleaner PE
N_HEADS          = 2       # EMB_DIM must be divisible by N_HEADS
N_LAYERS         = 2       # two stacked transformer blocks
FF_DIM           = 64      # feedforward inner dimension  (2 × EMB_DIM)
EPOCHS           = 300
LR               = 3e-3
BATCH_SIZE       = 4
WARMUP_STEPS     = 50
OUTPUTS_DIR      = Path("outputs")
INSPECT_SENTENCE = "I like cats"
# ──────────────────────────────────────────────────────────────────────────────


def show_dataset_samples(tok: Tokenizer, dataset: CausalDataset) -> None:
    """Prints the first few (input, target) pairs to make the objective concrete."""
    print("── Training pairs (BOS/EOS-wrapped) ────────────────")
    print("  Objective: given input tokens, predict the next token at each position")
    print("  BOS = beginning-of-sentence,  EOS = end-of-sentence\n")
    for src_t, tgt_t in list(dataset)[:4]:
        src_w = tok.decode(src_t.tolist(), strip_special=False)
        tgt_w = tok.decode(tgt_t.tolist(), strip_special=False)
        print(f"  input : {src_w}")
        print(f"  target: {tgt_w}")
        print()


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── 1. Tokenise ───────────────────────────────────────────────────────────
    print("\n── Tokeniser ───────────────────────────────────────")
    tok = Tokenizer(CORPUS_PATH)
    print(tok)
    print(f"  Special tokens: PAD={tok.pad_idx}  BOS={tok.bos_idx}  EOS={tok.eos_idx}")
    print(f"  Vocabulary: {[w for w in tok.word2idx if not w.startswith('<')]}\n")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    encoded = tok.encode_all()
    dataset = CausalDataset(encoded)
    print(dataset)
    show_dataset_samples(tok, dataset)

    # ── 3. Positional encoding preview ───────────────────────────────────────
    print("── Positional encoding ─────────────────────────────")
    print("  Sinusoidal PE adds a unique position vector to each token.")
    print("  Plotting the encoding matrix...\n")
    plot_positional_encoding(
        emb_dim   = EMB_DIM,
        max_len   = 20,
        save_path = str(OUTPUTS_DIR / "positional_encoding.png"),
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print("── Model ───────────────────────────────────────────")
    model = MiniTransformer(
        vocab_size = tok.vocab_size,
        emb_dim    = EMB_DIM,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        ff_dim     = FF_DIM,
        pad_idx    = tok.pad_idx,
    )
    print(model, "\n")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("── Training ────────────────────────────────────────")
    snapshot_every = max(1, EPOCHS // 30)
    snapshots: list = []

    history = train(
        model, dataset,
        pad_idx       = tok.pad_idx,
        epochs        = EPOCHS,
        lr            = LR,
        batch_size    = BATCH_SIZE,
        warmup_steps  = WARMUP_STEPS,
        snapshots     = snapshots,
        snapshot_every= snapshot_every,
    )
    final_loss, final_perp = history[-1]
    print(f"\n  Final loss: {final_loss:.4f}   Perplexity: {final_perp:.2f}\n")

    # ── 6. Attention inspection ───────────────────────────────────────────────
    inspect_words   = tok.decode(
        tok.encode(INSPECT_SENTENCE.split(), add_special=True),
        strip_special=False,
    )
    inspect_encoded = tok.encode(INSPECT_SENTENCE.split(), add_special=True)

    print(f"── Attention inspection: \"{INSPECT_SENTENCE}\" ─────────")
    all_weights = get_all_attention_weights(model, inspect_encoded)

    for layer_idx, w in enumerate(all_weights):
        for h in range(N_HEADS):
            print_attention_table(w, inspect_words, layer=layer_idx, head=h)

    # ── 7. Layer comparison ───────────────────────────────────────────────────
    compare_layers(model, inspect_encoded, inspect_words)

    # ── 8. Interactive generation ─────────────────────────────────────────────
    interactive_generate(model, tok)

    # ── 9. Save ───────────────────────────────────────────────────────────────
    save_path = OUTPUTS_DIR / "transformer.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}\n")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    plot_all_layers(
        all_weights, inspect_words,
        sentence_label = INSPECT_SENTENCE,
        save_path      = str(OUTPUTS_DIR / "attention_all_layers.png"),
    )

    plot_loss(
        history,
        save_path = str(OUTPUTS_DIR / "loss_perplexity.png"),
    )

    if snapshots:
        animate_attention(
            snapshots, inspect_words,
            layer     = 0,
            head      = 0,
            save_path = str(OUTPUTS_DIR / "attention_animation.gif"),
        )


if __name__ == "__main__":
    main()
