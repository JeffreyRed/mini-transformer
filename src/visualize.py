"""
visualize.py — Attention heatmaps, positional encoding grid,
               per-layer comparison, loss/perplexity curves.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

PALETTE = {
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "accent":    "#58a6ff",
    "highlight": "#f78166",
    "muted":     "#8b949e",
    "green":     "#3fb950",
    "purple":    "#bc8cff",
}


# ── Positional encoding visualisation ────────────────────────────────────────

def plot_positional_encoding(emb_dim: int, max_len: int = 20,
                              save_path: str = None) -> None:
    """
    Plots the sinusoidal positional encoding matrix as a heatmap.

    Rows = positions, Columns = embedding dimensions.
    Each row is the vector that gets added to a token at that position.
    You can see the alternating sin/cos pattern across dimensions.
    """
    pe  = np.zeros((max_len, emb_dim))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    im = ax.imshow(pe, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    ax.set_xlabel("Embedding dimension", color=PALETTE["muted"])
    ax.set_ylabel("Position in sequence", color=PALETTE["muted"])
    ax.set_title(
        "Sinusoidal Positional Encoding  ·  each row added to token at that position",
        color=PALETTE["text"], fontsize=11, pad=10,
    )
    ax.tick_params(colors=PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"PE plot saved → {save_path}")
    plt.show()


# ── Per-layer attention heatmaps ──────────────────────────────────────────────

def plot_all_layers(
    all_weights : List["torch.Tensor"],
    words       : List[str],
    sentence_label: str = "",
    save_path   : str = None,
) -> None:
    """
    Plots all heads of all layers in one figure.

    Layout: rows = layers, columns = heads.
    This makes it easy to see how different layers attend differently.

    Args:
        all_weights: list of (n_heads, T, T) tensors, one per layer
        words:       token strings for the sentence
        sentence_label: printed in the title
        save_path:   optional save path
    """
    n_layers = len(all_weights)
    n_heads  = all_weights[0].shape[0]

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(4 * n_heads, 4 * n_layers),
    )
    fig.patch.set_facecolor(PALETTE["bg"])

    # Ensure 2-D array of axes
    if n_layers == 1 and n_heads == 1:
        axes = [[axes]]
    elif n_layers == 1:
        axes = [axes]
    elif n_heads == 1:
        axes = [[ax] for ax in axes]

    for layer_idx, weights in enumerate(all_weights):
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            w  = weights[head_idx].numpy()

            ax.set_facecolor(PALETTE["bg"])
            ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")

            ax.set_xticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha="right",
                               fontsize=8, color=PALETTE["text"],
                               fontfamily="monospace")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=8, color=PALETTE["text"],
                               fontfamily="monospace")

            for i in range(len(words)):
                for j in range(len(words)):
                    val   = w[i, j]
                    color = "white" if val > 0.5 else PALETTE["muted"]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["grid"])

            ax.set_title(f"Layer {layer_idx + 1}  Head {head_idx}",
                         color=PALETTE["text"], fontsize=9, pad=6)

    fig.suptitle(
        f"All Attention Weights  ·  \"{sentence_label}\"",
        color=PALETTE["text"], fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"All-layers plot saved → {save_path}")
    plt.show()


# ── Loss + perplexity curves ──────────────────────────────────────────────────

def plot_loss(
    history   : list,
    save_path : str = None,
) -> None:
    """
    Plots training loss and perplexity on the same figure (twin axes).

    Args:
        history: list of (loss, perplexity) tuples from train()
    """
    losses = [h[0] for h in history]
    perps  = [h[1] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax1.set_facecolor(PALETTE["bg"])

    ax1.plot(losses, color=PALETTE["highlight"], linewidth=2, label="Loss")
    ax1.set_xlabel("Epoch", color=PALETTE["muted"])
    ax1.set_ylabel("Cross-Entropy Loss", color=PALETTE["highlight"])
    ax1.tick_params(axis="y", colors=PALETTE["highlight"])
    ax1.tick_params(axis="x", colors=PALETTE["muted"])
    ax1.grid(color=PALETTE["grid"], linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    ax2 = ax1.twinx()
    ax2.set_facecolor(PALETTE["bg"])
    ax2.plot(perps, color=PALETTE["accent"], linewidth=2,
             linestyle="--", label="Perplexity")
    ax2.set_ylabel("Perplexity", color=PALETTE["accent"])
    ax2.tick_params(axis="y", colors=PALETTE["accent"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=PALETTE["grid"], labelcolor=PALETTE["text"],
               fontsize=9, loc="upper right")

    ax1.set_title("Training  ·  Loss & Perplexity",
                  color=PALETTE["text"], fontsize=12, pad=10, loc="left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Loss curve saved → {save_path}")
    plt.show()


# ── Animated attention evolution ──────────────────────────────────────────────

def animate_attention(
    snapshots  : list,
    words      : List[str],
    layer      : int = 0,
    head       : int = 0,
    save_path  : str = None,
    interval   : int = 250,
) -> None:
    """
    Animates how one head's attention pattern evolves during training.

    Args:
        snapshots:  list of (epoch, [weights_layer0, weights_layer1, ...])
                    from train()
        words:      token strings for the snapshot sentence
        layer:      which layer to animate
        head:       which head to animate
        save_path:  saves as .gif if provided
        interval:   ms between frames
    """
    if not snapshots:
        print("No snapshots available.")
        return

    seq_len = len(words)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    im = ax.imshow(np.zeros((seq_len, seq_len)), cmap="Blues",
                   vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(words, rotation=45, ha="right",
                       fontsize=9, color=PALETTE["text"],
                       fontfamily="monospace")
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(words, fontsize=9, color=PALETTE["text"],
                       fontfamily="monospace")
    ax.set_xlabel("Keys  (attended to)", color=PALETTE["muted"], fontsize=8)
    ax.set_ylabel("Queries", color=PALETTE["muted"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    title = ax.set_title("", color=PALETTE["text"], fontsize=10, pad=8)

    def update(frame_idx):
        epoch, all_w = snapshots[frame_idx]
        w = all_w[layer]          # (batch=1, n_heads, T, T) or (n_heads, T, T)
        if w.dim() == 4:
            w = w[0]              # drop batch dim
        data = w[head, :seq_len, :seq_len].numpy()
        im.set_data(data)
        title.set_text(
            f"Attention  layer {layer + 1}  head {head}  ·  "
            f"epoch {epoch}  [{frame_idx + 1}/{len(snapshots)}]"
        )
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots),
        interval=interval, blit=False,
    )

    if save_path:
        anim.save(save_path, writer="pillow", dpi=100)
        print(f"Attention animation saved → {save_path}")

    plt.tight_layout()
    plt.show()
