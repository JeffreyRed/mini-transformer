"""
train.py — Training loop for the mini-transformer.

Same five-step PyTorch pattern as every previous project:
    zero_grad → forward → loss → backward → step

New additions vs mini-self-attention:
  - Learning rate warmup: the LR rises linearly for the first `warmup_steps`
    then holds steady.  Critical for transformers — jumping straight to a
    high LR causes instability in the early layers.
  - Gradient clipping: clips all gradients to max_norm=1.0 before the step.
    This prevents rare but catastrophic gradient explosions.
  - Perplexity reported alongside loss.  Perplexity = exp(loss) and is the
    standard metric for language models.  Perplexity of N means the model is
    roughly as confused as if it had to choose uniformly among N words.
    Lower is better; ideal = vocab_size at random init, ~1 if perfectly fit.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model   import MiniTransformer
from src.dataset import CausalDataset, make_collate


def train(
    model       : MiniTransformer,
    dataset     : CausalDataset,
    pad_idx     : int,
    epochs      : int   = 300,
    lr          : float = 3e-3,
    batch_size  : int   = 4,
    warmup_steps: int   = 50,
    verbose     : bool  = True,
    snapshots   : list  = None,
    snapshot_every: int = 10,
) -> list:
    """
    Trains MiniTransformer on causal language modelling.

    Args:
        model:          MiniTransformer instance
        dataset:        CausalDataset
        pad_idx:        padding index — excluded from loss
        epochs:         training epochs
        lr:             peak Adam learning rate
        batch_size:     sequences per gradient step
        warmup_steps:   linear LR warmup duration (in optimizer steps)
        verbose:        print progress every 10 epochs
        snapshots:      if provided, (epoch, attn_weights_layer0) tuples
                        appended every snapshot_every epochs
        snapshot_every: snapshot frequency in epochs

    Returns:
        List of (loss, perplexity) tuples per epoch.
    """
    loader    = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=make_collate(pad_idx),
    )
    loss_fn   = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    step      = 0
    history   = []   # list of (loss, perplexity)

    def get_lr(current_step: int) -> float:
        """Linear warmup then constant."""
        if current_step < warmup_steps:
            return lr * current_step / max(warmup_steps, 1)
        return lr

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in loader:
            step += 1

            # Adjust LR for warmup
            for pg in optimizer.param_groups:
                pg["lr"] = get_lr(step)

            optimizer.zero_grad()

            logits, _ = model(src)                    # (B, T, vocab)
            loss = loss_fn(logits.transpose(1, 2), tgt)  # (B, vocab, T) vs (B, T)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        mean_loss   = total_loss / len(loader)
        perplexity  = math.exp(min(mean_loss, 20))   # cap to avoid overflow
        history.append((mean_loss, perplexity))

        # Snapshot for animation
        if snapshots is not None and (epoch % snapshot_every == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                src0, _ = next(iter(loader))
                _, all_w = model(src0[:1])
                snapshots.append((epoch, [w.detach().clone() for w in all_w]))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"Loss: {mean_loss:.4f}  "
                f"Perplexity: {perplexity:.2f}  "
                f"LR: {get_lr(step):.5f}"
            )

    return history
