"""
positional_encoding.py — Sinusoidal positional encoding.

─────────────────────────────────────────────────────────────────────────────
THE PROBLEM
─────────────────────────────────────────────────────────────────────────────
Self-attention computes relationships between all pairs of positions
simultaneously.  That makes it fast, but it has no built-in sense of order:
the input  ["cats", "like", "I"]  and  ["I", "like", "cats"]  produce the
same attention scores if the embedding vectors are the same.

We need to inject the position of each token into its representation before
attention runs.

─────────────────────────────────────────────────────────────────────────────
THE SOLUTION — fixed sinusoidal encoding  (Vaswani et al. 2017)
─────────────────────────────────────────────────────────────────────────────
For each position pos and each dimension i of the embedding:

    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

This produces a unique vector for every position.  The frequencies decrease
as the dimension index grows, like a binary counter in continuous space:
high-frequency dimensions encode fine-grained position;
low-frequency dimensions encode coarse position.

Key properties:
  • Fixed (not learned) — one less thing to train, and works on sequences
    longer than those seen during training.
  • Added to (not concatenated with) the embedding — same dimensionality
    is preserved throughout.
  • PE(pos + k) is a linear function of PE(pos) for any offset k — the
    model can learn relative positions from the encoding.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to an input embedding tensor.

    The encoding matrix is computed once and registered as a buffer
    (not a parameter — it is never updated by the optimizer).

    Args:
        emb_dim (int):   embedding dimensionality  (must be even)
        max_len (int):   maximum sequence length to pre-compute (default 512)
        dropout (float): applied after adding the encoding (default 0.1)
    """

    def __init__(
        self,
        emb_dim: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the (max_len, emb_dim) encoding matrix
        pe  = torch.zeros(max_len, emb_dim)                    # (T, D)
        pos = torch.arange(0, max_len).unsqueeze(1).float()    # (T, 1)
        div = torch.exp(
            torch.arange(0, emb_dim, 2).float()
            * -(math.log(10000.0) / emb_dim)
        )                                                       # (D/2,)

        pe[:, 0::2] = torch.sin(pos * div)   # even dimensions
        pe[:, 1::2] = torch.cos(pos * div)   # odd  dimensions

        # Register as buffer: saved in state_dict, not updated by optimizer
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to x.

        Args:
            x: (batch, seq_len, emb_dim)

        Returns:
            (batch, seq_len, emb_dim) — same shape, position-aware
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

    def __repr__(self) -> str:
        _, max_len, emb_dim = self.pe.shape
        return f"PositionalEncoding(emb_dim={emb_dim}, max_len={max_len})"
