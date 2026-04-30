"""
attention.py — Scaled dot-product and multi-head self-attention.

Identical to mini-self-attention/src/attention.py.
Reproduced here so mini-transformer is a fully self-contained repo.

The causal mask (preventing position i from attending to positions > i)
is built in model.py and passed in as the `mask` argument.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Args:
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_v)
        mask: (batch, 1, seq_len, seq_len) — True = masked out

    Returns:
        output:  (batch, heads, seq_len, d_v)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    # Replace NaN from all-masked rows (e.g. padding) with 0
    weights = torch.nan_to_num(weights, nan=0.0)

    return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention  (Vaswani et al. 2017).

    Args:
        emb_dim (int): model dimension  (must be divisible by n_heads)
        n_heads (int): number of parallel attention heads
    """

    def __init__(self, emb_dim: int, n_heads: int) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0, (
            f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads

        self.W_Q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (batch, seq_len, emb_dim)
            mask: (batch, 1, seq_len, seq_len) optional

        Returns:
            output:  (batch, seq_len, emb_dim)
            weights: (batch, n_heads, seq_len, seq_len)
        """
        B, T, _ = x.shape

        Q = self._split(self.W_Q(x), B, T)
        K = self._split(self.W_K(x), B, T)
        V = self._split(self.W_V(x), B, T)

        out, weights = scaled_dot_product_attention(Q, K, V, mask)

        merged = out.transpose(1, 2).contiguous().view(B, T, self.emb_dim)
        return self.W_O(merged), weights

    def _split(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention("
            f"emb_dim={self.emb_dim}, "
            f"n_heads={self.n_heads}, "
            f"head_dim={self.head_dim})"
        )
