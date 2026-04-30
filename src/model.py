"""
model.py — Mini causal transformer (GPT-style decoder).

Architecture
────────────
    token indices  (batch, seq_len)
          │
          ▼
    nn.Embedding                       token embeddings
          │
          ▼
    PositionalEncoding                 adds position signal
          │
          ▼
    ┌─────────────────────────────┐
    │  TransformerBlock  × n_layers│   repeated N times
    │                             │
    │  ┌─ MultiHeadAttention ─┐   │
    │  │  causal mask applied │   │   each position sees only past tokens
    │  └──────────────────────┘   │
    │    + residual + LayerNorm   │
    │                             │
    │  ┌─ FeedForward ─────── ┐   │
    │  │  Linear→GELU→Linear  │   │   GELU instead of ReLU (like GPT-2)
    │  └──────────────────────┘   │
    │    + residual + LayerNorm   │
    └─────────────────────────────┘
          │
          ▼
    LayerNorm                          final normalisation
          │
          ▼
    nn.Linear → vocab logits           (emb_dim → vocab_size)

Why GELU instead of ReLU?
    GELU (Gaussian Error Linear Unit) is smoother than ReLU — it doesn't
    hard-zero negative inputs but smoothly gates them.  GPT-2 and most
    modern language models use GELU.  The math:
        GELU(x) ≈ x · σ(1.702 · x)

Why n_layers=2 for this corpus?
    The corpus has 10 sentences / 18 tokens.  More layers overfit immediately
    and make attention maps harder to read.  2 layers is the minimal
    configuration that shows the "refining" behaviour — layer 2's attention
    patterns differ visibly from layer 1's because it operates on vectors
    already partially processed by layer 1.

Causal mask
    A (seq_len, seq_len) upper-triangular boolean matrix where True = masked.
    Position i can attend to positions 0..i but not i+1..T.
    This is what makes the model a *language model* rather than an encoder.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from src.positional_encoding import PositionalEncoding
from src.attention           import MultiHeadAttention


# ── FeedForward ───────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise feed-forward block with GELU activation.

    Args:
        emb_dim (int): input/output dimension
        ff_dim  (int): inner hidden dimension (default 4 × emb_dim)
    """

    def __init__(self, emb_dim: int, ff_dim: int = None) -> None:
        super().__init__()
        ff_dim = ff_dim or 4 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Single transformer block ──────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One causal transformer decoder block:
        CausalSelfAttention → residual + LayerNorm
        FeedForward         → residual + LayerNorm

    Uses pre-norm layout (LayerNorm before the sublayer) which is more
    stable to train than the original post-norm from Vaswani et al.
    GPT-2 and most modern models use pre-norm.

    Args:
        emb_dim (int):   model dimension
        n_heads (int):   attention heads
        ff_dim  (int):   feedforward inner dimension
        dropout (float): dropout rate
    """

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        ff_dim: int  = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn  = MultiHeadAttention(emb_dim, n_heads)
        self.ff    = FeedForward(emb_dim, ff_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (batch, seq_len, emb_dim)
            mask: (batch, 1, seq_len, seq_len) causal mask

        Returns:
            out:     (batch, seq_len, emb_dim)
            weights: (batch, n_heads, seq_len, seq_len)
        """
        # Pre-norm self-attention + residual
        attn_out, weights = self.attn(self.norm1(x), mask)
        x = x + self.drop(attn_out)

        # Pre-norm feedforward + residual
        x = x + self.drop(self.ff(self.norm2(x)))

        return x, weights


# ── Full transformer ──────────────────────────────────────────────────────────

class MiniTransformer(nn.Module):
    """
    Minimal GPT-style causal language model.

    Args:
        vocab_size  (int):   vocabulary size (including special tokens)
        emb_dim     (int):   embedding / model dimension
        n_heads     (int):   attention heads per block
        n_layers    (int):   number of stacked TransformerBlocks
        ff_dim      (int):   feedforward inner dimension
        max_len     (int):   maximum sequence length
        dropout     (float): dropout probability
        pad_idx     (int):   padding token index (excluded from loss)
    """

    def __init__(
        self,
        vocab_size : int,
        emb_dim    : int,
        n_heads    : int,
        n_layers   : int   = 2,
        ff_dim     : int   = None,
        max_len    : int   = 64,
        dropout    : float = 0.1,
        pad_idx    : int   = 0,
    ) -> None:
        super().__init__()
        self.pad_idx  = pad_idx
        self.n_layers = n_layers
        self.emb_dim  = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(emb_dim, max_len, dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(emb_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights.
        # Standard in language models — reduces parameters and improves quality.
        self.head.weight = self.embedding.weight

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialises weights following GPT-2's scheme."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Builds the upper-triangular causal mask.

        Returns: (1, 1, seq_len, seq_len)  — True = must NOT attend
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len) — token indices

        Returns:
            logits:       (batch, seq_len, vocab_size)
            all_weights:  list of length n_layers, each (batch, n_heads, T, T)
        """
        B, T    = x.shape
        mask    = self._causal_mask(T, x.device)

        h           = self.embedding(x)     # (B, T, D)
        h           = self.pos_enc(h)       # + positional signal

        all_weights = []
        for block in self.blocks:
            h, w = block(h, mask)
            all_weights.append(w)

        h      = self.norm(h)
        logits = self.head(h)               # (B, T, vocab_size)

        return logits, all_weights

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids : torch.Tensor,
        max_new_tokens: int = 10,
        temperature   : float = 1.0,
        top_k         : int   = 0,
        greedy        : bool  = False,
    ) -> torch.Tensor:
        """
        Autoregressively generates tokens after a prompt.

        This is the key method that mini-gpt will expose as its main interface.
        It is included here so you can see generation working at this stage.

        Args:
            prompt_ids    : (1, prompt_len) — seed token indices
            max_new_tokens: how many tokens to generate
            temperature   : >1 = more random, <1 = more focused, 1 = neutral
            top_k         : if >0 sample only from the top-k most likely tokens
            greedy        : if True always pick argmax (ignores temperature/top_k)

        Returns:
            (1, prompt_len + max_new_tokens) — full sequence including prompt
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            logits, _ = self(ids)             # (1, T, vocab_size)
            next_logits = logits[:, -1, :]    # only the last position

            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(temperature, 1e-8)
                if top_k > 0:
                    # Zero out everything outside the top-k
                    vals, _ = torch.topk(next_logits, top_k)
                    threshold = vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float("-inf")
                probs   = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)

            # Stop at EOS if the tokenizer has one
            if hasattr(self, "eos_idx") and next_id.item() == self.eos_idx:
                break

        return ids

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        v, d   = self.embedding.weight.shape
        return (
            f"MiniTransformer(\n"
            f"  vocab={v}, emb_dim={d}, n_layers={self.n_layers},\n"
            f"  {self.blocks[0].attn},\n"
            f"  ff_dim={self.blocks[0].ff.net[0].out_features},\n"
            f"  parameters={params:,}\n"
            f")"
        )
