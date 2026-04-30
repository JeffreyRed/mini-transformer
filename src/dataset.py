"""
dataset.py — Causal language modelling dataset.

Same next-token prediction objective as mini-self-attention, but now:
  - sentences are wrapped with BOS and EOS tokens
  - a DataLoader-compatible Dataset class is used
  - collate_fn handles variable-length padding correctly

For each sentence  [BOS, w0, w1, w2, EOS]:
    input:  [BOS, w0, w1, w2]        (all tokens except last)
    target: [w0,  w1, w2,  EOS]      (all tokens except first, shifted by 1)

The model sees BOS and must predict w0.
Then it sees BOS,w0 and must predict w1.
...and so on until it predicts EOS.
This is exactly the GPT training objective.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class CausalDataset(Dataset):
    """
    Wraps BOS/EOS-encoded sentences as (input, target) pairs for
    causal language modelling.

    Args:
        encoded_sentences (List[List[int]]): from Tokenizer.encode_all()
        min_len (int): skip sentences shorter than this after encoding
    """

    def __init__(
        self,
        encoded_sentences: List[List[int]],
        min_len: int = 3,
    ) -> None:
        self.pairs: List[Tuple[List[int], List[int]]] = []
        for seq in encoded_sentences:
            if len(seq) < min_len + 1:
                continue
            self.pairs.append((seq[:-1], seq[1:]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.pairs[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )

    def __repr__(self) -> str:
        return f"CausalDataset(sentences={len(self.pairs)})"


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads variable-length sequences to the longest in the batch.

    Returns:
        src: (batch, max_len)  — input sequences
        tgt: (batch, max_len)  — target sequences (shifted by 1)
    """
    srcs, tgts = zip(*batch)
    max_len    = max(s.size(0) for s in srcs)

    src_pad = torch.full((len(srcs), max_len), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_len), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :s.size(0)] = s
        tgt_pad[i, :t.size(0)] = t

    return src_pad, tgt_pad


def make_collate(pad_idx: int):
    """Returns a collate_fn bound to a specific pad_idx."""
    def _collate(batch):
        return collate_fn(batch, pad_idx=pad_idx)
    return _collate
