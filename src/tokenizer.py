"""
tokenizer.py — Vocabulary builder and sentence encoder.

Identical to mini-self-attention's tokenizer so each repo is fully
self-contained and runnable without importing from sibling projects.
"""

from typing import List, Dict


class Tokenizer:
    """
    Builds a vocabulary from a plain-text corpus and encodes/decodes sentences.

    Special tokens:
        <PAD>  index 0  — used to pad shorter sequences in a batch
        <BOS>  index 1  — Beginning Of Sentence marker
        <EOS>  index 2  — End Of Sentence marker

    Having BOS/EOS is new here vs mini-self-attention:
    the causal language model needs to know where sentences start and end
    so it can learn to generate complete, bounded sequences.

    Args:
        path (str): path to a plain-text corpus (one sentence per line)
    """

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPECIAL   = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, path: str) -> None:
        with open(path) as f:
            self.sentences: List[List[str]] = [
                line.strip().split() for line in f if line.strip()
            ]
        self._build_vocab()

    def _build_vocab(self) -> None:
        words = sorted({w for s in self.sentences for w in s})
        all_tokens = self.SPECIAL + words
        self.word2idx: Dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.idx2word: Dict[int, str] = {i: t for t, i in self.word2idx.items()}
        self.vocab_size: int = len(self.word2idx)
        self.pad_idx = self.word2idx[self.PAD_TOKEN]
        self.bos_idx = self.word2idx[self.BOS_TOKEN]
        self.eos_idx = self.word2idx[self.EOS_TOKEN]

    # ------------------------------------------------------------------

    def encode(self, sentence: List[str], add_special: bool = True) -> List[int]:
        """Encodes a word list → index list, optionally wrapping with BOS/EOS."""
        ids = [self.word2idx[w] for w in sentence if w in self.word2idx]
        if add_special:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, indices: List[int], strip_special: bool = True) -> List[str]:
        """Decodes an index list → word list, optionally removing special tokens."""
        words = [self.idx2word.get(i, "?") for i in indices]
        if strip_special:
            words = [w for w in words if w not in self.SPECIAL]
        return words

    def encode_all(self) -> List[List[int]]:
        """Returns every corpus sentence as a BOS-wrapped index sequence."""
        return [self.encode(s) for s in self.sentences]

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"sentences={len(self.sentences)}, "
            f"special={self.SPECIAL})"
        )
