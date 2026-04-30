"""mini-transformer — source package."""
from src.tokenizer           import Tokenizer
from src.positional_encoding import PositionalEncoding
from src.attention           import scaled_dot_product_attention, MultiHeadAttention
from src.model               import FeedForward, TransformerBlock, MiniTransformer
from src.dataset             import CausalDataset, collate_fn, make_collate
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

__all__ = [
    "Tokenizer",
    "PositionalEncoding",
    "scaled_dot_product_attention", "MultiHeadAttention",
    "FeedForward", "TransformerBlock", "MiniTransformer",
    "CausalDataset", "collate_fn", "make_collate",
    "train",
    "generate_text", "interactive_generate",
    "get_all_attention_weights", "print_attention_table", "compare_layers",
    "plot_positional_encoding", "plot_all_layers", "plot_loss",
    "animate_attention",
]
