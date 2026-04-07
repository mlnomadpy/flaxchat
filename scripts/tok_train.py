"""
Train a BPE tokenizer on the dataset.

Usage:
    python -m scripts.tok_train --vocab-size=32768
"""

import os
import argparse

import pyarrow.parquet as pq

from flaxchat.common import get_base_dir, print0
from flaxchat.dataset import list_parquet_files, download_shards
from flaxchat.tokenizer import RustBPETokenizer, HuggingFaceTokenizer

parser = argparse.ArgumentParser(description="Train BPE tokenizer")
parser.add_argument("--vocab-size", type=int, default=32768, help="vocabulary size")
parser.add_argument("--num-shards", type=int, default=8, help="number of data shards to train on")
parser.add_argument("--backend", type=str, default="rustbpe", choices=["rustbpe", "huggingface"])
args = parser.parse_args()


def text_iterator(parquet_paths, max_chars=2_000_000_000):
    """Iterate over text from parquet files up to max_chars."""
    total_chars = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            for text in texts:
                yield text
                total_chars += len(text)
                if total_chars >= max_chars:
                    return


# Download data if needed
print0(f"Downloading {args.num_shards} data shards for tokenizer training...")
download_shards(0, args.num_shards)

parquet_paths = list_parquet_files()[:args.num_shards]
print0(f"Training tokenizer on {len(parquet_paths)} shards")

# Train
if args.backend == "rustbpe":
    tokenizer = RustBPETokenizer.train_from_iterator(
        text_iterator(parquet_paths), args.vocab_size
    )
else:
    tokenizer = HuggingFaceTokenizer.train_from_iterator(
        text_iterator(parquet_paths), args.vocab_size
    )

# Save
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# Verify
print0(f"Tokenizer trained! Vocab size: {tokenizer.get_vocab_size()}")
print0(f"Special tokens: {tokenizer.get_special_tokens()}")

# Quick compression test
test_text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(test_text)
print0(f"Test: '{test_text}' -> {len(tokens)} tokens (compression: {len(test_text)/len(tokens):.1f}x)")
