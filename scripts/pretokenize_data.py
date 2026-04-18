"""
Pre-tokenize all training data using Gemma 4 tokenizer (256K vocab).
Produces sharded .bin files (uint32 numpy) + metadata JSON for each source.

Designed to run on an n2-standard-32 GCE VM in europe-west4.

Usage:
    python pretokenize_data.py --output gs://flaxchat-data-eu/tokenized \
                               --workers 28 \
                               --tokenizer google/gemma-3-27b-it

Each shard = 1M tokens = 4 MB. Documents separated by EOS token.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="GCS path or local dir for tokenized shards")
parser.add_argument("--local-tmp", default="/tmp/tokenized", help="Local staging before GCS upload")
parser.add_argument("--tokenizer", default="google/gemma-3-27b-it", help="HF tokenizer name")
parser.add_argument("--workers", type=int, default=28)
parser.add_argument("--shard-tokens", type=int, default=1_048_576, help="Tokens per shard (~4MB)")
parser.add_argument("--sources", default="all", help="Comma-sep list or 'all'")
parser.add_argument("--max-tokens-per-source", type=int, default=0, help="Cap per source (0=use recipe default)")
args = parser.parse_args()

# ---- Data recipe (target tokens per source) ----
RECIPE = {
    "fineweb_edu":      {"tokens": 120_000_000_000, "dataset": "HuggingFaceFW/fineweb-edu", "subset": "sample-100BT", "split": "train", "text_key": "text"},
    "starcoder":        {"tokens":  40_000_000_000, "dataset": "bigcode/the-stack-v2-train-smol-ids", "subset": None, "split": "train", "text_key": "content"},
    "cosmopedia":       {"tokens":  30_000_000_000, "dataset": "HuggingFaceTB/cosmopedia-v2", "subset": None, "split": "train", "text_key": "text"},
    "wikipedia":        {"tokens":  15_000_000_000, "dataset": "wikimedia/wikipedia", "subset": "20231101.en", "split": "train", "text_key": "text"},
    "arxiv":            {"tokens":  15_000_000_000, "dataset": "togethercomputer/RedPajama-Data-V2", "subset": "default", "split": "train", "text_key": "raw_content"},
    "books":            {"tokens":  10_000_000_000, "dataset": "emozilla/pg19", "subset": None, "split": "train", "text_key": "text"},
    "openwebmath":      {"tokens":  10_000_000_000, "dataset": "open-web-math/open-web-math", "subset": None, "split": "train", "text_key": "text"},
    "stackexchange":    {"tokens":   6_000_000_000, "dataset": "HuggingFaceTB/smoltalk", "subset": None, "split": "train", "text_key": "content"},
}

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"Tokenizer: {args.tokenizer}, vocab_size={tok.vocab_size}")
    return tok


def tokenize_batch(texts, tokenizer):
    """Tokenize a list of texts, return flat uint32 array with EOS between docs."""
    eos = tokenizer.eos_token_id or 0
    all_ids = []
    for text in texts:
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            all_ids.extend(ids)
            all_ids.append(eos)
    return np.array(all_ids, dtype=np.uint32)


def process_source(source_name, source_cfg, tokenizer_name, local_dir, shard_tokens, max_tokens):
    """Process one data source: stream from HF, tokenize, write shards."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    eos = tok.eos_token_id or 0

    target_tokens = max_tokens if max_tokens > 0 else source_cfg["tokens"]
    ds_name = source_cfg["dataset"]
    subset = source_cfg.get("subset")
    split = source_cfg.get("split", "train")
    text_key = source_cfg.get("text_key", "text")

    out_dir = os.path.join(local_dir, source_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{source_name}] Loading {ds_name} (subset={subset}, split={split})")
    try:
        if subset:
            ds = load_dataset(ds_name, subset, split=split, streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"[{source_name}] FAILED to load: {e}")
        return source_name, 0, 0

    buffer = []
    buffer_tokens = 0
    shard_idx = 0
    total_tokens = 0
    total_docs = 0
    t0 = time.time()
    batch_size = 100

    batch_texts = []
    for example in ds:
        text = example.get(text_key, "")
        if not text or len(text.strip()) < 50:
            continue
        batch_texts.append(text[:50000])  # cap per-doc length
        total_docs += 1

        if len(batch_texts) >= batch_size:
            ids = tokenize_batch(batch_texts, tok)
            batch_texts = []
            buffer.append(ids)
            buffer_tokens += len(ids)

            # Write shard when buffer is full
            while buffer_tokens >= shard_tokens:
                combined = np.concatenate(buffer)
                shard_data = combined[:shard_tokens]
                remainder = combined[shard_tokens:]

                shard_path = os.path.join(out_dir, f"shard_{shard_idx:06d}.bin")
                shard_data.tofile(shard_path)
                shard_idx += 1
                total_tokens += len(shard_data)

                buffer = [remainder] if len(remainder) > 0 else []
                buffer_tokens = len(remainder) if len(remainder) > 0 else 0

            if total_tokens >= target_tokens:
                break

            if shard_idx % 100 == 0 and shard_idx > 0:
                elapsed = time.time() - t0
                rate = total_tokens / elapsed
                eta = (target_tokens - total_tokens) / rate if rate > 0 else 0
                print(f"[{source_name}] {total_tokens/1e9:.2f}B / {target_tokens/1e9:.0f}B tokens | "
                      f"{shard_idx} shards | {total_docs:,} docs | "
                      f"{rate/1e6:.1f}M tok/s | eta {eta/3600:.1f}h")

    # Flush remaining buffer
    if batch_texts:
        ids = tokenize_batch(batch_texts, tok)
        buffer.append(ids)
        buffer_tokens += len(ids)
    if buffer_tokens > 0:
        combined = np.concatenate(buffer)
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:06d}.bin")
        combined[:shard_tokens].tofile(shard_path)
        total_tokens += min(len(combined), shard_tokens)
        shard_idx += 1

    # Write metadata
    meta = {
        "source": source_name,
        "dataset": ds_name,
        "subset": subset,
        "n_shards": shard_idx,
        "n_tokens": int(total_tokens),
        "n_docs": total_docs,
        "shard_tokens": shard_tokens,
        "tokenizer": tokenizer_name,
        "vocab_size": tok.vocab_size,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"[{source_name}] DONE: {total_tokens/1e9:.2f}B tokens, {shard_idx} shards, "
          f"{total_docs:,} docs, {elapsed/3600:.1f}h")
    return source_name, total_tokens, shard_idx


def upload_to_gcs(local_dir, gcs_path):
    """Upload tokenized shards to GCS."""
    import subprocess
    print(f"Uploading {local_dir} → {gcs_path}")
    r = subprocess.run(
        f"gcloud storage cp -r {local_dir}/* {gcs_path}/",
        shell=True, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"Upload failed: {r.stderr[-500:]}")
    else:
        print(f"Upload complete: {gcs_path}")


def main():
    os.makedirs(args.local_tmp, exist_ok=True)

    # Verify tokenizer works
    tok = get_tokenizer()
    test = tok.encode("Hello world")
    print(f"Test encode: 'Hello world' → {test} (vocab={tok.vocab_size})")

    # Select sources
    if args.sources == "all":
        sources = list(RECIPE.keys())
    else:
        sources = [s.strip() for s in args.sources.split(",")]

    print(f"\nProcessing {len(sources)} sources: {sources}")
    total_target = sum(RECIPE[s]["tokens"] for s in sources if s in RECIPE)
    print(f"Total target: {total_target/1e9:.0f}B tokens\n")

    # Process sources sequentially (each source streams and is I/O bound)
    results = []
    for source_name in sources:
        if source_name not in RECIPE:
            print(f"Unknown source: {source_name}, skipping")
            continue
        cfg = RECIPE[source_name]
        max_tok = args.max_tokens_per_source if args.max_tokens_per_source > 0 else cfg["tokens"]
        name, n_tokens, n_shards = process_source(
            source_name, cfg, args.tokenizer, args.local_tmp,
            args.shard_tokens, max_tok,
        )
        results.append((name, n_tokens, n_shards))

        # Upload this source to GCS incrementally (don't wait for all to finish)
        if args.output.startswith("gs://"):
            upload_to_gcs(
                os.path.join(args.local_tmp, source_name),
                f"{args.output}/{source_name}",
            )
            # Free local disk after upload
            import shutil
            shutil.rmtree(os.path.join(args.local_tmp, source_name), ignore_errors=True)

    # Summary
    print("\n" + "=" * 60)
    print("TOKENIZATION SUMMARY")
    print("=" * 60)
    grand_total = 0
    grand_shards = 0
    for name, n_tokens, n_shards in results:
        print(f"  {name:<20} {n_tokens/1e9:>8.2f}B tokens  {n_shards:>6} shards")
        grand_total += n_tokens
        grand_shards += n_shards
    print(f"  {'TOTAL':<20} {grand_total/1e9:>8.2f}B tokens  {grand_shards:>6} shards")
    print(f"  Storage: ~{grand_total * 4 / 1e9:.0f} GB (uint32)")
    print(f"  GCS: {args.output}")

    # Write global metadata
    global_meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "shard_tokens": args.shard_tokens,
        "sources": {name: {"n_tokens": int(nt), "n_shards": ns} for name, nt, ns in results},
        "total_tokens": int(grand_total),
        "total_shards": grand_shards,
    }
    meta_path = os.path.join(args.local_tmp, "global_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(global_meta, f, indent=2)
    if args.output.startswith("gs://"):
        upload_to_gcs(meta_path, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
