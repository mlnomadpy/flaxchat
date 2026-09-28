"""Build a pinned, multilingual Global Voices contrastive corpus.

The 28 English–other-language *train* shards are fetched at an immutable Hub
revision. Raw Parquet hashes, tokenizer hash, selected row coordinates, and
the preparation manifests form a replayable receipt. No test labels are used
for sampling. Pass both previous pilot train and dev files as repeated
``--heldout`` arguments when resuming its checkpoint. This prevents earlier
training text from appearing in the new dev split.

This is an aligned news-domain source, not a representative multilingual
retrieval corpus. Global Voices content requires CC-BY-3.0 attribution.
"""

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys
import urllib.request

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from prepare_encoder_contrastive import file_hash, prepare, verify


DATASET = "sentence-transformers/parallel-sentences-global-voices"
REVISION = "4cc20add371f246bb1559b543f8b0dea178a1803"
LICENSE = "CC-BY-3.0"
LOCK = Path(__file__).resolve().parents[1] / "configs/contrastive_global_voices_28.json"


def train_shards(api):
    """Reject a changed Hub snapshot or incomplete language inventory."""
    if api.get("sha") != REVISION:
        raise ValueError("Dataset revision mismatch")
    paths = sorted(
        item["rfilename"]
        for item in api["siblings"]
        if re.fullmatch(r"en-[a-z]{2}/train-\d+-of-\d+\.parquet", item["rfilename"])
    )
    languages = {p.split("/")[0] for p in paths}
    if len(languages) != 28 or len(paths) != 28:
        raise ValueError(f"Expected exactly one train shard for each of 28 languages, got {len(paths)} shards and {len(languages)} languages")
    return paths


def select_rows(path, tokenizer, max_pairs, min_tokens, max_tokens):
    """Keep the smallest stable pair hashes, independent of Parquet batch size."""
    heap = []
    seen = eligible = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["english", "non_english"]):
        for english, translation in zip(
            batch.column("english").to_pylist(),
            batch.column("non_english").to_pylist(),
            strict=True,
        ):
            index = seen
            seen += 1
            if not all(isinstance(x, str) and x.strip() for x in (english, translation)):
                continue
            lengths = (len(tokenizer.encode(x).ids) for x in (english, translation))
            if not all(min_tokens <= n <= max_tokens for n in lengths):
                continue
            eligible += 1
            digest = hashlib.sha256((english + "\0" + translation).encode()).digest()
            rank = int.from_bytes(digest, "big")
            entry = (-rank, -index, index, english, translation)
            if len(heap) < max_pairs:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    chosen = sorted(heap, key=lambda item: (-item[0], item[2]))
    return [(index, english, translation) for _, _, index, english, translation in chosen], {
        "source_rows": seen,
        "eligible": eligible,
        "selected": len(chosen),
    }


def fetch_shard(cache, relative):
    dest = cache / relative
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{relative}"
    pending = dest.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as source, pending.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        pending.rename(dest)
    finally:
        pending.unlink(missing_ok=True)
    return dest


def combine_heldout(paths, destination):
    """Seal all earlier-stage train and dev text before making a new dev split."""
    inventory = []
    with destination.open("w", encoding="utf-8") as output:
        for path in paths:
            path = Path(path)
            digest = file_hash(path)
            count = 0
            with path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if any(not isinstance(row.get(k), str) or not row[k].strip()
                           for k in ("sentence1", "sentence2")):
                        raise ValueError("Invalid previous-stage held-out row")
                    output.write(json.dumps({
                        "sentence1": row["sentence1"],
                        "sentence2": row["sentence2"],
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
            if file_hash(path) != digest:
                raise ValueError("Previous-stage file changed during preparation")
            inventory.append({"path": str(path), "sha256": digest, "rows": count})
    return inventory


def build(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output must be fresh")
    if file_hash(args.sealed_manifest) != args.sealed_manifest_sha256:
        raise ValueError("Sealed manifest checksum mismatch")
    api_path = Path(args.api)
    api = json.loads(api_path.read_text())
    paths = train_shards(api)
    lock = json.loads(LOCK.read_text())
    if (lock.get("dataset") != DATASET or lock.get("revision") != REVISION
            or lock.get("api_sha256") != file_hash(api_path)
            or set(lock.get("train_shards_sha256", {})) != set(paths)):
        raise ValueError("Pinned Global Voices lock mismatch")
    tokenizer_path = Path(args.tokenizer)
    tokenizer_sha = file_hash(tokenizer_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()
    cache = Path(args.cache)
    output.mkdir(parents=True)
    heldout = output / "previous-stage-heldout.jsonl" if args.heldout else None
    heldout_sources = combine_heldout(args.heldout, heldout) if heldout else []
    heldout_sha = file_hash(heldout) if heldout else None
    sources = []
    statistics = []
    for relative in paths:
        raw = fetch_shard(cache, relative)
        raw_sha = file_hash(raw)
        if raw_sha != lock["train_shards_sha256"][relative]:
            raise ValueError(f"Raw Global Voices shard checksum mismatch: {relative}")
        pair = relative.split("/")[0]
        selected, stats = select_rows(
            raw, tok, args.max_pairs_per_language, args.min_tokens, args.sequence_length
        )
        if file_hash(raw) != raw_sha:
            raise ValueError(f"Raw Global Voices shard changed during selection: {relative}")
        selected_path = output / (pair + ".jsonl")
        with selected_path.open("w", encoding="utf-8") as stream:
            for index, english, translation in selected:
                stream.write(json.dumps({
                    "sentence1": english,
                    "sentence2": translation,
                    "language1": "en",
                    "language2": pair[3:],
                    "upstream_file": relative,
                    "upstream_row": index,
                }, ensure_ascii=False) + "\n")
        sources.append({
            "path": selected_path.name,
            "sha256": file_hash(selected_path),
            "dataset": DATASET,
            "revision": REVISION,
            "license": LICENSE,
            "split": "train",
            "upstream_parquet_sha256": raw_sha,
        })
        statistics.append({"language_pair": pair, **stats})
        print(json.dumps({"pair": pair, **stats}), flush=True)
    manifest = output / "source-manifest.json"
    manifest.write_text(json.dumps({
        "files": sources,
        "api_sha256": file_hash(api_path),
        "lock_sha256": file_hash(LOCK),
        "dataset": DATASET,
        "revision": REVISION,
        "selection": "smallest SHA256(english NUL translation) per shard among eligible token lengths",
        "max_pairs_per_language": args.max_pairs_per_language,
        "min_tokens": args.min_tokens,
        "max_tokens": args.sequence_length,
        "tokenizer_sha256": tokenizer_sha,
        "statistics": statistics,
        "license": LICENSE,
        "attribution": "Global Voices contributors; OPUS alignment; Sentence Transformers packaging",
    }, indent=2, sort_keys=True) + "\n")
    prepared = prepare(
        manifest,
        args.sealed_manifest,
        output / "prepared",
        source_manifest_sha256=file_hash(manifest),
        sealed_manifest_sha256=args.sealed_manifest_sha256,
        heldout_path=heldout,
        heldout_sha256=heldout_sha,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        tokenizer=tokenizer_path,
        tokenizer_sha256=tokenizer_sha,
        sequence_length=args.sequence_length,
    )
    verify(output / "prepared")
    (output / "receipt.json").write_text(json.dumps({
        "source_manifest_sha256": file_hash(manifest),
        "lock_sha256": file_hash(LOCK),
        "prepared_manifest_sha256": file_hash(output / "prepared/manifest.json"),
        "sealed_manifest_sha256": args.sealed_manifest_sha256,
        "heldout_sha256": heldout_sha,
        "heldout_sources": heldout_sources,
        "statistics": statistics,
        "prepared_files": prepared["files"],
        "excluded": prepared["excluded"],
        "limitations": "News-domain English-pivot parallel text; no semantic decontamination or article-level attribution reconstruction",
    }, indent=2, sort_keys=True) + "\n")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, help="Cached pinned Hub dataset API JSON")
    parser.add_argument("--cache", required=True, help="Raw Parquet cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sealed-manifest", required=True)
    parser.add_argument("--sealed-manifest-sha256", required=True)
    parser.add_argument("--heldout", action="append", default=[],
                        help="Previous-stage train and dev JSONL; repeat for each when resuming")
    parser.add_argument("--max-pairs-per-language", type=int, default=10000)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--dev-fraction", type=float, default=0.02)
    args = parser.parse_args()
    if args.max_pairs_per_language < 1 or not 1 <= args.min_tokens <= args.sequence_length:
        parser.error("Invalid pair cap or token lengths")
    print(json.dumps(build(args)["files"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
