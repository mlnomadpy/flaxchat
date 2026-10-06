"""Prepare pinned MS MARCO, multilingual, or code retrieval data on a GCP VM.

Only official training splits are consumed. This script performs data work, not
model inference. Each source gets its own exact-text-disjoint train/dev split
and memory-mappable token arrays for a TPU training process.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from flaxchat.embedding_data import IDENTITY_POLICY, row_identities, text_identity

import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


SOURCES = {
    "msmarco": {
        "repo": "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "revision": "84ed2d35626f617d890bd493b4d6db69a741e0e2",
        "config": "triplet-hard",
        "train_limit": 1_250_000,
        "candidate_limit": 1_500_000,
    },
    "miracl": {
        "repo": "nlpai-lab/miracl-multilingual-triplets",
        "revision": "71bc9f8e7d86b55203ed3104362b0661789a6c31",
        "config": "all 51 language subsets",
        "group_alignment_policy": "upstream-id-across-languages-v1",
        "train_limit": None,
        "candidate_limit": None,
    },
    "code": {
        "repo": "sentence-transformers/codesearchnet",
        "revision": "079a958b01dc87cf07b66a68414c4b4196d889cc",
        "config": "pair",
        "train_limit": 300_000,
        "candidate_limit": 350_000,
    },
}


def text_hash(value: str, modality: str = "text") -> str:
    return text_identity(value, modality)


def load_registry(path: Path) -> dict:
    """Explicit pinned train-only pair/triplet sources, including bitext replay."""
    import re
    registry = json.loads(path.read_text())
    if not isinstance(registry, dict) or not registry:
        raise ValueError("Dataset registry must be a nonempty object")
    for name, spec in registry.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValueError("Invalid dataset registry entry")
        if spec.get("split", "train") != "train":
            raise ValueError("Preparation registry must use official training splits")
        if not isinstance(spec.get("fields"), dict) or not {"query", "positive"} <= set(spec["fields"]):
            raise ValueError("Registry requires query/positive field mapping")
        if "jsonl" in spec:
            if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")):
                raise ValueError("Local replay source must have SHA256 identity")
        elif not spec.get("repo") or not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
            raise ValueError("Hugging Face replay source must pin a commit revision")
        for limit in ("candidate_limit", "train_limit"):
            if spec.get(limit) is not None and (type(spec[limit]) is not int or spec[limit] < 2):
                raise ValueError("Registry row limits must be positive integers or null")
        spec.setdefault("train_limit", None)
        spec.setdefault("candidate_limit", None)
    return registry


def _registry_rows(spec, seed):
    from datasets import load_dataset
    if "jsonl" in spec:
        path = Path(spec["jsonl"])
        if file_hash(path) != spec["sha256"]:
            raise ValueError("Replay source checksum mismatch")
        rows = _raw_rows(path)
    else:
        rows = load_dataset(spec["repo"], spec.get("config"), split="train",
                            revision=spec["revision"]).shuffle(seed=seed)
    for position, original in enumerate(rows):
        if spec["candidate_limit"] is not None and position >= spec["candidate_limit"]:
            break
        fields = spec["fields"]
        yield {"query": original[fields["query"]], "positive": original[fields["positive"]],
               "negative": original[fields["negative"]] if fields.get("negative") else None,
               "group": str(original[fields["group"]]) if fields.get("group") else None,
               "upstream_group": str(original[fields["group"]]) if fields.get("group") else None,
               "language": str(original[fields["language"]]) if fields.get("language") else spec.get("language", "und"),
               "programming_language": str(original[fields["programming_language"]]) if fields.get("programming_language") else spec.get("programming_language", "unknown"),
               "programming_language_provenance": str(original[fields["programming_language_provenance"]]) if fields.get("programming_language_provenance") else (f"column:{fields['programming_language']}" if fields.get("programming_language") else ("registry-constant" if spec.get("programming_language") else "unavailable")),
               "modalities": spec.get("modalities", {}), "coordinate": str(position)}


def selected_rows(source: str, seed: int):
    from datasets import get_dataset_config_names, load_dataset

    spec = SOURCES[source]
    if "fields" in spec:
        yield from _registry_rows(spec, seed)
        return
    if source == "miracl":
        configs = sorted(get_dataset_config_names(spec["repo"], revision=spec["revision"]))
        if len(configs) != 51 or "en" not in configs:
            raise ValueError(f"MIRACL config inventory changed: {configs}")
        for language in configs:
            dataset = load_dataset(spec["repo"], language, split="train",
                                   revision=spec["revision"])
            for position, row in enumerate(dataset):
                yield dict(query=row["query"], positive=row["positive"],
                           negative=row["negative"], group=f"miracl:{row['id']}", upstream_group=str(row['id']),
                           coordinate=f"{language}:{position}", language=language)
        return
    dataset = load_dataset(spec["repo"], spec["config"], split="train",
                           revision=spec["revision"])
    if source == "msmarco" and len(dataset) != 11_662_655:
        raise ValueError("MS MARCO hard-triplet inventory changed")
    if source == "code" and len(dataset) != 1_375_067:
        raise ValueError("CodeSearchNet pair inventory changed")
    shuffled = dataset.shuffle(seed=seed)
    limit = spec["candidate_limit"]
    for position, row in enumerate(shuffled.select(range(min(limit, len(shuffled))))):
        if source == "msmarco":
            query, positive, negative = row["query"], row["positive"], row["negative"]
        else:
            query, positive, negative = row["comment"], row["code"], None
        record = dict(query=query, positive=positive, negative=negative,
                      group=None, coordinate=f"shuffle:{position}", language="und")
        if source == "code":
            record.update(code_language_provenance(row))
        yield record


def code_language_provenance(row):
    """Retain declared code provenance; never infer it from natural text or syntax."""
    for field in ('programming_language', 'language', 'lang'):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return {'programming_language': value.strip().lower(),
                    'programming_language_provenance': f'column:{field}'}
    return {'programming_language': 'unknown', 'programming_language_provenance': 'unavailable'}


def _raw_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _row_batches(path: Path, size: int = 1024):
    batch = []
    for row in _raw_rows(path):
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _tokenize(directory: Path, tokenizer: Path, query_length: int,
              document_length: int, pad_id: int) -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    vocab = tok.get_vocab()
    declared_padding = tok.padding
    recognized_pad = {vocab[token] for token in ("[PAD]", "<pad>") if token in vocab}
    if (pad_id not in vocab.values() or recognized_pad and pad_id not in recognized_pad
            or declared_padding is not None and declared_padding["pad_id"] != pad_id):
        raise ValueError("Prepared pad ID does not match tokenizer padding")
    tok.no_padding()
    tok.no_truncation()
    inventory = {}
    text_ids: dict[str, int] = {}
    group_ids: dict[str, int] = {}
    truncated = Counter()
    code_provenance = {split: Counter() for split in ('train', 'dev')}

    def assign(mapping: dict[str, int], key: str) -> int:
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        return mapping[key]

    for split in ("train", "dev"):
        count = sum(1 for _ in _raw_rows(directory / f"{split}.jsonl"))
        folder = directory / split
        folder.mkdir()
        specs = {
            "query_tokens": (np.int32, (count, query_length)),
            "positive_tokens": (np.int32, (count, document_length)),
            "negative_tokens": (np.int32, (count, document_length)),
            "query_text_ids": (np.int32, (count,)),
            "positive_text_ids": (np.int32, (count,)),
            "negative_text_ids": (np.int32, (count,)),
            "positive_group_ids": (np.int32, (count,)),
            "negative_valid": (np.bool_, (count,)),
        }
        arrays = {name: np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+",
                                                   dtype=dtype, shape=shape)
                  for name, (dtype, shape) in specs.items()}
        index = 0
        for rows in _row_batches(directory / f"{split}.jsonl"):
            for field, length in (("query", query_length), ("positive", document_length),
                                  ("negative", document_length)):
                contents = [row[field] if row[field] is not None else row["positive"]
                            for row in rows]
                encoded = tok.encode_batch(contents)
                token_array = arrays[f"{field}_tokens"]
                token_array[index:index + len(rows)] = pad_id
                for offset, item in enumerate(encoded):
                    ids = item.ids
                    if not ids:
                        raise ValueError("Tokenizer produced an empty sequence")
                    if len(ids) > length:
                        truncated[f"{split}:{field}"] += 1
                        # Keep the terminal special token after truncation.
                        ids = ids[:length - 1] + ids[-1:]
                    token_array[index + offset, :len(ids)] = ids
            for offset, row in enumerate(rows):
                if any(value == 'code' for value in row.get('modalities', {}).values()):
                    code_provenance[split][str(row.get('programming_language', 'unknown')).strip().lower()] += 1
                for field in ("query", "positive", "negative"):
                    arrays[f"{field}_text_ids"][index + offset] = assign(
                        text_ids, row_identities(row, directory.name)[field])
                arrays["positive_group_ids"][index + offset] = assign(group_ids, row["group"])
                arrays["negative_valid"][index + offset] = row["negative"] is not None
            index += len(rows)
            if index // 100_000 != (index - len(rows)) // 100_000:
                print(json.dumps({"event": "tokenized", "split": split, "rows": index}), flush=True)
        if index != count:
            raise ValueError("Raw row count changed during tokenization")
        for name, array in arrays.items():
            array.flush()
            path = folder / f"{name}.npy"
            inventory[str(path.relative_to(directory))] = {
                "sha256": file_hash(path), "shape": list(array.shape), "dtype": str(array.dtype)}
        del arrays
    if max(len(text_ids), len(group_ids)) > np.iinfo(np.int32).max:
        raise ValueError("ID space exceeds int32")
    return {"files": inventory, "vocab_size": tok.get_vocab_size(),
            "programming_languages": {split: dict(counts) for split, counts in code_provenance.items()},
            "programming_language_policy": "declared-provenance-only-v1; separate-from-natural-language",
            "special_token_ids": sorted({pad_id} | {item["id"] for item in json.loads(tok.to_str()).get("added_tokens", [])
                                                       if item.get("special")} | {i for token, i in vocab.items()
                                             if token in ("[PAD]", "[MASK]", "[CLS]", "[SEP]", "<s>", "</s>", "<pad>")}),
            "truncation_policy": "disable-inherited; terminal-token-preserving-v1",
            "text_id_count": len(text_ids),
            "group_id_count": len(group_ids), "truncated": dict(truncated)}


def prepare(source: str, output: Path, tokenizer: Path, *, seed: int = 29,
            query_length: int = 128, document_length: int = 256,
            pad_id: int = 0, development_exclusions: Path | None = None) -> dict:
    if (source not in SOURCES or output.exists() or seed < 0 or query_length < 2
            or document_length < 2 or pad_id < 0):
        raise ValueError("Invalid source, output, seed, or sequence length")
    from flaxchat.embedding_development_quarantine import load_candidate_exclusions
    exclusion = load_candidate_exclusions(development_exclusions) if development_exclusions is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".yat-{source}-", dir=output.parent))
    try:
        candidate = staging / "candidates.jsonl"
        seen = set()
        rejected = Counter()
        with candidate.open("w", encoding="utf-8") as stream:
            for row in selected_rows(source, seed):
                values = (row["query"], row["positive"], row["negative"])
                if any(not isinstance(value, str) or not value.strip()
                       for value in values[:2]) or (values[2] is not None and
                                                    (not isinstance(values[2], str) or not values[2].strip())):
                    rejected["empty"] += 1
                    continue
                row.setdefault("modalities", {"query": "text", "positive": "code" if source == "code" else "text",
                                     "negative": "code" if source == "code" else "text"})
                hashes = [row_identities(row, source)[field] for field in ("query", "positive", "negative")
                          if row[field] is not None]
                if exclusion is not None:
                    reason = exclusion.matches(row, SOURCES[source], source)
                    if reason is not None:
                        rejected['independent_dev_' + reason] += 1
                        continue
                if row["group"] is None:
                    row["group"] = f"{source}:{hashes[0]}"
                if len(set(hashes)) != len(hashes):
                    rejected["identical_text"] += 1
                    continue
                key = tuple(hashes)
                if key in seen:
                    rejected["duplicate_triplet"] += 1
                    continue
                seen.add(key)
                row["text_hashes"] = hashes
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                if len(seen) % 100_000 == 0:
                    print(json.dumps({"event": "selected", "source": source,
                                      "rows": len(seen)}), flush=True)
        del seen
        dev_hashes = set()
        for row in _raw_rows(candidate):
            if int(hashlib.sha256(f"{seed}:{row['group']}".encode()).hexdigest(), 16) % 100 < 1:
                dev_hashes.update(row["text_hashes"])
        counts = Counter()
        with (staging / "train.jsonl").open("w", encoding="utf-8") as train, \
             (staging / "dev.jsonl").open("w", encoding="utf-8") as dev:
            for row in _raw_rows(candidate):
                held_out = int(hashlib.sha256(f"{seed}:{row['group']}".encode()).hexdigest(), 16) % 100 < 1
                if held_out:
                    dev.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["dev"] += 1
                elif any(value in dev_hashes for value in row["text_hashes"]):
                    rejected["dev_text_overlap"] += 1
                elif SOURCES[source]["train_limit"] is None or counts["train"] < SOURCES[source]["train_limit"]:
                    train.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["train"] += 1
        candidate.unlink()
        if counts["train"] < 2 or counts["dev"] < 2:
            raise ValueError("Empty training or held-out split")
        train_hashes = {h for row in _raw_rows(staging / "train.jsonl") for h in row["text_hashes"]}
        if train_hashes & dev_hashes:
            raise ValueError("Exact train/dev text overlap")
        tokenization = _tokenize(staging, tokenizer, query_length, document_length, pad_id)
        manifest = {"format": "flaxchat-yat-embedding-triplets-v2", "identity_policy": IDENTITY_POLICY,
                    "preparation_sha256": file_hash(Path(__file__)),
                    "identity_implementation_sha256": file_hash(Path(__file__).parents[1] / "flaxchat/embedding_data.py"), "source": source,
                    "source_identity": SOURCES[source], "source_split": "train",
                    "seed": seed, "rows": dict(counts), "rejected": dict(rejected),
                    "selection": "seeded Hugging Face shuffle for MS MARCO/code; all pinned MIRACL subsets",
                    "dev_split": "1% deterministic source-group hash; exact text quarantine",
                    "tokenizer_sha256": file_hash(tokenizer), "query_length": query_length,
                    "document_length": document_length, "pad_id": pad_id,
                    "tokenization": tokenization, "vocab_size": tokenization["vocab_size"],
                    "raw_files": {f"{split}.jsonl": file_hash(staging / f"{split}.jsonl")
                                  for split in ("train", "dev")}}
        if exclusion is not None:
            verified = load_candidate_exclusions(development_exclusions)
            if verified.identity != exclusion.identity:
                raise ValueError('Development exclusions changed during preparation')
            manifest['development_quarantine'] = {'identity': exclusion.identity,
                'applied_to_source': source, 'rejected': {k: v for k, v in rejected.items() if k.startswith('independent_dev_')},
                'parent_exposure_checked': False}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--registry", type=Path, help="Pinned train-only pair/triplet JSON registry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--query-length", type=int, default=128)
    parser.add_argument("--document-length", type=int, default=256)
    parser.add_argument("--pad-id", type=int, default=0)
    parser.add_argument('--development-exclusions', type=Path, help='Authenticated independent candidate development directory to quarantine before selection')
    args = parser.parse_args()
    if args.registry:
        registry = load_registry(args.registry)
        if set(registry) & set(SOURCES):
            raise ValueError("Registry cannot override built-in source identities")
        SOURCES.update(registry)
    print(json.dumps(prepare(args.source, args.output, args.tokenizer,
                             seed=args.seed, query_length=args.query_length,
                             document_length=args.document_length, pad_id=args.pad_id,
                             development_exclusions=args.development_exclusions),
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
