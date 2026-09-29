"""Prepare auditable positive pairs without model imports or benchmark leakage.

Input manifests contain ``files`` entries with path, sha256, dataset, revision,
license and split. Training files must be official train splits. Sealed files
must be test splits. JSONL rows use sentence1, sentence2, language1, language2;
sealed rows need only sentence1 and sentence2. Paths resolve beside manifests.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty text required")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            yield number, row


def inputs(path, expected_hash, split):
    path = Path(path)
    if file_hash(path) != expected_hash:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Nonempty pinned files inventory required")
    result = []
    for item in files:
        if any(
            not isinstance(item.get(k), str) or not item[k].strip()
            for k in ("path", "sha256", "dataset", "revision", "license", "split")
        ):
            raise ValueError("Incomplete source identity")
        if item["split"] != split or item["revision"].lower() in (
            "main",
            "master",
            "latest",
        ):
            raise ValueError("Require immutable revision and explicit official split")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid source digest")
        source = (path.parent / item["path"]).resolve()
        if file_hash(source) != item["sha256"]:
            raise ValueError("Source checksum mismatch")
        result.append((source, item))
    return result


def prepare(
    source_manifest,
    sealed_manifest,
    output,
    *,
    source_manifest_sha256,
    sealed_manifest_sha256,
    seed=18,
    dev_fraction=0.02,
    tokenizer=None,
    tokenizer_sha256=None,
    sequence_length=256,
    pad_token_id=0,
    heldout_path=None,
    heldout_sha256=None,
    prior_train_path=None,
    prior_train_sha256=None,
):
    """Split connected text components so no normalized text crosses train/dev."""
    output = Path(output)
    if output.exists() or type(seed) is not int or seed < 0 or not 0 < dev_fraction < 1:
        raise ValueError(
            "Fresh output, nonnegative seed and dev fraction in (0,1) required"
        )
    sources = inputs(source_manifest, source_manifest_sha256, "train")
    sealed = inputs(sealed_manifest, sealed_manifest_sha256, "test")
    forbidden = set()
    for path, _ in sealed:
        for _, row in rows(path):
            forbidden.update(text_hash(row[k]) for k in ("sentence1", "sentence2"))
    sealed_texts = forbidden.copy()
    sealed_unique_texts = len(forbidden)
    heldout_texts = set()
    if (heldout_path is None) != (heldout_sha256 is None):
        raise ValueError("Held-out path and SHA-256 must be supplied together")
    if heldout_path is not None:
        if file_hash(heldout_path) != heldout_sha256:
            raise ValueError("Held-out checksum mismatch")
        for _, row in rows(Path(heldout_path)):
            heldout_texts.update(text_hash(row[k]) for k in ("sentence1", "sentence2"))
        forbidden.update(heldout_texts)
    prior_train_texts = set()
    if (prior_train_path is None) != (prior_train_sha256 is None):
        raise ValueError("Prior train path and SHA-256 must be supplied together")
    if prior_train_path is not None:
        if file_hash(prior_train_path) != prior_train_sha256:
            raise ValueError("Prior train checksum mismatch")
        for _, row in rows(Path(prior_train_path)):
            prior_train_texts.update(text_hash(row[k]) for k in ("sentence1", "sentence2"))
    pairs = {}
    excluded = Counter()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for path, identity in sources:
        for number, row in rows(path):
            left, right = (text_hash(row[k]) for k in ("sentence1", "sentence2"))
            if any(
                not isinstance(row.get(k), str) or not row[k].strip()
                for k in ("language1", "language2")
            ):
                raise ValueError("Explicit language1 and language2 required")
            if left in sealed_texts or right in sealed_texts:
                excluded["sealed_text_overlap"] += 1
                continue
            if left in heldout_texts or right in heldout_texts:
                excluded["heldout_text_overlap"] += 1
                continue
            if left == right:
                excluded["identical_text"] += 1
                continue
            key = tuple(sorted((left, right)))
            provenance = {
                k: identity[k]
                for k in ("dataset", "revision", "license", "split", "sha256")
            }
            provenance["line"] = number
            if key in pairs:
                pairs[key]["provenance"].append(provenance)
                excluded["duplicate_or_reversed_pair"] += 1
                continue
            a, b = find(left), find(right)
            parent[max(a, b)] = min(a, b)
            pairs[key] = dict(
                sentence1=row["sentence1"],
                sentence2=row["sentence2"],
                language1=row["language1"],
                language2=row["language2"],
                text_hashes=[left, right],
                provenance=[provenance],
            )
    if not pairs:
        raise ValueError("No eligible pairs after quarantine")
    partition = {"train": [], "dev": []}
    components = Counter()
    forced_train_components = {find(h) for h in prior_train_texts if h in parent}
    forced_train_pairs = 0
    for key, row in sorted(pairs.items()):
        component = find(key[0])
        draw = (
            int(hashlib.sha256(f"{seed}:{component}".encode()).hexdigest(), 16) / 2**256
        )
        split = "train" if component in forced_train_components else (
            "dev" if draw < dev_fraction else "train"
        )
        forced_train_pairs += component in forced_train_components
        row.update(
            id=hashlib.sha256(":".join(key).encode()).hexdigest(),
            split=split,
            component=component,
        )
        partition[split].append(row)
        components[component] += 1
    if not all(partition.values()):
        raise ValueError(
            "Empty train/dev partition; add source data, do not silently resample"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".contrastive-", dir=output.parent))
    try:
        inventory = {}
        for split, values in partition.items():
            dest = staging / f"{split}.jsonl"
            with dest.open("w") as stream:
                for row in values:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
            inventory[dest.name] = dict(
                sha256=file_hash(dest),
                rows=len(values),
                language_pairs=dict(
                    sorted(
                        Counter(
                            r["language1"] + ":" + r["language2"] for r in values
                        ).items()
                    )
                ),
            )
        # Detect inputs changing during construction before publishing anything.
        inputs(source_manifest, source_manifest_sha256, "train")
        inputs(sealed_manifest, sealed_manifest_sha256, "test")
        if heldout_path is not None and file_hash(heldout_path) != heldout_sha256:
            raise ValueError("Held-out checksum changed during preparation")
        if prior_train_path is not None and file_hash(prior_train_path) != prior_train_sha256:
            raise ValueError("Prior train checksum changed during preparation")
        receipt = dict(
            format="flaxchat-contrastive-text-pairs-v1",
            files=inventory,
            source_manifest_sha256=source_manifest_sha256,
            sealed_manifest_sha256=sealed_manifest_sha256,
            seed=seed,
            dev_fraction=dev_fraction,
            excluded=dict(excluded),
            sealed_unique_texts=sealed_unique_texts,
            heldout_unique_texts=len(heldout_texts),
            heldout_sha256=heldout_sha256,
            prior_train_sha256=prior_train_sha256,
            prior_train_unique_texts=len(prior_train_texts),
            forced_train_components=len(forced_train_components),
            forced_train_pairs=forced_train_pairs,
            components=len(components),
            largest_component_pairs=max(components.values()),
            normalization="NFKC, casefold, whitespace collapse; SHA256 UTF8",
            partition="Hash of seed and lexicographically minimum text hash in undirected connected component",
            limitations="Exact normalized text exclusion only; no semantic/substring or pretrained-exposure guarantee; no negatives or tokenizer applied",
            quality_qualified=False,
        )
        if tokenizer is not None:
            receipt["tokenization"] = tokenize(
                staging, tokenizer, tokenizer_sha256, sequence_length, pad_token_id
            )
        elif tokenizer_sha256 is not None:
            raise ValueError("Tokenizer path required with digest")
        (staging / "manifest.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def tokenize(directory, tokenizer_path, expected_sha256, sequence_length, pad_token_id):
    """Produce paired arrays with collision-free IDs; never truncate silently."""
    import numpy as np
    from tokenizers import Tokenizer

    if file_hash(tokenizer_path) != expected_sha256:
        raise ValueError("Tokenizer checksum mismatch")
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("Positive sequence length required")
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()
    if type(pad_token_id) is not int or not 0 <= pad_token_id < tok.get_vocab_size():
        raise ValueError("Invalid pad token ID")
    paired = {
        s: [row for _, row in rows(directory / f"{s}.jsonl")] for s in ("train", "dev")
    }
    hashes = sorted(
        {h for values in paired.values() for row in values for h in row["text_hashes"]}
    )
    groups = sorted({row["id"] for values in paired.values() for row in values})
    if max(len(hashes), len(groups)) > np.iinfo(np.int32).max:
        raise ValueError("Text/group ID space exceeds int32")
    text_ids = {h: i for i, h in enumerate(hashes)}
    group_ids = {h: i for i, h in enumerate(groups)}
    inventory = {}
    for split, values in paired.items():
        arrays = {}
        for side, field, column in [
            ("query", "sentence1", 0),
            ("document", "sentence2", 1),
        ]:
            encoded = np.full(
                (len(values), sequence_length), pad_token_id, dtype=np.int32
            )
            for i, row in enumerate(values):
                ids = tok.encode(row[field]).ids
                if not ids or len(ids) > sequence_length:
                    raise ValueError(
                        "Empty or overlength pair; refusing silent truncation"
                    )
                encoded[i, : len(ids)] = ids
            arrays[side + "_tokens"] = encoded
            arrays[side + "_text_ids"] = np.asarray(
                [text_ids[r["text_hashes"][column]] for r in values], np.int32
            )
        arrays["positive_group_ids"] = np.asarray(
            [group_ids[r["id"]] for r in values], np.int32
        )
        folder = directory / split
        folder.mkdir()
        for name, array in arrays.items():
            path = folder / (name + ".npy")
            np.save(path, array, allow_pickle=False)
            inventory[str(path.relative_to(directory))] = dict(
                sha256=file_hash(path), shape=list(array.shape), dtype=str(array.dtype)
            )
    mapping = directory / "text_ids.json"
    mapping.write_text(
        json.dumps(
            dict(text_hashes=hashes, positive_pair_hashes=groups), sort_keys=True
        )
        + "\n"
    )
    if file_hash(tokenizer_path) != expected_sha256:
        raise ValueError("Tokenizer changed during preparation")
    return dict(
        tokenizer_sha256=expected_sha256,
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        vocab_size=tok.get_vocab_size(),
        files=inventory,
        id_mapping_sha256=file_hash(mapping),
        id_policy="Sorted full SHA256 identities mapped bijectively to int32; positive groups are unique deduplicated pairs, not inferred semantic equivalence",
        truncation=False,
    )


def verify(directory):
    """Validate integrity and cross-split exact text separation before consumption."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "flaxchat-contrastive-text-pairs-v1":
        raise ValueError("Unsupported pair format")
    tokenization = manifest.get("tokenization")
    if tokenization:
        required = {
            f"{split}/{name}.npy"
            for split in ("train", "dev")
            for name in (
                "query_tokens",
                "document_tokens",
                "query_text_ids",
                "document_text_ids",
                "positive_group_ids",
            )
        }
        if set(tokenization["files"]) != required:
            raise ValueError("Incomplete token array inventory")
        if file_hash(directory / "text_ids.json") != tokenization["id_mapping_sha256"]:
            raise ValueError("ID mapping integrity failure")
        for name, item in tokenization["files"].items():
            path = (directory / name).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or file_hash(path) != item["sha256"]
            ):
                raise ValueError("Token array integrity failure")
    texts = {}
    for split in ("train", "dev"):
        path = directory / f"{split}.jsonl"
        item = manifest["files"][path.name]
        if file_hash(path) != item["sha256"]:
            raise ValueError("Prepared file integrity failure")
        seen = set()
        count = 0
        for _, row in rows(path):
            hashes = [text_hash(row[k]) for k in ("sentence1", "sentence2")]
            if row["split"] != split or hashes != row["text_hashes"]:
                raise ValueError("Prepared row identity failure")
            seen.update(hashes)
            count += 1
        if count != item["rows"]:
            raise ValueError("Prepared row count mismatch")
        texts[split] = seen
    if texts["train"] & texts["dev"]:
        raise ValueError("Train/dev text overlap")
    return manifest


def load_tokenized(directory, tokenizer_path, *, split, sequence_length):
    """Return (arrays_dict, manifest) after validating prepared token identities."""
    import numpy as np

    directory = Path(directory)
    if split not in ("train", "dev"):
        raise ValueError("Expected train or dev split")
    manifest = verify(directory)
    info = manifest.get("tokenization")
    if (
        not info
        or info["tokenizer_sha256"] != file_hash(tokenizer_path)
        or info["sequence_length"] != sequence_length
    ):
        raise ValueError("Missing/mismatched prepared tokenizer or sequence length")
    arrays = {}
    size = manifest["files"][split + ".jsonl"]["rows"]
    for name in (
        "query_tokens",
        "document_tokens",
        "query_text_ids",
        "document_text_ids",
        "positive_group_ids",
    ):
        value = np.load(
            directory / split / (name + ".npy"), mmap_mode="r", allow_pickle=False
        )
        expected = (size, sequence_length) if name.endswith("_tokens") else (size,)
        if value.dtype != np.int32 or value.shape != expected or np.any(value < 0):
            raise ValueError("Invalid prepared array shape/dtype/range")
        if name.endswith("_tokens") and np.any(value >= info["vocab_size"]):
            raise ValueError("Token ID outside vocabulary")
        arrays[name] = value
    return arrays, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source-manifest",
        "sealed-manifest",
        "output",
        "source-manifest-sha256",
        "sealed-manifest-sha256",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--dev-fraction", type=float, default=0.02)
    parser.add_argument("--heldout-path")
    parser.add_argument("--heldout-sha256")
    parser.add_argument("--prior-train-path")
    parser.add_argument("--prior-train-sha256")
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
