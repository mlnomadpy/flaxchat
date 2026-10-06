"""Prepare authenticated independent candidate retrieval/bitext/code gate arrays.

No models or network access. Known contrastive parent exposure must be checked
from actual committed stage metadata and complete historical training raw files.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from flaxchat.encoder_data import file_hash
from flaxchat.embedding_quality import paired_dev_ids
from flaxchat.embedding_dev_exposure import parent_exposure
from flaxchat.embedding_development_quarantine import load_candidate_exclusions


def portable_exposure_path(path, root, output):
    """Bind a relocation-stable reference without permitting external links."""
    root, path, output = Path(root).absolute(), Path(path).absolute(), Path(output).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Portable root must be an existing regular directory')
    for target in (path, output):
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Portable exposure path must remain inside portable root')
        for component in (target, *target.parents):
            if component == root:
                break
            if component.is_symlink():
                raise ValueError('Portable exposure paths must not traverse symlinks')
    if not path.exists():
        raise ValueError('Portable historical input must exist')
    return os.path.relpath(path.resolve(), output.resolve())


def prepare(
    candidate,
    names,
    exposure_input,
    parent_manifest,
    tokenizer,
    encoder_config,
    output,
    *,
    query_length=128,
    document_length=256,
    timeout_seconds=600,
    portable_root=None,
):
    from tokenizers import Tokenizer

    candidate, output = Path(candidate), Path(output)
    if output.exists() or not names or len(set(names)) != len(names):
        raise ValueError("Fresh output and unique candidate source names required")
    index = load_candidate_exclusions(candidate)
    receipt = json.loads((candidate / "exclusions.json").read_text())
    if any(name not in receipt["sources"] for name in names):
        raise ValueError("Unknown candidate source")
    tasks = {receipt["sources"][name]["task"] for name in names}
    if len(tasks) != 1 or not tasks <= {"retrieval", "bitext", "code"}:
        raise ValueError("One independent retrieval/bitext/code task per gate required")
    parent_hashes = json.loads(Path(parent_manifest).read_text())
    if set(parent_hashes) != {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
    } or any(
        not re.fullmatch("[0-9a-f]{64}", value) for value in parent_hashes.values()
    ):
        raise ValueError("Actual immutable parent file inventory required")
    if (
        file_hash(tokenizer) != parent_hashes["tokenizer.json"]
        or file_hash(encoder_config) != parent_hashes["config.json"]
    ):
        raise ValueError(
            "Parent tokenizer/config differs from immutable parent inventory"
        )
    # Seal paths against the original inventory base before copying. A portable
    # bundle uses relative references into its authenticated shared history.
    exposure_spec = json.loads(Path(exposure_input).read_text())
    for stage in exposure_spec.get("stages", []):
        original = Path(exposure_input).parent / stage["checkpoint_metadata"]
        stage["checkpoint_metadata"] = (portable_exposure_path(original, portable_root, output)
            if portable_root is not None else str(original.resolve()))
        for entry in stage.get("sources", []):
            original = Path(exposure_input).parent / entry["directory"]
            entry["directory"] = (portable_exposure_path(original, portable_root, output)
                if portable_root is not None else str(original.resolve()))
            if portable_root is not None:
                for name in ('manifest.json', 'train.jsonl'):
                    portable_exposure_path(original / name, portable_root, output)
    exposure_spec_sha = file_hash(exposure_input)
    rows = []
    for name in names:
        rows.extend(
            json.loads(line)
            for line in (candidate / (name + ".jsonl")).read_text().splitlines()
        )
    if len(rows) < 2 or any(
        not isinstance(row[field], str) or not row[field].strip()
        for row in rows
        for field in ("query", "positive")
    ):
        raise ValueError("Independent development requires real nonempty pairs")
    ids = paired_dev_ids(rows)
    config = json.loads(Path(encoder_config).read_text())
    if any(
        not 2 <= length <= config["max_position_embeddings"]
        for length in (query_length, document_length)
    ):
        raise ValueError("Invalid independent development context lengths")
    tok = Tokenizer.from_file(str(tokenizer))
    tok.no_padding()
    tok.no_truncation()
    if (
        tok.get_vocab_size() != config["vocab_size"]
        or config["pad_token_id"] not in tok.get_vocab().values()
    ):
        raise ValueError("Independent development tokenizer schema mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".independent-dev-", dir=output.parent))
    try:
        shutil.copytree(candidate, staging / "candidate")
        shutil.copyfile(tokenizer, staging / "tokenizer.json")
        # The copied bundle must still be the one audited before tokenization.
        if load_candidate_exclusions(staging / "candidate").identity != index.identity:
            raise ValueError("Candidate changed during independent preparation")
        (staging / "parent-exposure-input.json").write_text(
            json.dumps(exposure_spec, sort_keys=True, indent=2) + "\n"
        )
        if file_hash(exposure_input) != exposure_spec_sha:
            raise ValueError("Parent exposure inventory changed during preparation")
        proof = parent_exposure(
            staging / "parent-exposure-input.json",
            index,
            parent_hashes,
            timeout_seconds=timeout_seconds,
        )
        (staging / "parent-exposure-proof.json").write_text(
            json.dumps(proof, sort_keys=True, indent=2) + "\n"
        )
        (staging / "raw.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            )
        )
        for name in ("query_text_ids", "positive_text_ids", "positive_group_ids"):
            np.save(staging / (name + ".npy"), ids[name], allow_pickle=False)
        truncated = {}
        for field, name, length in (
            ("query", "query_tokens", query_length),
            ("positive", "positive_tokens", document_length),
        ):
            tokens = np.full((len(rows), length), config["pad_token_id"], np.int32)
            truncated[name] = 0
            for first in range(0, len(rows), 1024):
                for offset, encoded in enumerate(
                    tok.encode_batch([row[field] for row in rows[first : first + 1024]])
                ):
                    values = encoded.ids
                    if not values:
                        raise ValueError(
                            "Independent tokenizer produced empty sequence"
                        )
                    if len(values) > length:
                        values = values[: length - 1] + values[-1:]
                        truncated[name] += 1
                    tokens[first + offset, : len(values)] = values
            np.save(staging / (name + ".npy"), tokens, allow_pickle=False)
        files = (
            "raw.jsonl",
            "tokenizer.json",
            "query_tokens.npy",
            "positive_tokens.npy",
            "query_text_ids.npy",
            "positive_text_ids.npy",
            "positive_group_ids.npy",
            "parent-exposure-input.json",
            "parent-exposure-proof.json",
        )
        manifest = {
            "format": "flaxchat-embedding-retrieval-dev-v1",
            "task": next(iter(tasks)),
            "selection_split": "independent-candidate-holdout",
            "candidate_sources": names,
            "candidate_identity": index.identity,
            "source_identity": {name: receipt["sources"][name] for name in names},
            "parent_exposure": {
                "proof_sha256": file_hash(staging / "parent-exposure-proof.json"),
                "complete": proof["complete"],
                "scope": "declared-known-contrastive-stage-inventory",
                "checked_stage_identities": sorted(proof["checked_stages"]),
                "unresolved_exposure": proof["unresolved_exposure"],
            },
            "parent_files_sha256": parent_hashes,
            "tokenizer_sha256": file_hash(tokenizer),
            "vocab_size": config["vocab_size"],
            "pad_id": config["pad_token_id"],
            "rows": len(rows),
            "query_length": query_length,
            "document_length": document_length,
            "truncated": truncated,
            "truncation_policy": "disable-inherited;terminal-token-preserving-v1",
            "preparation_sha256": file_hash(Path(__file__)),
            "files": {name: file_hash(staging / name) for name in files},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate",
        "exposure-input",
        "parent-manifest",
        "tokenizer",
        "encoder-config",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-name", action="append", required=True)
    parser.add_argument("--query-length", type=int, default=128)
    parser.add_argument("--document-length", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument('--portable-root', type=Path,
                        help='Package root containing output and complete authenticated historical inputs; seal relative references')
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.candidate,
                args.source_name,
                args.exposure_input,
                args.parent_manifest,
                args.tokenizer,
                args.encoder_config,
                args.output,
                query_length=args.query_length,
                document_length=args.document_length,
                timeout_seconds=args.timeout_seconds,
                portable_root=args.portable_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
