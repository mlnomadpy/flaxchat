"""Freeze a released-mmBERT versus YAT-contrastive Tatoeba comparison.

This reuses a fully pinned, prepared 112-subset bitext inventory. It reads
checkpoint metadata and input hashes; it does not allocate TPU resources or
score the sealed test set.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from flaxchat.checkpoint import load_checkpoint_metadata
from flaxchat.encoder import EncoderConfig
from flaxchat.encoder_data import file_hash
from scripts.export_encoder_bitext_campaign import _metadata_sha256, verify_plan


_RUNTIME_FIELDS = (
    "compute_dtype", "residual_dtype", "use_remat", "attention_backend",
    "loss_chunk_size", "mlm_projection", "mlm_loss_backend", "mlm_vocab_tile",
)
_SHARED_FIELDS = (
    "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
    "max_position_embeddings", "pad_token_id", "mask_token_id",
)


def build(source_plan, checkpoint, step, overlap_audit, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen benchmark plan")
    original, _, entries, directories, snapshot_hashes = verify_plan(source_plan)
    metadata = load_checkpoint_metadata(checkpoint, step=step)
    if metadata.get("model_family") != "modernbert_contrastive_encoder" or metadata.get("step") != step:
        raise ValueError("Require explicit YAT contrastive checkpoint step")
    if metadata.get("tokenizer_identity") != snapshot_hashes["tokenizer.json"]:
        raise ValueError("Candidate and released-mmBERT tokenizers differ")
    overlap_audit = Path(overlap_audit)
    audit_hash = file_hash(overlap_audit)
    audit = json.loads(overlap_audit.read_text())
    if (audit.get("passed") is not True or audit.get("overlapping_unique_texts") != 0
            or audit.get("prepared_manifest_sha256") != metadata.get("data_manifest_identity")
            or audit.get("sealed_shards") != len(entries)
            or audit.get("sealed_pairs") != original["pairs"]):
        raise ValueError("Candidate training data lacks a matching zero-overlap Tatoeba audit")
    for entry in entries:
        subset = entry["subset"]
        task = json.loads((directories[subset] / "judgments.json").read_text())
        record = audit.get("per_subset", {}).get(subset, {})
        if (record.get("pairs") != entry["pairs"] or
                record.get("source_sha256") != task.get("source_sha256") or
                record.get("overlapping_unique_texts") != 0):
            raise ValueError(f"Tatoeba audit differs from frozen subset {subset}")
    candidate_config = EncoderConfig(**metadata["resolved_config"]["encoder"])
    snapshot = Path(original["baseline"]["snapshot"])
    released_config = EncoderConfig.from_hf(
        json.loads((snapshot / "config.json").read_text()),
        **{name: getattr(candidate_config, name) for name in _RUNTIME_FIELDS},
    )
    for name in _SHARED_FIELDS:
        if getattr(candidate_config, name) != getattr(released_config, name):
            raise ValueError(f"Unmatched model geometry: {name}")
    if candidate_config.ffn_type != "yat_glu" or candidate_config.attention_score != "yat_softmax":
        raise ValueError("Candidate is not the intended YAT FFN/attention architecture")
    plan = dict(original)
    plan["scope"] = "frozen_released_mmbert_vs_yat_contrastive_bitext"
    plan["require_physical_tpu"] = True
    plan["training_overlap_audit"] = dict(path=str(overlap_audit), sha256=audit_hash,
                                           prepared_manifest_sha256=audit["prepared_manifest_sha256"])
    plan["baseline"] = dict(original["baseline"], encoder_config=asdict(released_config))
    plan["candidate"] = dict(
        checkpoint=str(checkpoint), step=step,
        model_family=metadata["model_family"],
        metadata_sha256=_metadata_sha256(metadata),
    )
    plan["limitations"] = [
        "This is a matched zero-shot embedding comparison, not the mmBERT paper's full suite.",
        "Tatoeba is sealed test data and must not be used to select hyperparameters.",
        "Native bitext search has deterministic tie handling; official MTEB parity is unverified.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n")
    try:
        verify_plan(output)
    except Exception:
        output.unlink()
        raise
    return dict(path=str(output), sha256=file_hash(output), subsets=plan["subsets"], pairs=plan["pairs"], candidate=plan["candidate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--overlap-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_plan, args.checkpoint, args.step, args.overlap_audit, args.output)))


if __name__ == "__main__":
    main()
