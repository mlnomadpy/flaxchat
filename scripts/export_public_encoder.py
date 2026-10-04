"""Export a final Orbax model item to a portable weight-only Hub directory."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from scripts.release_contract import artifact_hashes, canonical_hash, checkpoint_identity
import orbax.checkpoint as ocp

from flaxchat.checkpoint import _restore_args_on_current_topology
from flaxchat.encoder import EncoderConfig
from flaxchat.public_encoder import export_state_safetensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_model", type=Path)
    parser.add_argument("checkpoint_metadata", type=Path)
    parser.add_argument("checkpoint_manifest", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-family", default="modernbert")
    parser.add_argument("--step", type=int, default=48236)
    args = parser.parse_args()
    metadata = json.loads(args.checkpoint_metadata.read_text())
    manifest = json.loads(args.checkpoint_manifest.read_text())
    if metadata.get("model_family") != args.model_family or manifest.get("step") != args.step:
        raise ValueError(f"Expected {args.model_family} step-{args.step} encoder checkpoint")
    if (canonical_hash(metadata) != manifest.get("metadata_sha256") or canonical_hash(manifest.get("identity")) != manifest.get("identity_sha256")
            or manifest.get("identity") != checkpoint_identity(metadata)):
        raise ValueError("Checkpoint metadata/identity hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.checkpoint_model.resolve()
    metadata_tree = ocp.PyTreeCheckpointHandler().metadata(model_path)
    state = ocp.PyTreeCheckpointer().restore(
        str(model_path), item=metadata_tree,
        restore_args=_restore_args_on_current_topology(metadata_tree),
    )
    report = export_state_safetensors(
        state, args.output / "model.safetensors",
        expected_manifest=manifest["model_state"],
    )
    config = EncoderConfig(**metadata["resolved_config"]["encoder"])
    (args.output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    shutil.copyfile(args.tokenizer, args.output / "tokenizer.json")
    (args.output / "checkpoint-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "checkpoint-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "export.json").write_text(json.dumps({
        "format": "flaxchat-authenticated-encoder-export-v2",
        "artifacts_sha256": artifact_hashes(args.output, ("model.safetensors", "config.json", "tokenizer.json",
                                                         "checkpoint-metadata.json", "checkpoint-manifest.json")),
        "source_checkpoint_step": args.step,
        "source_model_family": args.model_family,
        "source_checkpoint_metadata_sha256": manifest["metadata_sha256"],
        "source_checkpoint_manifest_identity_sha256": manifest["identity_sha256"],
        "source_python_sha256": metadata["source_python_sha256"],
        "tokenizer_identity": metadata["tokenizer_identity"],
        **report,
    }, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
