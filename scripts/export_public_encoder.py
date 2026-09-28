"""Export a final Orbax model item to a portable weight-only Hub directory."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.checkpoint_metadata.read_text())
    manifest = json.loads(args.checkpoint_manifest.read_text())
    if metadata.get("model_family") != "modernbert" or manifest.get("step") != 48236:
        raise ValueError("Expected the final step-48236 encoder checkpoint")
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
    (args.output / "export.json").write_text(json.dumps({
        "source_checkpoint_step": 48236,
        "source_python_sha256": metadata["source_python_sha256"],
        "tokenizer_identity": metadata["tokenizer_identity"],
        **report,
    }, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
