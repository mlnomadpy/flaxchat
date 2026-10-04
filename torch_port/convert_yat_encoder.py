"""Convert a verified FlaxChat YAT safetensors release to PyTorch layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts.release_contract import artifact_hashes, validate_export
from torch_port.yat_encoder import YatEncoderConfig, YatTorchEncoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_name(name: str) -> tuple[str, bool]:
    if name == "['decoder_bias']":
        return "decoder_bias", False
    parts = re.findall(r"\['([^']+)'\]", name)
    if "".join(f"['{part}']" for part in parts) != name:
        raise ValueError(f"Malformed Flax tensor path: {name}")
    if parts == ["embedding", "embedding"]:
        return "embedding.weight", False
    if len(parts) == 2 and parts[1] == "scale" and parts[0] in (
            "embedding_norm", "final_norm", "head_norm"):
        return f"{parts[0]}.weight", False
    if parts == ["head_dense", "kernel"]:
        return "head_dense.weight", True
    if len(parts) == 3 and parts[0] == "layers" and parts[1].isdigit():
        if parts[2] in ("yat_alpha", "yat_attention_alpha"):
            return f"layers.{parts[1]}.{parts[2]}", False
    if len(parts) == 4 and parts[0] == "layers" and parts[1].isdigit():
        if parts[3] == "kernel" and parts[2] in ("qkv", "attn_out", "wi", "wo"):
            return f"layers.{parts[1]}.{parts[2]}.weight", True
        if parts[3] == "scale" and parts[2] in ("attn_norm", "mlp_norm"):
            return f"layers.{parts[1]}.{parts[2]}.weight", False
    raise ValueError(f"Unknown Flax tensor path: {name}")


def convert(source: Path, output: Path) -> dict:
    source_export = json.loads((source / "export.json").read_text())
    source_weights = source / "model.safetensors"
    source_identity = (source_export["source_model_family"], source_export["source_checkpoint_step"])
    validate_export(source, source_export)
    config = YatEncoderConfig.from_json(source / "config.json")
    with torch.device("meta"):
        expected = YatTorchEncoder(config).state_dict()
    tensors = {}
    with safe_open(source_weights, framework="pt", device="cpu") as reader:
        for name in reader.keys():
            target, transpose = map_name(name)
            if target in tensors or target not in expected:
                raise ValueError(f"Duplicate or unexpected target tensor: {target}")
            value = reader.get_tensor(name)
            if transpose:
                value = value.T.contiguous()
            if tuple(value.shape) != tuple(expected[target].shape) or value.dtype != expected[target].dtype:
                raise ValueError(f"Tensor schema mismatch: {name} -> {target}")
            tensors[target] = value
    if set(tensors) != set(expected) or len(tensors) != source_export["tensors"]:
        raise ValueError("Conversion does not cover the full model state")
    output.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(output / "model.safetensors"), metadata={
        "format": "pt", "source": f"flaxchat-yat-encoder-step-{source_identity[1]}"})
    for filename in ("config.json", "tokenizer.json"):
        shutil.copyfile(source / filename, output / filename)
    if (source / "tokenizer_config.json").is_file():
        shutil.copyfile(source / "tokenizer_config.json", output / "tokenizer_config.json")
    shutil.copyfile(Path(__file__).with_name("yat_encoder.py"), output / "yat_encoder.py")
    for filename, target in (("export.json", "source-export.json"),
                             ("checkpoint-metadata.json", "checkpoint-metadata.json"),
                             ("checkpoint-manifest.json", "checkpoint-manifest.json")):
        shutil.copyfile(source / filename, output / target)
    receipt = {
        "format": "flaxchat-yat-pytorch-v2",
        "artifacts_sha256": artifact_hashes(output),
        "conversion_source_sha256": sha256(Path(__file__)),
        "source_export_sha256": sha256(source / "export.json"),
        "source_artifacts_sha256": source_export["artifacts_sha256"],
        "source_model_family": source_identity[0],
        "source_checkpoint_step": source_identity[1],
        "source_weights_sha256": source_export["sha256"],
        "source_checkpoint_metadata_sha256": source_export["source_checkpoint_metadata_sha256"],
        "torch_weights_sha256": sha256(output / "model.safetensors"),
        "tokenizer_sha256": sha256(output / "tokenizer.json"),
        "tensor_count": len(tensors),
        "bytes": (output / "model.safetensors").stat().st_size,
    }
    (output / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
