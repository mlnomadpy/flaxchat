"""Export authenticated PyTorch YAT weights as real INT8 storage, without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

SCHEMA = "flaxchat-yat-int8-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantize_rows(array, *, chunk_rows=1024):
    """Symmetric signed INT8, nearest-even rounding, one FP32 scale per row."""
    if array.ndim != 2 or array.dtype != np.float32 or chunk_rows < 1:
        raise ValueError("Expected an FP32 matrix and a positive chunk size")
    quantized = np.empty(array.shape, dtype=np.int8)
    scales = np.empty((array.shape[0], 1), dtype=np.float32)
    max_error = 0.0
    for start in range(0, array.shape[0], chunk_rows):
        rows = array[start : start + chunk_rows]
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite weights")
        maximum = np.max(np.abs(rows), axis=1, keepdims=True)
        scale = np.maximum(maximum / np.float32(127), np.finfo(np.float32).tiny)
        scale = np.where(maximum == 0, np.float32(1), scale)
        q = np.clip(np.rint(rows / scale), -127, 127).astype(np.int8)
        quantized[start : start + len(rows)] = q
        scales[start : start + len(rows)] = scale
        max_error = max(
            max_error, float(np.max(np.abs(rows - q.astype(np.float32) * scale)))
        )
    return quantized, scales, max_error


def export(source, output, *, expected_sha256):
    source, output = Path(source), Path(output)
    parent = source / "model.safetensors"
    if digest(parent) != expected_sha256:
        raise ValueError("Parent full weight SHA256 mismatch")
    config = json.loads((source / "config.json").read_text())
    if config.get("yat_bias") != 1 or config.get("yat_epsilon") != 0.01:
        raise ValueError("YAT fixed constants must remain bias=1, epsilon=0.01")
    source_policy = config.get("weight_quantization", "none")
    if source_policy not in ("none", "int8_per_channel_ste"):
        raise ValueError("Unsupported source weight quantization policy")
    output.mkdir(exist_ok=False, parents=True)
    state, tensors = {}, {}
    with safe_open(parent, framework="np") as weights:
        for name in weights.keys():
            tensor = weights.get_tensor(name)
            if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
                raise ValueError(f"Expected finite released FP32 weights: {name}")
            if tensor.ndim == 2:
                if not name.endswith(".weight"):
                    raise ValueError(f"Unsupported matrix name: {name}")
                q, scales, error = quantize_rows(tensor)
                prefix = name[: -len("weight")]
                state[prefix + "qweight"] = q
                state[prefix + "scale"] = scales
                tensors[name] = {
                    "shape": list(tensor.shape),
                    "maximum_weight_error": error,
                    "quantized": True,
                }
            else:
                state[name] = tensor
                tensors[name] = {"shape": list(tensor.shape), "quantized": False}
    save_file(state, output / "model.safetensors", metadata={"format": SCHEMA})
    for name in ("config.json", "tokenizer.json"):
        shutil.copyfile(source / name, output / name)
    implementation = Path(__file__).resolve().parents[1] / "torch_port"
    for name in ("yat_encoder.py", "yat_quantized.py"):
        shutil.copyfile(implementation / name, output / name)
    files = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "yat_encoder.py",
        "yat_quantized.py",
    )
    receipt = {
        "schema": SCHEMA,
        "scheme": "symmetric-int8-per-row-fp32-scale",
        "source_weight_quantization": source_policy,
        "weight_storage": "int8-matrices-with-fp32-scales-and-unquantized-vectors",
        "source_weight_storage": "fp32-masters",
        "parent_weights_sha256": expected_sha256,
        "parent_files_sha256": {
            n: digest(source / n) for n in ("config.json", "tokenizer.json")
        },
        "exporter_sha256": digest(Path(__file__)),
        "files_sha256": {n: digest(output / n) for n in files},
        "parent_weight_bytes": parent.stat().st_size,
        "quantized_weight_bytes": (output / "model.safetensors").stat().st_size,
        "tensors": tensors,
        "arithmetic": "BF16 features/distance, FP32 residual/scores/scales; no INT8 GEMM",
        "physical_tpu_validated": False,
    }
    (output / "quantization.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    args = p.parse_args()
    receipt = export(args.source, args.output, expected_sha256=args.expected_sha256)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "parent_weight_bytes",
                    "quantized_weight_bytes",
                    "physical_tpu_validated",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
