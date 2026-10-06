"""INT8 weight-only YAT inference. Arithmetic remains BF16/FP32, not INT8 GEMM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

if __package__:
    from .yat_encoder import YatEncoderConfig, YatTorchEncoder
else:
    from yat_encoder import YatEncoderConfig, YatTorchEncoder

SCHEMA = "flaxchat-yat-int8-v1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Int8Linear(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.register_buffer("qweight", torch.empty(outputs, inputs, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(outputs, 1, dtype=torch.float32))

    @property
    def weight(self) -> torch.Tensor:
        return (self.qweight.float() * self.scale).to(torch.bfloat16)


class Int8Embedding(nn.Module):
    def __init__(self, rows: int, width: int):
        super().__init__()
        self.register_buffer("qweight", torch.empty(rows, width, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(rows, 1, dtype=torch.float32))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Gather first: never materialize the full vocabulary in floating point.
        return F.embedding(ids, self.qweight).float() * F.embedding(ids, self.scale)


def load_quantized(
    directory: str | Path, *, device: str | torch.device
) -> YatTorchEncoder:
    """Load authenticated compressed buffers; model execution requires an accelerator."""
    root = Path(directory)
    receipt = json.loads((root / "quantization.json").read_text())
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("scheme") != "symmetric-int8-per-row-fp32-scale"
    ):
        raise ValueError("Unsupported YAT quantization format")
    expected_files = {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "yat_encoder.py",
        "yat_quantized.py",
    }
    if set(receipt["files_sha256"]) != expected_files:
        raise ValueError("Incomplete quantized artifact identity")
    for name, expected in receipt["files_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    # Bind the implementation actually imported, not merely a copy next to weights.
    for name in ("yat_encoder.py", "yat_quantized.py"):
        if file_sha256(Path(__file__).parent / name) != receipt["files_sha256"][name]:
            raise ValueError(f"Imported implementation differs: {name}")
    source_policy = json.loads((root / "config.json").read_text()).get("weight_quantization", "none")
    if source_policy != "none" and receipt.get("source_weight_quantization") != source_policy:
        raise ValueError("QAT source policy is missing or differs from compressed artifact receipt")
    # Only this authenticated compressed-buffer path may consume QAT configs;
    # ordinary FP32 loading would silently discard their forward quantization.
    config = YatEncoderConfig.from_json(root / "config.json", compressed_int8=True)
    with torch.device("meta"):
        model = YatTorchEncoder(config)
        model.embedding = Int8Embedding(config.vocab_size, config.hidden_size)
        for parent in model.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    if child.bias is not None:
                        raise ValueError(
                            "Only bias-free released YAT linears are supported"
                        )
                    setattr(
                        parent, name, Int8Linear(child.in_features, child.out_features)
                    )
    state = load_file(str(root / "model.safetensors"), device="cpu")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Quantized state names differ from architecture")
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Quantized tensor shape/dtype mismatch: {name}")
        # Host-side artifact validation only; no CPU model execution.
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite tensor: {name}")
        if name.endswith(".scale") and not bool((value > 0).all()):
            raise ValueError(f"Nonpositive quantization scale: {name}")
        if name.endswith(".qweight") and bool((value == -128).any()):
            raise ValueError(f"Out-of-range symmetric INT8 tensor: {name}")
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False)
    return model.to(device).eval()
