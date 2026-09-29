"""Accelerator-only JAX/PyTorch YAT encoder parity, in separate processes.

JAX and torch_xla own incompatible libtpu runtimes, so the two forward passes
must run sequentially in separate Python environments on a TPU VM. This script
refuses a CPU backend for either forward pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def prepare_inputs(path: Path, tokenizer_path: Path) -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    examples = [
        "Searching multilingual documents for the correct answer. " * 22,
        "Find the implementation of a function by its behavior. " * 9,
        "ابحث عن المستند الذي يجيب عن السؤال بدقة.",
        "कोड में सही फ़ंक्शन और उसकी परिभाषा खोजें।",
        "在代码库中查找实现此功能的文件和函数。",
        "Rechercher la fonction qui traite cette erreur dans le dépôt.",
        "def search_files(query): return [path for path in paths if query in path]",
        "Short query",
    ]
    tokens = np.zeros((len(examples), 128), dtype=np.int32)
    for row, text in enumerate(examples):
        ids = tokenizer.encode(text).ids[:125]
        tokens[row, :len(ids)] = ids
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens)


def run_jax(model_dir: Path, input_path: Path, output: Path) -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flaxchat.public_encoder import load_public_encoder

    if jax.default_backend() != "tpu":
        raise RuntimeError("JAX parity forward requires a physical TPU")
    model = load_public_encoder(model_dir)
    ids = jnp.asarray(np.load(input_path))
    segments = jnp.where(ids == model.config.pad_token_id, -1, 0)
    positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)

    @nnx.jit
    def forward(model, tokens, segs, pos):
        x = model.embedding_norm(model.embedding(tokens).astype(jnp.float32))
        embedding = x
        first = None
        middle = None
        for i, layer in enumerate(model.layers):
            x = layer(x, segs, pos, packed=False)
            if i == 0:
                first = x
            if i == 10:
                middle = x
        hidden = jnp.where((segs >= 0)[..., None], model.final_norm(x), 0)
        pool = hidden.sum(axis=1) / jnp.maximum((segs >= 0).sum(axis=1, keepdims=True), 1)
        return embedding, first, middle, hidden, pool

    outputs = forward(model, ids, segments, positions)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **{name: np.asarray(value) for name, value in zip(
        ("embedding", "layer_0", "layer_10", "hidden", "pool"), outputs, strict=True)})


def run_torch(model_dir: Path, input_path: Path, output: Path) -> None:
    import torch
    import torch_xla
    from torch_port.yat_encoder import YatTorchEncoder, _norm

    torch_xla.runtime.set_device_type("TPU")
    device = torch_xla.device()
    if device.type != "xla" or not any("TPU" in d for d in torch_xla.real_devices()):
        raise RuntimeError("PyTorch parity forward requires a physical TPU")
    model = YatTorchEncoder.from_pretrained(model_dir, device=device)
    ids = torch.as_tensor(np.load(input_path), device=device, dtype=torch.long)
    segs = (ids != model.config.pad_token_id).long() - 1
    pos = torch.arange(ids.shape[1], device=device)[None, :].expand_as(ids)
    with torch.no_grad():
        x = _norm(model.embedding_norm, model.embedding(ids))
        embedding = x
        first = None
        middle = None
        for i, layer in enumerate(model.layers):
            x = layer(x, segs, pos)
            if i == 0:
                first = x
            if i == 10:
                middle = x
        hidden = torch.where((segs >= 0)[..., None], _norm(model.final_norm, x), 0)
        pool = hidden.sum(dim=1) / (segs >= 0).sum(dim=1, keepdim=True).clamp_min(1)
        torch_xla.sync()
        values = [value.cpu().numpy() for value in (embedding, first, middle, hidden, pool)]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **dict(zip(
        ("embedding", "layer_0", "layer_10", "hidden", "pool"), values, strict=True)))


def compare(jax_path: Path, torch_path: Path, output: Path) -> dict:
    reference, candidate = np.load(jax_path), np.load(torch_path)
    if set(reference.files) != set(candidate.files):
        raise ValueError("Parity output keys differ")
    metrics = {}
    for key in reference.files:
        a = reference[key].astype(np.float32)
        b = candidate[key].astype(np.float32)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid parity output: {key}")
        delta = np.abs(a - b)
        norm_product = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        cosine = np.sum(a * b, axis=-1) / np.maximum(norm_product, 1e-12)
        cosine = cosine[norm_product > 1e-12]
        metrics[key] = {
            "shape": list(a.shape),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "p99_abs": float(np.quantile(delta, .99)),
            "min_vector_cosine": float(cosine.min()),
            "mean_vector_cosine": float(cosine.mean()),
        }
    a = reference["pool"].astype(np.float32)
    b = candidate["pool"].astype(np.float32)
    an = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    sa, sb = an @ an.T, bn @ bn.T
    np.fill_diagonal(sa, -np.inf)
    np.fill_diagonal(sb, -np.inf)
    top_jax = np.argsort(-sa, axis=-1)[:, :3]
    top_torch = np.argsort(-sb, axis=-1)[:, :3]
    retrieval = {
        "max_cosine_score_drift": float(np.max(np.abs(sa[np.isfinite(sa)] - sb[np.isfinite(sb)]))),
        "top_1_agreement": float(np.mean(top_jax[:, 0] == top_torch[:, 0])),
        "top_3_set_agreement": float(np.mean([
            set(left) == set(right) for left, right in zip(top_jax, top_torch, strict=True)])),
    }
    report = {"jax_backend": "physical TPU", "torch_backend": "torch_xla TPU",
              "metrics": metrics, "retrieval": retrieval}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("inputs")
    a.add_argument("tokenizer", type=Path)
    a.add_argument("output", type=Path)
    for mode in ("jax", "torch"):
        a = sub.add_parser(mode)
        a.add_argument("model", type=Path)
        a.add_argument("inputs", type=Path)
        a.add_argument("output", type=Path)
    a = sub.add_parser("compare")
    a.add_argument("jax_output", type=Path)
    a.add_argument("torch_output", type=Path)
    a.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.mode == "inputs":
        prepare_inputs(args.output, args.tokenizer)
    elif args.mode == "jax":
        run_jax(args.model, args.inputs, args.output)
    elif args.mode == "torch":
        run_torch(args.model, args.inputs, args.output)
    else:
        print(json.dumps(compare(args.jax_output, args.torch_output, args.report), indent=2))


if __name__ == "__main__":
    main()
