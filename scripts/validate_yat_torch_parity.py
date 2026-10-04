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
from scripts.release_contract import VERSION, ARTIFACTS, artifact_hashes, digest, validate_metrics, intermediate_scope, metric_names, validate_runtime_identity
from scripts.evaluation_contract import provenance, write_atomic


def prepare_inputs(path: Path, tokenizer_path: Path, length: int = 128) -> None:
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
    if length < 32:
        raise ValueError("Parity length must be at least 32")
    tokenizer.no_truncation()
    tokenizer.no_padding()
    # Separate queries from a larger document corpus; include multilingual/code,
    # long truncated documents, empty strings and a fully padded edge row.
    examples += [examples[i % 8] + f" Document {i}: " + ("context " * (length + i))
                 for i in range(62)] + ["", ""]
    config_path = tokenizer_path.with_name("config.json")
    pad = json.loads(config_path.read_text())["pad_token_id"]
    tokens = np.full((len(examples), length), pad, dtype=np.int32)
    for row, text in enumerate(examples):
        ids = tokenizer.encode(text).ids[:length]
        tokens[row, :len(ids)] = ids
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens[-1, :] = pad
    np.save(path, tokens)
    write_atomic(Path(str(path) + ".json"), {"sequence_length": length, "rows": len(examples),
                 "retrieval_protocol": "8-query/64-corpus-v1", "inputs_sha256": digest(path),
                 "fixture": "multilingual/code/long/empty/all-padding-v2"})


def run_jax(model_dir: Path, input_path: Path, output: Path, batch_size: int = 8) -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flaxchat.public_encoder import load_public_encoder

    current_identity("jax", model_dir)
    if jax.default_backend() != "tpu":
        raise RuntimeError("JAX parity forward requires a physical TPU")
    model = load_public_encoder(model_dir)
    indices = intermediate_scope(json.loads((model_dir / "config.json").read_text()))["intermediate_indices"]
    ids = jnp.asarray(np.load(input_path))
    segments = jnp.where(ids == model.config.pad_token_id, -1, 0)
    positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)

    @nnx.jit
    def forward(model, tokens, segs, pos):
        x = model.embedding_norm(model.embedding(tokens).astype(jnp.float32))
        embedding = x
        intermediates = []
        for i, layer in enumerate(model.layers):
            x = layer(x, segs, pos, packed=False)
            if i in indices:
                intermediates.append(x)
        hidden = jnp.where((segs >= 0)[..., None], model.final_norm(x), 0)
        pool = hidden.sum(axis=1) / jnp.maximum((segs >= 0).sum(axis=1, keepdims=True), 1)
        return (embedding, *intermediates, hidden, pool)

    if batch_size < 1:
        raise ValueError("Positive batch size required")
    chunks = [forward(model, ids[i:i + batch_size], segments[i:i + batch_size], positions[i:i + batch_size])
              for i in range(0, len(ids), batch_size)]
    outputs = [np.concatenate([np.asarray(chunk[index]) for chunk in chunks]) for index in range(len(indices) + 3)]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **{name: np.asarray(value, dtype=np.float32) for name, value in zip(
        ("embedding", *(f"layer_{i}" for i in indices), "hidden", "pool"), outputs, strict=True)})
    _run_receipt("jax", model_dir, input_path, output, batch_size,
                 [{"kind": d.device_kind, "platform": d.platform} for d in jax.devices()])


def run_torch(model_dir: Path, input_path: Path, output: Path, batch_size: int = 8) -> None:
    import torch
    import torch_xla
    from torch_port.yat_encoder import YatTorchEncoder, _norm

    current_identity("torch", model_dir)
    torch_xla.runtime.set_device_type("TPU")
    device = torch_xla.device()
    if device.type != "xla" or not any("TPU" in d for d in torch_xla.real_devices()):
        raise RuntimeError("PyTorch parity forward requires a physical TPU")
    model = YatTorchEncoder.from_pretrained(model_dir, device=device)
    indices = intermediate_scope(json.loads((model_dir / "config.json").read_text()))["intermediate_indices"]
    ids = torch.as_tensor(np.load(input_path), device=device, dtype=torch.long)
    segs = (ids != model.config.pad_token_id).long() - 1
    pos = torch.arange(ids.shape[1], device=device)[None, :].expand_as(ids)
    def forward(ids, segs, pos):
        x = _norm(model.embedding_norm, model.embedding(ids))
        embedding = x
        intermediates = []
        for i, layer in enumerate(model.layers):
            x = layer(x, segs, pos)
            if i in indices:
                intermediates.append(x)
        hidden = torch.where((segs >= 0)[..., None], _norm(model.final_norm, x), 0)
        pool = hidden.sum(dim=1) / (segs >= 0).sum(dim=1, keepdim=True).clamp_min(1)
        torch_xla.sync()
        return [value.float().cpu().numpy() for value in (embedding, *intermediates, hidden, pool)]
    if batch_size < 1:
        raise ValueError("Positive batch size required")
    with torch.no_grad():
        chunks = [forward(ids[i:i + batch_size], segs[i:i + batch_size], pos[i:i + batch_size])
                  for i in range(0, len(ids), batch_size)]
    values = [np.concatenate([chunk[index] for chunk in chunks]) for index in range(len(indices) + 3)]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **dict(zip(
        ("embedding", *(f"layer_{i}" for i in indices), "hidden", "pool"), values, strict=True)))
    _run_receipt("torch", model_dir, input_path, output, batch_size,
                 [{"kind": kind} for kind in torch_xla.real_devices()])


def current_identity(backend, model_dir):
    """Import numerical settings only; no model construction or forward."""
    if backend == "jax":
        import importlib
        importlib.import_module('jax')
    elif backend == "torch":
        import torch_xla
        torch_xla.runtime.set_device_type("TPU")
    else:
        raise ValueError("Unknown parity backend")
    sources = [Path(__file__), Path("scripts/release_contract.py"), Path("scripts/evaluation_contract.py"),
               Path("scripts/run_yat_parity_case.py"), Path("scripts/run_yat_parity_campaign.py")]
    if backend == "jax":
        sources += [Path("flaxchat/public_encoder.py"), Path("flaxchat/encoder.py"), Path("flaxchat/yat.py"),
                    Path("flaxchat/yat_attention.py"), Path("flaxchat/yat_attention_centered.py")]
    else:
        sources += [Path("torch_port/yat_encoder.py")]
    protocol = provenance(sources, devices=[], batch_size=1)
    runtime = {key: protocol[key] for key in ("runtime_packages", "interpreter", "numeric_environment", "effective_jax_config", "effective_torch_config", "deployment_receipts_sha256")}
    validate_runtime_identity(runtime)
    return {"runtime": runtime, "source_sha256": protocol["source_sha256"],
            "model_sha256": artifact_hashes(model_dir, ARTIFACTS if backend == "torch" else ARTIFACTS[:-1])}


def _run_receipt(backend, model_dir, input_path, output, batch_size, devices):
    fixture = json.loads(Path(str(input_path) + ".json").read_text())
    if fixture["inputs_sha256"] != digest(input_path):
        raise ValueError("Fixture input identity changed")
    names = ("model.safetensors", "config.json", "tokenizer.json")
    if backend == "torch":
        names += ("yat_encoder.py",)
        import sys
        actual_implementation = Path(sys.modules["torch_port.yat_encoder"].__file__)
        if digest(actual_implementation) != digest(model_dir / "yat_encoder.py"):
            raise ValueError("Torch runtime implementation differs from shipped release")
    identity = current_identity(backend, model_dir)
    runtime = identity["runtime"]
    record = {"format": VERSION, "backend": backend, "model_sha256": artifact_hashes(model_dir, names),
              "inputs_sha256": digest(input_path), "output_sha256": digest(output),
              "scope": {**fixture, "batch_size": batch_size, **intermediate_scope(json.loads((model_dir / "config.json").read_text()))}, "devices": devices,
              "runtime": runtime, "source_sha256": identity["source_sha256"]}
    if backend == "torch":
        record["conversion_sha256"] = digest(model_dir / "conversion.json")
    write_atomic(Path(str(output) + ".json"), record)


def validate_output_arrays(reference, candidate, scope):
    required = metric_names(scope)
    if set(reference.files) != required or set(candidate.files) != required:
        raise ValueError("Parity outputs must contain every declared layer")
    rows, length = scope["rows"], scope["sequence_length"]
    if rows != 72 or type(length) is not int or length < 32:
        raise ValueError("Unexpected parity fixture shape")
    width = reference["pool"].shape[-1]
    for key in required:
        a, b = reference[key], candidate[key]
        expected = (rows, width) if key == "pool" else (rows, length, width)
        if (a.dtype != np.float32 or b.dtype != np.float32 or a.shape != expected or b.shape != expected
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError(f"Parity output differs from frozen fixture shape: {key}")
        left = np.linalg.norm(a.astype(np.float64), axis=-1)
        right = np.linalg.norm(b.astype(np.float64), axis=-1)
        if np.any((left <= 1e-12) != (right <= 1e-12)):
            raise ValueError(f"Zero/nonzero parity mismatch: {key}")
        if not np.any(left > 1e-12):
            raise ValueError(f"Collapsed parity outputs: {key}")
        if key == "pool" and (np.any(left[:-1] <= 1e-12) or left[-1] > 1e-12):
            raise ValueError("Only the declared all-padding row may have a zero pooled vector")


def compare(jax_path: Path, torch_path: Path, output: Path) -> dict:
    runs = {name: json.loads(Path(str(path) + ".json").read_text())
            for name, path in (("jax", jax_path), ("torch", torch_path))}
    for backend, path in (("jax", jax_path), ("torch", torch_path)):
        if runs[backend]["backend"] != backend or runs[backend]["output_sha256"] != digest(path):
            raise ValueError("Output/backend receipt mismatch")
    shared = set(runs["jax"]["source_sha256"]) & set(runs["torch"]["source_sha256"])
    evaluator_keys = [key for key in shared if key == 'scripts/validate_yat_torch_parity.py'
                      or key.endswith('/scripts/validate_yat_torch_parity.py')]
    if (len(evaluator_keys) != 1 or runs['jax']['source_sha256'][evaluator_keys[0]] != digest(Path(__file__))
            or any(runs["jax"]["source_sha256"][key] != runs["torch"]["source_sha256"][key] for key in shared)):
        raise ValueError("Parity evaluator/contract implementation changed between runs")
    for run in runs.values():
        validate_runtime_identity(run["runtime"])
    if runs["jax"]["scope"] != runs["torch"]["scope"]:
        raise ValueError("Parity input/batch fixture differs")
    reference, candidate = np.load(jax_path), np.load(torch_path)
    validate_output_arrays(reference, candidate, runs["jax"]["scope"])
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
            "min_vector_cosine": float(cosine.min()) if cosine.size else 1.,
            "mean_vector_cosine": float(cosine.mean()) if cosine.size else 1.,
        }
    a = reference["pool"].astype(np.float32)
    b = candidate["pool"].astype(np.float32)
    an = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    if len(an) != 72:
        raise ValueError("Expanded 8-query/64-corpus fixture required")
    sa, sb = an[:8] @ an[8:].T, bn[:8] @ bn[8:].T
    top_jax = np.argsort(-sa, axis=-1)[:, :3]
    top_torch = np.argsort(-sb, axis=-1)[:, :3]
    retrieval = {
        "max_cosine_score_drift": float(np.max(np.abs(sa[np.isfinite(sa)] - sb[np.isfinite(sb)]))),
        "top_1_agreement": float(np.mean(top_jax[:, 0] == top_torch[:, 0])),
        "top_3_set_agreement": float(np.mean([
            set(left) == set(right) for left, right in zip(top_jax, top_torch, strict=True)])),
    }
    report = {"format": VERSION, "runs": runs, "scope": runs["jax"]["scope"],
              "conversion_sha256": runs["torch"]["conversion_sha256"],
              "artifacts_sha256": runs["torch"]["model_sha256"], "jax_backend": "physical TPU", "torch_backend": "torch_xla TPU",
              "metrics": metrics, "retrieval": retrieval}
    validate_metrics(report)
    write_atomic(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("inputs")
    a.add_argument("tokenizer", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--sequence-length", type=int, default=128)
    a = sub.add_parser("identity")
    a.add_argument("backend", choices=("jax", "torch"))
    a.add_argument("model", type=Path)
    a.add_argument("output", type=Path)
    for mode in ("jax", "torch"):
        a = sub.add_parser(mode)
        a.add_argument("model", type=Path)
        a.add_argument("inputs", type=Path)
        a.add_argument("output", type=Path)
        a.add_argument("--batch-size", type=int, default=8)
    a = sub.add_parser("matrix")
    a.add_argument("reports", type=Path, nargs="+")
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("compare")
    a.add_argument("jax_output", type=Path)
    a.add_argument("torch_output", type=Path)
    a.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.mode == "identity":
        write_atomic(args.output, current_identity(args.backend, args.model))
    elif args.mode == "inputs":
        prepare_inputs(args.output, args.tokenizer, args.sequence_length)
    elif args.mode == "jax":
        run_jax(args.model, args.inputs, args.output, args.batch_size)
    elif args.mode == "torch":
        run_torch(args.model, args.inputs, args.output, args.batch_size)
    elif args.mode == "matrix":
        cases = [json.loads(path.read_text()) for path in args.reports]
        for case in cases:
            validate_metrics(case)
        if not cases or any(case["conversion_sha256"] != cases[0]["conversion_sha256"]
                            or case["artifacts_sha256"] != cases[0]["artifacts_sha256"] for case in cases):
            raise ValueError("Mixed parity matrix")
        write_atomic(args.output, {"format": VERSION, "cases": cases,
                     "conversion_sha256": cases[0]["conversion_sha256"],
                     "artifacts_sha256": cases[0]["artifacts_sha256"]})
    else:
        print(json.dumps(compare(args.jax_output, args.torch_output, args.report), indent=2))


if __name__ == "__main__":
    main()
