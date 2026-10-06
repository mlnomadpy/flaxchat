"""Paired INT8 weight-perturbation evaluation on a physical TPU, native JAX only.

This does not qualify the Torch compressed loader, INT8 GEMM or compressed HBM.
Host arithmetic only reconstructs authenticated artifact bytes; forwards are TPU-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

from scripts.benchmark_embedding_speed_tpu import ROOT, digest, write_report


def torch_name(name):
    """Pure metadata equivalent of convert_yat_encoder.map_name (no Torch import)."""
    parts = re.findall(r"\['([^']+)'\]", name)
    if "".join(f"['{p}']" for p in parts) != name:
        raise ValueError(f"Invalid canonical path: {name}")
    if parts == ["decoder_bias"]:
        return "decoder_bias", False
    if parts == ["embedding", "embedding"]:
        return "embedding.weight", False
    if (
        len(parts) == 2
        and parts[0] in ("embedding_norm", "final_norm", "head_norm")
        and parts[1] == "scale"
    ):
        return f"{parts[0]}.weight", False
    if parts == ["head_dense", "kernel"]:
        return "head_dense.weight", True
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        prefix = f"layers.{parts[1]}."
        if len(parts) == 3 and parts[2] in ("yat_alpha", "yat_attention_alpha"):
            return prefix + parts[2], False
        if (
            len(parts) == 4
            and parts[2] in ("qkv", "attn_out", "wi", "wo")
            and parts[3] == "kernel"
        ):
            return prefix + parts[2] + ".weight", True
        if (
            len(parts) == 4
            and parts[2] in ("attn_norm", "mlp_norm")
            and parts[3] == "scale"
        ):
            return prefix + parts[2] + ".weight", False
    raise ValueError(f"Unknown tensor: {name}")


def reconstructed_matrix(reader, name, *, scale_name=None, transpose=False):
    """One FP32 output matrix, bounded 1024-row scratch; never full FP32 copies."""
    import numpy as np

    view = reader.get_slice(name)
    rows, width = view.get_shape()
    result = np.empty((width, rows) if transpose else (rows, width), np.float32)
    scales = reader.get_slice(scale_name) if scale_name else None
    if scales is not None and (
        scales.get_shape() != [rows, 1]
        or scales.get_dtype() != "F32"
        or view.get_dtype() != "I8"
    ):
        raise ValueError(f"Invalid INT8 matrix schema: {name}")
    if scales is None and view.get_dtype() != "F32":
        raise ValueError(f"Expected FP32 original: {name}")
    for start in range(0, rows, 1024):
        end = min(rows, start + 1024)
        values = view[start:end]
        if scales is not None:
            scale = scales[start:end]
            if (
                not np.isfinite(scale).all()
                or np.any(scale <= 0)
                or np.any(values == -128)
            ):
                raise ValueError(f"Invalid quantized values: {name}")
            dequantized = values.astype(np.float32) * scale
        else:
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite original: {name}")
            maximum = np.max(np.abs(values), axis=1, keepdims=True)
            scale = np.maximum(maximum / np.float32(127), np.finfo(np.float32).tiny)
            scale = np.where(maximum == 0, np.float32(1), scale)
            q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
            dequantized = q.astype(np.float32) * scale
        if not np.isfinite(dequantized).all():
            raise ValueError(f"Nonfinite reconstructed weights: {name}")
        if transpose:
            result[:, start:end] = dequantized.T
        else:
            result[start:end] = dequantized
    return result


def authenticate_quantized(root, original, parent_sha):
    receipt = json.loads((root / "quantization.json").read_text())
    expected = {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "yat_encoder.py",
        "yat_quantized.py",
    }
    if (
        receipt.get("schema") != "flaxchat-yat-int8-v1"
        or receipt.get("scheme") != "symmetric-int8-per-row-fp32-scale"
        or set(receipt.get("files_sha256", {})) != expected
    ):
        raise ValueError("Incomplete/unsupported quantization manifest")
    for name, value in receipt["files_sha256"].items():
        if digest(root / name) != value:
            raise ValueError(f"Quantized hash mismatch: {name}")
    for name in ("config.json", "tokenizer.json"):
        if digest(original / name) != receipt["parent_files_sha256"][name] or digest(
            root / name
        ) != digest(original / name):
            raise ValueError(f"Parent mismatch: {name}")
    # The compressed manifest pins Torch-layout parent bytes, distinct from the
    # canonical Flax original. Conversion receipt links both independently.
    conversion = json.loads((original / "torch-conversion.json").read_text())
    if (
        conversion["source_weights_sha256"] != parent_sha
        or conversion["torch_weights_sha256"] != receipt["parent_weights_sha256"]
    ):
        raise ValueError("INT8 candidate does not descend from pinned original")
    config = json.loads((root / "config.json").read_text())
    if config.get("yat_bias") != 1 or config.get("yat_epsilon") != 0.01:
        raise ValueError("YAT constants changed")
    return receipt, conversion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("yat", "mmbert"), required=True)
    p.add_argument("--model-directory", type=Path, required=True)
    p.add_argument("--quantized-directory", type=Path)
    p.add_argument("--expected-weights-sha256", required=True)
    p.add_argument("--expected-config-sha256", required=True)
    p.add_argument("--expected-tokenizer-sha256", required=True)
    p.add_argument("--expected-quantization-manifest-sha256")
    p.add_argument("--expected-torch-conversion-sha256")
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--fixture-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if any(
        not re.fullmatch("[0-9a-f]{64}", value)
        for value in (
            a.expected_weights_sha256,
            a.expected_config_sha256,
            a.expected_tokenizer_sha256,
            a.fixture_sha256,
        )
    ) or (a.role == "yat") != (a.quantized_directory is not None):
        p.error("Exact SHA256 pins required; quantized directory required only for YAT")
    if a.role == "yat" and any(
        not value or not re.fullmatch("[0-9a-f]{64}", value)
        for value in (
            a.expected_quantization_manifest_sha256,
            a.expected_torch_conversion_sha256,
        )
    ):
        p.error("YAT requires quantization manifest and Torch conversion SHA256 pins")
    npz = a.output.with_suffix(".npz")
    if a.output.exists() or npz.exists():
        p.error("Previous receipts must not be overwritten")
    report = {
        "format": "flaxchat-int8-native-tpu-quality-v1",
        "complete": False,
        "role": a.role,
        "scope": "native-JAX reconstruction of per-row INT8 weights; not Torch-loader/HBM/GEMM qualification",
        "scheme": "symmetric-int8-per-Torch-output-row-fp32-scale",
        "fixture_sha256": a.fixture_sha256,
        "batch_size": 8,
        "sequence_length": 256,
        "pooling": "mean nonpadding hidden states including special tokens",
        "normalization": "raw pools saved; downstream cosine uses FP32 L2",
        "batches": [],
    }
    write_report(a.output, report)
    originals, candidates = [], []
    try:
        import jax
        import numpy as np
        from flax import nnx
        from safetensors import safe_open
        from tokenizers import Tokenizer
        from flaxchat.checkpoint import _canonical_manifest_paths
        from flaxchat.encoder import EncoderConfig, ModernBert
        from flaxchat.public_encoder import load_public_encoder
        from scripts.train_encoder import load_pretrained

        if (
            jax.default_backend() != "tpu"
            or jax.process_count() != 1
            or any(d.platform != "tpu" for d in jax.local_devices())
        ):
            raise RuntimeError(
                "Physical single-host TPU required; CPU fallback forbidden"
            )
        device = jax.local_devices()[0]
        if (
            a.fixture.stat().st_size > 32 * 1024 * 1024
            or digest(a.fixture) != a.fixture_sha256
        ):
            raise ValueError("Fixture identity/size mismatch")
        rows = json.loads(a.fixture.read_text())
        if (
            not isinstance(rows, list)
            or not 2 <= len(rows) <= 10000
            or any(
                not isinstance(r, dict)
                or not isinstance(r.get("id"), str)
                or not isinstance(r.get("text"), str)
                or not r["text"]
                for r in rows
            )
            or len({r["id"] for r in rows}) != len(rows)
        ):
            raise ValueError("Require 2..10000 unique typed text rows")
        for filename, expected in (
            ("config.json", a.expected_config_sha256),
            ("tokenizer.json", a.expected_tokenizer_sha256),
        ):
            if digest(a.model_directory / filename) != expected:
                raise ValueError(f"Parent hash mismatch: {filename}")
        source_weight = "model.safetensors" if a.role == "yat" else "pytorch_model.bin"
        if digest(a.model_directory / source_weight) != a.expected_weights_sha256:
            raise ValueError("Published parent weight SHA mismatch")
        if a.role == "mmbert":
            from scripts.convert_encoder_checkpoint import convert

            conversion = (
                convert(a.model_directory)
                if not (a.model_directory / "model.safetensors").exists()
                else json.loads((a.model_directory / "conversion.json").read_text())
            )
            if conversion["source_sha256"] != a.expected_weights_sha256 or conversion[
                "safetensors_sha256"
            ] != digest(a.model_directory / "model.safetensors"):
                raise ValueError("mmBERT conversion identity mismatch")
            report["conversion"] = conversion
        else:
            if (
                digest(a.quantized_directory / "quantization.json")
                != a.expected_quantization_manifest_sha256
                or digest(a.model_directory / "torch-conversion.json")
                != a.expected_torch_conversion_sha256
            ):
                raise ValueError("Quantization/conversion manifest hash mismatch")
            report["quantization_manifest_sha256"] = (
                a.expected_quantization_manifest_sha256
            )
            report["torch_conversion_sha256"] = a.expected_torch_conversion_sha256
            receipt, conversion = authenticate_quantized(
                a.quantized_directory, a.model_directory, a.expected_weights_sha256
            )
            report["quantization"] = receipt
            report["conversion"] = conversion
        report["artifacts_sha256"] = {
            name: digest(a.model_directory / name)
            for name in {
                source_weight,
                "model.safetensors",
                "config.json",
                "tokenizer.json",
            }
        }
        report["source_sha256"] = {
            str(path.relative_to(ROOT)): digest(path)
            for path in sorted(ROOT.glob("flaxchat/**/*.py"))
            + [
                Path(__file__),
                ROOT / "scripts/train_encoder.py",
                ROOT / "scripts/convert_encoder_checkpoint.py",
            ]
        }
        report["runtime"] = {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "jax",
                    "jaxlib",
                    "flax",
                    "numpy",
                    "safetensors",
                    "tokenizers",
                    "libtpu",
                )
            },
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_default_matmul_precision": str(
                jax.config.jax_default_matmul_precision
            ),
        }
        report["hardware"] = {
            "device": str(device),
            "kind": device.device_kind,
            "utilized_devices": 1,
            "available_devices": len(jax.local_devices()),
        }
        report["rows"] = [{k: v for k, v in row.items() if k != "text"} for row in rows]
        tokenizer = Tokenizer.from_file(str(a.model_directory / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        encoded = tokenizer.encode_batch([row["text"] for row in rows])
        with jax.default_device(device):
            if a.role == "yat":
                model = load_public_encoder(a.model_directory)
            else:
                config = EncoderConfig.from_hf(
                    json.loads((a.model_directory / "config.json").read_text()),
                    compute_dtype="bfloat16",
                    residual_dtype="float32",
                    attention_backend="xla",
                )
                model = ModernBert(config, rngs=nnx.Rngs(0))
                load_pretrained(model, a.model_directory)
            c = model.config
            if (
                c.compute_dtype != "bfloat16"
                or c.residual_dtype != "float32"
                or c.yat_local_shards
            ):
                raise ValueError("Require matched BF16/FP32 unsharded model policy")
            report["encoder_config"] = asdict(c)
            tokens = np.full(((len(rows) + 7) // 8 * 8, 256), c.pad_token_id, np.int32)
            for i, item in enumerate(encoded):
                tokens[i, : min(len(item.ids), 256)] = item.ids[:256]
            # Dummy final-batch rows duplicate the last valid input, then discard.
            tokens[len(rows) :] = tokens[len(rows) - 1]
            report["token_counts"] = [len(item.ids) for item in encoded]
            report["truncated_rows"] = sum(len(item.ids) > 256 for item in encoded)
            report["input_int32_sha256"] = hashlib.sha256(tokens.tobytes()).hexdigest()

            @nnx.jit
            def pool(m, ids):
                return m.pool(ids)

            def run(label, output):
                for start in range(0, len(rows), 8):
                    begin = time.monotonic()
                    values = np.asarray(
                        pool(model, jax.device_put(tokens[start : start + 8], device))
                    )[: min(8, len(rows) - start)].copy()
                    if not np.isfinite(values).all() or np.any(
                        np.linalg.norm(values, axis=1) <= 1e-12
                    ):
                        raise ValueError(f"Invalid {label} embedding")
                    output.append(values)
                    if start % 64 == 0 or start + 8 >= len(rows):
                        np.savez(
                            npz,
                            original=np.concatenate(originals)
                            if originals
                            else np.empty((0, c.hidden_size), np.float32),
                            quantized=np.concatenate(candidates)
                            if candidates
                            else np.empty((0, c.hidden_size), np.float32),
                        )
                        report["persisted_original_rows"] = sum(
                            len(v) for v in originals
                        )
                        report["persisted_quantized_rows"] = sum(
                            len(v) for v in candidates
                        )
                    report["batches"].append(
                        {
                            "variant": label,
                            "start": start,
                            "rows": len(values),
                            "seconds_including_compile_and_transfer": time.monotonic()
                            - begin,
                        }
                    )
                    report["pooled_npz_sha256"] = digest(npz)
                    write_report(a.output, report)

            first_ids = jax.device_put(tokens[:8], device)
            repeat_a = np.asarray(pool(model, first_ids))
            repeat_b = np.asarray(pool(model, first_ids))
            if not np.array_equal(repeat_a, repeat_b):
                raise ValueError("Original native repeat failed bitwise determinism")
            report["original_first_batch_repeat_bitwise_equal"] = True
            run("original", originals)
            state = nnx.state(model)
            tree = nnx.to_pure_dict(state)
            weights = (
                a.quantized_directory / "model.safetensors"
                if a.role == "yat"
                else a.model_directory / "model.safetensors"
            )
            coverage = set()
            mapped_names = set()
            schema = {}
            with safe_open(str(weights), framework="np") as reader:

                def restore(path, current):
                    canonical = next(
                        iter(
                            _canonical_manifest_paths(
                                {jax.tree_util.keystr(path): None}
                            )
                        )
                    )
                    name, transpose = torch_name(canonical)
                    if a.role == "mmbert":
                        if name == "embedding.weight":
                            name = "model.embeddings.tok_embeddings.weight"
                        elif name == "embedding_norm.weight":
                            name = "model.embeddings.norm.weight"
                        elif name == "final_norm.weight":
                            name = "model.final_norm.weight"
                        elif name == "head_dense.weight":
                            name = "head.dense.weight"
                        elif name == "head_norm.weight":
                            name = "head.norm.weight"
                        elif name == "decoder_bias":
                            name = "decoder.bias"
                        else:
                            name = "model." + name
                            for old, new in (
                                (".qkv.", ".attn.Wqkv."),
                                (".attn_out.", ".attn.Wo."),
                                (".wi.", ".mlp.Wi."),
                                (".wo.", ".mlp.Wo."),
                            ):
                                name = name.replace(old, new)
                    mapped_names.add(name)
                    if a.role == "yat":
                        record = receipt["tensors"].get(name)
                        expected_shape = (
                            list(reversed(current.shape))
                            if transpose
                            else list(current.shape)
                        )
                        if (
                            not record
                            or record.get("shape") != expected_shape
                            or record.get("quantized") != (len(current.shape) == 2)
                        ):
                            raise ValueError(
                                f"Quantization tensor manifest mismatch: {name}"
                            )
                    if len(current.shape) == 2:
                        if a.role == "yat":
                            prefix = name[: -len("weight")]
                            qname, sname = prefix + "qweight", prefix + "scale"
                            value = reconstructed_matrix(
                                reader, qname, scale_name=sname, transpose=transpose
                            )
                            coverage.update((qname, sname))
                        else:
                            value = reconstructed_matrix(
                                reader, name, transpose=transpose
                            )
                            coverage.add(name)
                        schema[canonical] = {
                            "shape": list(current.shape),
                            "quantized": True,
                        }
                    else:
                        value = reader.get_tensor(name)
                        coverage.add(name)
                        if value.dtype != np.float32 or not np.array_equal(
                            value, np.asarray(current)
                        ):
                            raise ValueError(
                                f"Preserved norm/alpha/bias changed: {canonical}"
                            )
                        schema[canonical] = {
                            "shape": list(current.shape),
                            "quantized": False,
                        }
                    if (
                        tuple(value.shape) != tuple(current.shape)
                        or value.dtype != np.float32
                        or (value.ndim != 2 and not np.isfinite(value).all())
                    ):
                        raise ValueError(f"Restored schema mismatch: {canonical}")
                    return jax.device_put(value, device)

                restored = jax.tree_util.tree_map_with_path(restore, tree)
                if coverage != set(reader.keys()):
                    raise ValueError(
                        f"Quantized weight coverage differs: {set(reader.keys()) ^ coverage}"
                    )
            if a.role == "yat" and mapped_names != set(receipt["tensors"]):
                raise ValueError("Quantization manifest tensor inventory mismatch")
            nnx.replace_by_pure_dict(state, restored)
            nnx.update(model, state)
            del restored, tree, state
            report["quantized_tensor_schema"] = schema
            report["original_pooled_rows"] = len(rows)
            write_report(a.output, report)
            run("quantized", candidates)
            original_values, quantized_values = (
                np.concatenate(originals),
                np.concatenate(candidates),
            )
            report["quantized_outputs_changed"] = bool(
                np.any(original_values != quantized_values)
            )
            report["maximum_raw_pool_change"] = float(
                np.max(np.abs(original_values - quantized_values))
            )
            if not report["quantized_outputs_changed"]:
                raise ValueError(
                    "Quantized outputs unchanged; reject stale/unmodified state"
                )
        report["complete"] = True
        report["quantized_pooled_rows"] = len(rows)
        write_report(a.output, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report(a.output, report)
        raise


if __name__ == "__main__":
    main()
