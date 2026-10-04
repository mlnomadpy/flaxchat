"""Static, chunked analysis of authenticated FP32 Safetensors weight bytes.

No model imports or forward passes. Weight geometry is not activation quality.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    return dict(
        zip(
            ("min", "p01", "p25", "median", "p75", "p99", "max"),
            map(float, np.quantile(values, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])),
            strict=True,
        )
    )


def statistics(a):
    flat = a.reshape(-1)
    n = flat.size
    total = square = fourth = 0.0
    zero = bad = near = 0
    largest = 0.0
    for begin in range(0, n, 262144):
        v = flat[begin : begin + 262144].astype(np.float64)
        finite = np.isfinite(v)
        bad += int((~finite).sum())
        v = v[finite]
        if not v.size:
            continue
        total += float(v.sum())
        s = v * v
        square += float(s.sum())
        fourth += float((s * s).sum())
        zero += int((v == 0).sum())
        near += int((np.abs(v) < 1e-8).sum())
        largest = max(largest, float(np.abs(v).max()))
    k = n - bad
    mean = total / k if k else 0.0
    variance = max(0, square / k - mean * mean) if k else 0.0
    std = variance**0.5
    # Raw fourth moment ratio, not centered kurtosis.
    return {
        "elements": n,
        "nonfinite": bad,
        "zero_elements": zero,
        "abs_below_1e_minus8": near,
        "mean": mean,
        "std": std,
        "rms": (square / k) ** 0.5 if k else 0.0,
        "frobenius_norm": square**0.5,
        "max_abs": largest,
        "max_abs_over_std": largest / std if std else None,
        "raw_fourth_over_second_squared": fourth * k / (square * square)
        if square
        else None,
    }


def analyze(path, expected_sha):
    path = Path(path)
    actual = digest(path)
    if actual != expected_sha:
        raise ValueError(
            "Full weight SHA256 differs from independently pinned artifact"
        )
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= size <= 1024 * 1024:
            raise ValueError("Unexpected Safetensors header length")
        header = json.loads(f.read(size))
    offset = 8 + size
    weights = {}
    intervals = []
    for key, item in header.items():
        if key == "__metadata__":
            continue
        if item["dtype"] != "F32":
            raise ValueError("This analysis requires the released FP32 stored weights")
        start, end = item["data_offsets"]
        shape = tuple(item["shape"])
        count = int(np.prod(shape))
        if start < 0 or end - start != count * 4 or offset + end > path.stat().st_size:
            raise ValueError("Invalid tensor byte bounds")
        intervals.append((start, end))
        weights[key] = np.memmap(
            path, dtype="<f4", mode="r", offset=offset + start, shape=shape
        )
    ordered = sorted(intervals)
    if (
        ordered[0][0] != 0
        or any(a[1] != b[0] for a, b in zip(ordered, ordered[1:], strict=False))
        or offset + ordered[-1][1] != path.stat().st_size
    ):
        raise ValueError("Overlapping, missing or trailing tensor bytes")
    tensors = {}
    layers = {}
    embeddings = None
    for key, a in weights.items():
        rec = {"shape": list(a.shape), "stored_dtype": "float32", **statistics(a)}
        tensors[key] = rec
        match = re.search(r"\['layers'\]\['(\d+)'\]", key)
        layer = (
            layers.setdefault(int(match.group(1)), {"layer": int(match.group(1))})
            if match
            else None
        )
        if "alpha" in key:
            value = float(a)
            rec["scalar_value"] = value
            if layer is not None:
                layer[
                    "attention_alpha" if "attention_alpha" in key else "ffn_alpha"
                ] = value
        if key.endswith("['scale']"):
            rec["gain_quantiles"] = quantiles(a)
            rec["negative_gain_count"] = int((a < 0).sum())
        if key.endswith("['wi']['kernel']"):
            mid = a.shape[1] // 2
            proto = a[:, :mid]
            gate = a[:, mid:]
            pn = np.linalg.norm(proto.astype(np.float64), axis=0)
            gn = np.linalg.norm(gate.astype(np.float64), axis=0)
            rng = np.random.default_rng(17)
            i = rng.integers(0, mid, 4096)
            j = (i + rng.integers(1, mid, 4096)) % mid
            cosine = np.sum(
                proto[:, i].astype(np.float64) * proto[:, j], axis=0
            ) / np.maximum(pn[i] * pn[j], 1e-30)
            layer.update(
                prototype_norm_quantiles=quantiles(pn),
                gate_norm_quantiles=quantiles(gn),
                prototype_gate_mean_norm_ratio=float(pn.mean() / gn.mean()),
                sampled_distinct_prototype_cosine_quantiles=quantiles(cosine),
                zero_prototype_columns=int((pn == 0).sum()),
            )
        if key.endswith("['qkv']['kernel']"):
            width = a.shape[0]
            heads = 12
            if width != 768 or a.shape[1] != 3 * width:
                raise ValueError("Unexpected projection geometry")
            qkv = a.reshape(width, 3, heads, width // heads).astype(np.float64)
            layer["qkv_per_head_frobenius_norms"] = {
                name: np.sqrt((qkv[:, index] ** 2).sum(axis=(0, 2))).tolist()
                for index, name in enumerate(["q", "k", "v"])
            }
        if key == "['embedding']['embedding']":
            norms = np.empty(a.shape[0], dtype=np.float64)
            for start in range(0, a.shape[0], 1024):
                norms[start : start + 1024] = np.linalg.norm(
                    a[start : start + 1024].astype(np.float64), axis=1
                )
            indices = np.argsort(norms)[-10:][::-1]
            rng = np.random.default_rng(42)
            chosen = np.sort(rng.choice(a.shape[0], 512, replace=False))
            sample = a[chosen].astype(np.float64)
            sample /= np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-30)
            cov = sample.T @ sample / len(sample)
            eig = np.maximum(np.linalg.eigvalsh(cov), 0)
            total = float(eig.sum())
            mass = eig[eig > 1e-14] / total
            embeddings = {
                "rows": a.shape[0],
                "width": a.shape[1],
                "row_norm_quantiles": quantiles(norms),
                "zero_rows": int((norms == 0).sum()),
                "top_norm_token_ids": [
                    {"id": int(i), "norm": float(norms[i])} for i in indices
                ],
                "sampled_token_ids": chosen.tolist(),
                "sample_size": 512,
                "sampled_uncentered_effective_rank": float(
                    np.exp(-(mass * np.log(mass)).sum())
                ),
                "sampled_uncentered_top_direction_energy_fraction": float(
                    eig[-1] / total
                ),
                "sampled_mean_distinct_cosine": float(
                    ((sample.sum(axis=0) ** 2).sum() - len(sample))
                    / (len(sample) * (len(sample) - 1))
                ),
                "caution": "Static token embeddings sampled across vocabulary, not pooled sentence embeddings or language-balanced representation quality",
            }
    count = sum(r["elements"] for r in tensors.values())
    nonfinite = sum(r["nonfinite"] for r in tensors.values())
    alphas = [
        v
        for r in layers.values()
        for k, v in r.items()
        if k in ["ffn_alpha", "attention_alpha"]
    ]
    return {
        "format": "static-yat-weight-analysis-v1",
        "weight_sha256": actual,
        "weight_bytes": path.stat().st_size,
        "tensor_count": len(weights),
        "parameter_count": count,
        "stored_dtype": "float32",
        "nonfinite_elements": nonfinite,
        "zero_elements": sum(r["zero_elements"] for r in tensors.values()),
        "negative_or_zero_alphas": sum(x <= 0 for x in alphas),
        "source_sha256": digest(__file__),
        "numpy_version": np.__version__,
        "scope": "Static weight-byte analysis only; no model forward/backward or performance measurements",
        "layers": [layers[i] for i in sorted(layers)],
        "token_embeddings": embeddings,
        "tensors": tensors,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("Refusing to overwrite an analysis receipt")
    report = analyze(a.weights, a.expected_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "parameter_count",
                    "tensor_count",
                    "nonfinite_elements",
                    "negative_or_zero_alphas",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
