"""Verify and summarize a paired, handcrafted 88-text TPU diagnostic.

Only retained scalar receipts and pooled vectors are read; no model is imported.
These probes compare contrastively fine-tuned YAT with MLM-pretrained mmBERT.
They do not isolate architecture or reproduce an official benchmark.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from scripts.analyze_encoder_activations_tpu import geometry
from scripts.benchmark_embedding_speed_tpu import digest, write_report

PINS = {
    "yat": (
        "mlnomad/yat-mmbert-base-embedding-v1",
        "571d0ef3d8e1381055845fce334a48be253c3fea",
        "model.safetensors",
        "6ee56349adbf10a3c3d5805d94539f987250aea2c98611b4d89c6fd9b7142aff",
    ),
    "mmbert": (
        "jhu-clsp/mmBERT-base",
        "c5955035435e2bf121cde7f3c8863ef52ff35d82",
        "pytorch_model.bin",
        "8ea64ec1ea4eb8fca0fc14b69a2ae571de6bfbc25fd214bb932dd4aba6a3a04e",
    ),
}
INDICES = [0, 8, 16, 24, 32, 40, 48, 56, 64, 65, 66, 67, 72, 74, 80, 82]
HEAD_FIELDS = (
    "attention_entropy_nats_per_head",
    "attention_top1_mass_per_head",
    "mean_q_norm_per_head",
    "mean_k_norm_per_head",
)
RMS_FIELDS = ("input_rms", "attention_delta_rms", "post_attention_rms", "ffn_delta_rms")
FRACTIONS = (
    "ffn_feature_zero_fraction_valid",
    "ffn_feature_abs_lt_1e_minus4_fraction_valid",
)
COUNTS = (
    "adaptive_sensitive_pairs_including_padding",
    "adaptive_sensitive_tiles_including_padding",
    "adaptive_total_pairs_including_padding",
    "adaptive_sensitive_valid_pairs",
    "adaptive_total_tiles_including_padding",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_json(path, maximum=8 * 1024**2):
    path = Path(path)
    require(
        path.is_file() and path.stat().st_size <= maximum,
        "Missing or oversized JSON: " + str(path),
    )
    return json.loads(path.read_text())


def valid_hash(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def fixture(root, pin):
    require(
        valid_hash(pin) and digest(root / "inputs.json") == pin,
        "Independent fixture SHA256 differs",
    )
    rows = bounded_json(root / "inputs.json", 1024**2)
    require(
        isinstance(rows, list) and len(rows) == 88,
        "Probe layout requires exactly 88 declared rows",
    )
    require(
        all(
            isinstance(r, dict)
            and set(r) == {"id", "domain", "language", "group", "variant", "text"}
            and all(isinstance(v, str) and v for v in r.values())
            for r in rows
        ),
        "Fixture row schema differs",
    )
    require(len({r["id"] for r in rows}) == 88, "Fixture IDs must be unique")
    for block, lang in enumerate(("en", "es", "fr", "de", "ar", "zh", "ja", "hi")):
        for concept in range(8):
            r = rows[8 * block + concept]
            require(
                (r["id"], r["domain"], r["language"], r["group"], r["variant"])
                == (
                    f"{lang}-{concept}",
                    "english" if block == 0 else "multilingual",
                    lang,
                    f"text-{concept}",
                    lang,
                ),
                "Translation probe layout differs",
            )
    for start, variant, domain, lang in (
        (64, "query", "code_query", "en"),
        (72, "python", "code", "python"),
        (80, "javascript", "code", "javascript"),
    ):
        for concept in range(8):
            r = rows[start + concept]
            require(
                (r["id"], r["domain"], r["language"], r["group"], r["variant"])
                == (f"{variant}-{concept}", domain, lang, f"code-{concept}", variant),
                "Code probe layout differs",
            )
    return rows


def validate(root, role, rows, pin, *, native_only=False):
    report = bounded_json(root / (role + ".json"))
    repo, revision, weight_name, weight_pin = PINS[role]
    require(
        report.get("format") == "flaxchat-light-activation-diagnostic-v1"
        and (
            native_only
            or (report.get("complete") is True and not report.get("failure"))
        ),
        "Incomplete/failed " + role + " receipt; layers must not be interpreted",
    )
    require(
        (report.get("role"), report.get("repo"), report.get("revision"))
        == (role, repo, revision),
        "Model revision/role differs",
    )
    artifacts = report.get("artifacts_sha256", {})
    require(
        artifacts.get(weight_name) == weight_pin
        and all(
            valid_hash(artifacts.get(n))
            for n in ("model.safetensors", "tokenizer.json", "config.json")
        ),
        "Published model artifact pins differ",
    )
    if role == "mmbert":
        conversion = report.get("conversion", {})
        require(
            conversion.get("source_sha256") == weight_pin
            and conversion.get("safetensors_sha256") == artifacts["model.safetensors"],
            "Baseline conversion is not bound to original published weights",
        )
    require(
        report.get("fixture_sha256") == pin
        and report.get("rows")
        == [{k: v for k, v in r.items() if k != "text"} for r in rows],
        "Receipt fixture identity differs",
    )
    require(
        report.get("native_pooled_rows") == 88
        and report.get("diagnostic_indices") == INDICES
        and report.get("activation_rows") == 16,
        "Pooling/activation scopes differ",
    )
    require(
        report.get("reference_absolute_drift_bound") == 0.01
        and report.get("reference_relative_l2_drift_bound") == 0.005,
        "Declared trace bounds changed",
    )
    require(
        report.get("hardware", {}).get("utilized_devices") == 1
        and "TPU" in report.get("hardware", {}).get("kind", ""),
        "Physical one-device TPU receipt required",
    )
    config = report.get("encoder_config", {})
    require(
        (
            config.get("hidden_size"),
            config.get("num_hidden_layers"),
            config.get("num_attention_heads"),
            config.get("compute_dtype"),
            config.get("residual_dtype"),
            config.get("yat_local_shards"),
        )
        == (768, 22, 12, "bfloat16", "float32", False),
        "Native geometry or precision differs",
    )
    if role == "yat":
        require(
            (
                config.get("yat_bias"),
                config.get("yat_epsilon"),
                config.get("yat_alpha_trainable"),
                config.get("ffn_type"),
                config.get("attention_score"),
                config.get("yat_ffn_compute_mode"),
                config.get("yat_attention_implementation"),
            )
            == (
                1.0,
                0.01,
                True,
                "yat_glu",
                "yat_softmax",
                "bf16_adaptive",
                "centered_fp32_scores",
            ),
            "YAT native math differs",
        )
    else:
        require(
            config.get("ffn_type") == "geglu"
            and config.get("attention_score") == "dot_product",
            "Baseline native math differs",
        )
    counts = report.get("token_counts")
    require(
        isinstance(counts, list)
        and len(counts) == 88
        and all(type(v) is int and v > 0 for v in counts),
        "Token counts missing",
    )
    require(
        report.get("truncated_rows") == sum(v > 128 for v in counts)
        and valid_hash(report.get("input_int32_sha256")),
        "Token/input identity differs",
    )
    batches = report.get("batches")
    if not native_only:
        require(
            isinstance(batches, list) and len(batches) == 2,
            "Complete two-batch trace required",
        )
    for index, batch in enumerate([] if native_only else batches):
        selected = INDICES[index * 8 : (index + 1) * 8]
        require(
            batch.get("row_indices") == selected
            and batch.get("rows") == 8
            and batch.get("valid_tokens") == sum(min(counts[v], 128) for v in selected),
            "Trace row/token scope differs",
        )
        require(
            batch.get("trace_native_parity_accepted") is True
            and np.isfinite(batch.get("reference_max_absolute_drift", np.nan))
            and 0 <= batch["reference_max_absolute_drift"] <= 0.01
            and np.isfinite(batch.get("reference_relative_l2_drift", np.nan))
            and 0 <= batch["reference_relative_l2_drift"] <= 0.005,
            "Trace/native parity failed",
        )
        layers = batch.get("layers")
        require(
            isinstance(layers, list) and len(layers) == 22, "Missing layer diagnostics"
        )
        for i, layer in enumerate(layers):
            require(layer.get("layer") == i, "Layer order differs")
            for key in RMS_FIELDS + HEAD_FIELDS + FRACTIONS:
                value = np.asarray(layer.get(key), dtype=float)
                require(
                    value.shape == ((12,) if key in HEAD_FIELDS else ())
                    and np.isfinite(value).all()
                    and np.all(value >= 0),
                    "Invalid scalar/head metric " + key,
                )
                if key in FRACTIONS or key == "attention_top1_mass_per_head":
                    require(np.all(value <= 1 + 1e-6), "Fraction outside range")
            require(
                layer["input_rms"] > 0 and layer["post_attention_rms"] > 0,
                "Collapsed residual stream",
            )
            require(
                all(type(layer.get(k)) is int and layer[k] >= 0 for k in COUNTS),
                "Invalid collision counts",
            )
            if role == "yat":
                width = config["intermediate_size"]
                require(
                    layer["adaptive_total_pairs_including_padding"] == 8 * 128 * width
                    and layer["adaptive_total_tiles_including_padding"]
                    == 64 * ((width + 63) // 64)
                    and layer["adaptive_sensitive_valid_pairs"]
                    <= batch["valid_tokens"] * width
                    and layer["adaptive_sensitive_valid_pairs"]
                    <= layer["adaptive_sensitive_pairs_including_padding"]
                    <= 8 * 128 * width
                    and layer["adaptive_sensitive_tiles_including_padding"]
                    <= layer["adaptive_total_tiles_including_padding"],
                    "Collision scope/counts differ",
                )
    path = root / (role + ".npz")
    require(
        path.is_file()
        and path.stat().st_size <= 8 * 1024**2
        and digest(path) == report.get("pooled_npz_sha256"),
        "Pooled byte SHA differs",
    )
    with zipfile.ZipFile(path) as archive:
        require(
            archive.namelist() == ["pooled.npy"]
            and archive.getinfo("pooled.npy").file_size <= 8 * 1024**2,
            "NPZ inventory/expanded bound differs",
        )
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays["pooled"].copy()
    require(
        values.shape == (88, 768)
        and values.dtype == np.float32
        and np.isfinite(values).all()
        and np.all(np.linalg.norm(values, axis=1) > 1e-12),
        "Invalid retained native pooled vectors",
    )
    if not native_only:
        trace_path = root / (role + ".trace.npz")
        require(
            trace_path.is_file()
            and trace_path.stat().st_size <= 8 * 1024**2
            and digest(trace_path) == report.get("trace_npz_sha256"),
            "Raw trace comparison evidence SHA differs",
        )
        with zipfile.ZipFile(trace_path) as archive:
            require(
                set(archive.namelist())
                == {
                    "traced.npy",
                    "same_batch_native.npy",
                    "initial_native.npy",
                    "row_indices.npy",
                }
                and sum(v.file_size for v in archive.infolist()) <= 8 * 1024**2,
                "Trace evidence expanded inventory differs",
            )
        with np.load(trace_path, allow_pickle=False) as arrays:
            traced, same, initial, indices = [
                arrays[k].copy()
                for k in (
                    "traced",
                    "same_batch_native",
                    "initial_native",
                    "row_indices",
                )
            ]
        require(
            indices.shape == (16,)
            and np.array_equal(indices, INDICES)
            and all(
                v.shape == (16, 768) and v.dtype == np.float32 and np.isfinite(v).all()
                for v in (traced, same, initial)
            )
            and np.array_equal(initial, values[INDICES]),
            "Raw trace evidence shape/initial vectors differ",
        )
        for i, batch in enumerate(batches):
            candidate, reference = (
                traced[i * 8 : (i + 1) * 8],
                same[i * 8 : (i + 1) * 8],
            )
            absolute = float(np.max(np.abs(candidate - reference)))
            relative = float(
                np.linalg.norm(candidate - reference)
                / max(np.linalg.norm(reference), 1e-12)
            )
            require(
                batch.get("reference_scope")
                == "fresh native model.pool on identical diagnostic row grouping"
                and np.isclose(
                    absolute,
                    batch["reference_max_absolute_drift"],
                    rtol=1e-6,
                    atol=1e-8,
                )
                and np.isclose(
                    relative, batch["reference_relative_l2_drift"], rtol=1e-6, atol=1e-8
                )
                and absolute <= 0.01
                and relative <= 0.005,
                "Raw same-batch trace parity replay failed",
            )
    return report, values


def summarize(report, vectors, rows, role, *, native_only=False):
    unit = vectors.astype(np.float64) / np.linalg.norm(
        vectors.astype(np.float64), axis=1, keepdims=True
    )
    cosine = unit @ unit.T

    def probe(queries, documents):
        ranked = documents[
            np.argsort(-cosine[np.ix_(queries, documents)], axis=1, kind="stable")
        ]
        positives = np.array(
            [
                [rows[q]["group"] == rows[d]["group"] for d in ranked[i]]
                for i, q in enumerate(queries)
            ]
        )
        require(positives.any(axis=1).all(), "Missing positive probe candidate")
        first = positives.argmax(axis=1) + 1
        return {
            "queries": len(queries),
            "candidates": len(documents),
            "recall_at_1": float(positives[:, 0].mean()),
            "mrr": float((1 / first).mean()),
            "tie_policy": "stable declared candidate order",
            "top1_ids": [rows[d]["id"] for d in ranked[:, 0]],
            "first_positive_ranks": first.tolist(),
        }

    text, english, codeq, code = (
        np.arange(8, 64),
        np.arange(8),
        np.arange(64, 72),
        np.arange(72, 88),
    )
    result = {
        "geometry": geometry(vectors),
        "translation_probe": probe(text, english),
        "code_probe": probe(codeq, code),
        "translation_by_language": {
            lang: probe(
                np.array([i for i in text if rows[i]["language"] == lang]), english
            )
            for lang in sorted({rows[i]["language"] for i in text})
        },
        "geometry_by_domain": {
            domain: geometry(vectors[[r["domain"] == domain for r in rows]])
            for domain in sorted({r["domain"] for r in rows})
        },
        "geometry_by_language": {
            lang: geometry(vectors[[r["language"] == lang for r in rows]])
            for lang in sorted({r["language"] for r in rows})
        },
        "layers": [],
    }
    tri = np.triu_indices(88, 1)
    paired = np.array(
        [rows[i]["group"] == rows[j]["group"] for i, j in zip(*tri, strict=True)]
    )
    result["paired_cosine_mean"] = float(cosine[tri][paired].mean())
    result["cross_group_cosine_mean"] = float(cosine[tri][~paired].mean())
    result["cosine_margin"] = (
        result["paired_cosine_mean"] - result["cross_group_cosine_mean"]
    )
    result["native_vectors_complete"] = True
    result["original_worker_complete"] = report["complete"]
    result["original_worker_failure"] = report.get("failure")
    result["trace_metrics_interpreted"] = not native_only
    result["trace_outcomes_preserved"] = [
        {k: v for k, v in b.items() if k != "layers"} for b in report.get("batches", [])
    ]
    if native_only:
        return result
    batches = report["batches"]
    weights = np.array([b["valid_tokens"] for b in batches])
    total = weights.sum()
    for i in range(22):
        records = [b["layers"][i] for b in batches]
        layer = {"layer": i, "valid_tokens": int(total)}
        for key in RMS_FIELDS:
            layer[key] = float(
                np.sqrt(np.average([r[key] ** 2 for r in records], weights=weights))
            )
        layer["attention_update_ratio"] = (
            layer["attention_delta_rms"] / layer["input_rms"]
        )
        layer["ffn_update_ratio"] = layer["ffn_delta_rms"] / layer["post_attention_rms"]
        for key in HEAD_FIELDS + FRACTIONS:
            value = np.average([r[key] for r in records], axis=0, weights=weights)
            layer[key] = value.tolist()
        if role == "yat":
            layer["sensitive_valid_pair_fraction"] = sum(
                r["adaptive_sensitive_valid_pairs"] for r in records
            ) / (total * report["encoder_config"]["intermediate_size"])
            layer["sensitive_padded_pair_fraction"] = sum(
                r["adaptive_sensitive_pairs_including_padding"] for r in records
            ) / sum(r["adaptive_total_pairs_including_padding"] for r in records)
            layer["sensitive_padded_tile_fraction"] = sum(
                r["adaptive_sensitive_tiles_including_padding"] for r in records
            ) / sum(r["adaptive_total_tiles_including_padding"] for r in records)
        result["layers"].append(layer)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--fixture-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--native-only",
        action="store_true",
        help="Validate native88vectors; preserve tracefailure and never interpret traces",
    )
    p.add_argument("--roles", nargs="+", choices=tuple(PINS), default=list(PINS))
    p.add_argument(
        "--source-map",
        type=Path,
        help="Optional independently frozen JSON source-SHA map",
    )
    p.add_argument(
        "--source-map-sha256",
        help="Mandatory independent byte SHA when source-map is provided",
    )
    a = p.parse_args()
    require(not a.output.exists(), "Refusing to overwrite prior report")
    rows = fixture(a.directory, a.fixture_sha256)
    require(
        len(set(a.roles)) == len(a.roles)
        and (a.native_only or set(a.roles) == set(PINS)),
        "Single-role reporting requires native-only",
    )
    if a.source_map is not None:
        require(
            valid_hash(a.source_map_sha256)
            and digest(a.source_map) == a.source_map_sha256,
            "Independent source-map byte SHA differs",
        )
        frozen_sources = bounded_json(a.source_map)
    else:
        require(a.source_map_sha256 is None, "source-map required with its SHA")
        frozen_sources = None
    loaded = {
        role: validate(
            a.directory, role, rows, a.fixture_sha256, native_only=a.native_only
        )
        for role in a.roles
    }
    for receipt, _ in loaded.values():
        require(
            isinstance(receipt.get("source_sha256"), dict)
            and bool(receipt["source_sha256"])
            and all(valid_hash(v) for v in receipt["source_sha256"].values()),
            "Missing worker source identity",
        )
        if frozen_sources is not None:
            require(
                receipt["source_sha256"] == frozen_sources,
                "Worker source differs from independent frozen map",
            )
    if len(loaded) == 2:
        left, right = loaded["yat"][0], loaded["mmbert"][0]
        require(
            left["artifacts_sha256"]["tokenizer.json"]
            == right["artifacts_sha256"]["tokenizer.json"]
            and left["input_int32_sha256"] == right["input_int32_sha256"]
            and left["token_counts"] == right["token_counts"],
            "Paired tokenizer/tokenized inputs differ",
        )
        require(
            left["source_sha256"] == right["source_sha256"]
            and all(valid_hash(v) for v in left["source_sha256"].values()),
            "Paired worker sources differ",
        )
        require(
            left["runtime"] == right["runtime"]
            and left["hardware"] == right["hardware"],
            "Paired execution runtime/hardware differ",
        )
    report = {
        "format": "flaxchat-verified-activation-summary-v1",
        "fixture_sha256": a.fixture_sha256,
        "scope": "88 handcrafted texts; 16 traced texts; diagnostic only, not official benchmark",
        "comparison_limit": "YAT contrastive fine-tune vs mmBERT MLM base; architecture/training recipe not controlled",
        "interpretation": [
            "Head statistics are valid-token-weighted; entropy depends on eligible keys.",
            "RMS ratios are update amplitudes, not causal percentages.",
            "Nearzero fraction uses absolute1e-4 threshold; not dead-neuron proof.",
            "Sensitive tiles are candidate tiles; actual dense/sparse repair dispatch not instrumented.",
        ],
        "receipts_sha256": {
            role: digest(a.directory / (role + ".json")) for role in loaded
        },
        "native_only": a.native_only,
        "paired_comparison_complete": len(loaded) == 2,
        "independent_source_map_sha256": a.source_map_sha256,
        "reporter_source_sha256": digest(Path(__file__)),
        "models": {
            role: summarize(receipt, vectors, rows, role, native_only=a.native_only)
            for role, (receipt, vectors) in loaded.items()
        },
    }
    write_report(a.output, report)
    print(
        json.dumps(
            {"output": str(a.output), "sha256": digest(a.output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
