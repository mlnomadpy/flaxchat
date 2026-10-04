"""Bounded real-text activation diagnostics on one physical TPU device.

This is a diagnostic fixture, not a benchmark-quality or training-data claim.
No model forward runs on CPU. NumPy summarizes already-produced pooled vectors.
Run models sequentially in separate processes with an external finite lease.
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


def geometry(vectors):
    import numpy as np

    values = np.asarray(vectors, dtype=np.float64)
    normalized = values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12
    )

    def ranks(matrix):
        singular = np.linalg.svd(matrix, compute_uv=False)
        energy = singular**2
        if energy.sum() <= 1e-30:
            return {
                "energy_entropy_effective_rank": 0.0,
                "participation_rank": 0.0,
                "leading_energy_fraction": 0.0,
            }
        energy = energy / energy.sum()
        positive = energy[energy > 0]
        return {
            "energy_entropy_effective_rank": float(
                np.exp(-(positive * np.log(positive)).sum())
            ),
            "participation_rank": float(1 / max((energy**2).sum(), 1e-30)),
            "leading_energy_fraction": float(energy.max()),
        }

    cos = normalized @ normalized.T
    off = cos[~np.eye(len(cos), dtype=bool)]
    return {
        "rows": len(values),
        "raw_norm_mean": float(np.linalg.norm(values, axis=1).mean()),
        "mean_offdiagonal_cosine": float(off.mean()) if off.size else None,
        "offdiagonal_cosine_p10_p50_p90": np.quantile(off, [0.1, 0.5, 0.9]).tolist()
        if off.size
        else [],
        "uncentered_unit_vectors": ranks(normalized),
        "centered_unit_vectors": ranks(
            normalized - normalized.mean(axis=0, keepdims=True)
        ),
        "rank_limit": min(values.shape),
        "centered_rank_limit": min(len(values) - 1, values.shape[1]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("yat", "mmbert"), required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--expected-weights-sha256", required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--fixture-sha256", required=True)
    p.add_argument("--model-directory", type=Path)
    p.add_argument("--download-root", type=Path, default=Path("/tmp/activation-models"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-reference-absolute-drift", type=float, default=0.01)
    p.add_argument("--max-reference-relative-l2-drift", type=float, default=0.005)
    p.add_argument(
        "--diagnostic-indices",
        required=True,
        help="Comma-separated unique row indices, multiple of8; all rows still receive native pooling",
    )
    a = p.parse_args()
    if (
        a.repo
        != {
            "yat": "mlnomad/yat-mmbert-base-embedding-v1",
            "mmbert": "jhu-clsp/mmBERT-base",
        }[a.role]
        or not re.fullmatch("[0-9a-f]{40}", a.revision)
        or any(
            not re.fullmatch("[0-9a-f]{64}", h)
            for h in (a.expected_weights_sha256, a.fixture_sha256)
        )
        or not 0 <= a.max_reference_absolute_drift <= 0.01
        or not 0 <= a.max_reference_relative_l2_drift <= 0.005
    ):
        p.error(
            "Require supported immutable model, artifact/fixture SHA and strict finite drift bound"
        )
    npz = a.output.with_suffix(".npz")
    trace_npz = a.output.with_suffix(".trace.npz")
    if a.output.exists() or npz.exists() or trace_npz.exists():
        p.error("Prior results must not be overwritten")
    if a.fixture.stat().st_size > 1024 * 1024 or digest(a.fixture) != a.fixture_sha256:
        p.error("Fixture bytes differ or exceed bound")
    rows = json.loads(a.fixture.read_text())
    if (
        not isinstance(rows, list)
        or not 2 <= len(rows) <= 128
        or len(rows) % 8
        or any(
            not isinstance(r, dict)
            or not {"id", "domain", "text", "group", "variant"} <= set(r)
            or bool(set(r) - {"id", "domain", "text", "group", "variant", "language"})
            or any(not isinstance(v, str) or not v for v in r.values())
            for r in rows
        )
        or len({r["id"] for r in rows}) != len(rows)
    ):
        p.error("Require 8-aligned 2..128 unique typed diagnostic rows")

    indices = [int(v) for v in a.diagnostic_indices.split(",")]
    if (
        not indices
        or len(indices) % 8
        or len(indices) > 32
        or len(set(indices)) != len(indices)
        or any(not 0 <= v < len(rows) for v in indices)
    ):
        p.error("Require unique bounded 8-aligned diagnostic row indices")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from tokenizers import Tokenizer
    from flaxchat.encoder import (
        EncoderConfig,
        ModernBert,
        _rope,
        bidirectional_attention,
        yat_glu,
    )
    from flaxchat.public_encoder import load_public_encoder
    from flaxchat.yat import squared_distance_bf16
    from scripts.train_encoder import load_pretrained

    if jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise RuntimeError(
            "Physical single-host TPU required; CPU model fallback forbidden"
        )
    device = jax.local_devices()[0]
    if device.platform != "tpu":
        raise RuntimeError("Physical TPU required")
    sources = sorted(ROOT.glob("flaxchat/**/*.py")) + [
        Path(__file__),
        ROOT / "scripts/benchmark_embedding_speed_tpu.py",
        ROOT / "scripts/train_encoder.py",
        ROOT / "scripts/convert_encoder_checkpoint.py",
    ]
    packages = {}
    for name in (
        "jax",
        "jaxlib",
        "flax",
        "numpy",
        "tokenizers",
        "safetensors",
        "torch",
        "huggingface-hub",
        "libtpu",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "format": "flaxchat-light-activation-diagnostic-v1",
        "complete": False,
        "role": a.role,
        "repo": a.repo,
        "revision": a.revision,
        "fixture_sha256": a.fixture_sha256,
        "rows": [{k: v for k, v in r.items() if k != "text"} for r in rows],
        "scope": "generated real-text diagnostic; not evaluation benchmark; one TPU device",
        "runtime": {"python": platform.python_version(), "packages": packages},
        "hardware": {
            "kind": device.device_kind,
            "measured_device": str(device),
            "utilized_devices": 1,
        },
        "source_sha256": {str(v.relative_to(ROOT)): digest(v) for v in sources},
        "adaptive_diagnostic": {
            "relative_threshold": 0.125,
            "row_tile": 16,
            "column_tile": 64,
            "scope": "sensitive candidate pairs/tiles; actual dense-vs-sparse repair dispatch not instrumented",
        },
        "batches": [],
        "diagnostic_indices": indices,
        "activation_rows": len(indices),
        "trace_execution": "single full-forward nnx.jit, native remat policy, same-batch reference",
        "reference_absolute_drift_bound": a.max_reference_absolute_drift,
        "reference_relative_l2_drift_bound": a.max_reference_relative_l2_drift,
    }
    write_report(a.output, report)
    try:
        directory = a.model_directory
        if directory is None:
            from huggingface_hub import snapshot_download

            directory = Path(
                snapshot_download(
                    a.repo,
                    revision=a.revision,
                    token=False,
                    local_dir=a.download_root / a.role,
                    allow_patterns=[
                        "config.json",
                        "tokenizer.json",
                        "model.safetensors",
                        "pytorch_model.bin",
                    ],
                )
            )
        directory = Path(directory)
        weight = "model.safetensors" if a.role == "yat" else "pytorch_model.bin"
        if digest(directory / weight) != a.expected_weights_sha256:
            raise ValueError("Published trained weights differ from independent pin")
        if a.role == "mmbert":
            from scripts.convert_encoder_checkpoint import convert

            conversion = (
                convert(directory)
                if not (directory / "model.safetensors").exists()
                else json.loads((directory / "conversion.json").read_text())
            )
            if conversion["source_sha256"] != a.expected_weights_sha256 or conversion[
                "safetensors_sha256"
            ] != digest(directory / "model.safetensors"):
                raise ValueError(
                    "Baseline conversion does not authenticate pinned original weights"
                )
            report["conversion"] = conversion
        report["artifacts_sha256"] = {
            n: digest(directory / n)
            for n in {"config.json", "tokenizer.json", "model.safetensors", weight}
        }
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        encoded = tokenizer.encode_batch([r["text"] for r in rows])
        with jax.default_device(device):
            if a.role == "yat":
                model = load_public_encoder(directory)
            else:
                c = EncoderConfig.from_hf(
                    json.loads((directory / "config.json").read_text()),
                    compute_dtype="bfloat16",
                    residual_dtype="float32",
                    attention_backend="xla",
                )
                model = ModernBert(c, rngs=nnx.Rngs(0))
                load_pretrained(model, directory)
            c = model.config
            if c.compute_dtype != "bfloat16" or c.residual_dtype != "float32":
                raise ValueError(
                    "Matched BF16 features / FP32 residual policy required"
                )
            if a.role == "yat" and (
                c.yat_ffn_compute_mode != "bf16_adaptive"
                or c.yat_attention_implementation != "centered_fp32_scores"
            ):
                raise ValueError(
                    "Diagnostic supports published adaptiveFFN/FP32-score YAT only"
                )
            if c.yat_local_shards:
                raise ValueError(
                    "One-device diagnostic requires native unsharded YAT dispatch"
                )
            report["encoder_config"] = asdict(c)
            ids_host = np.full((len(rows), 128), c.pad_token_id, np.int32)
            for i, item in enumerate(encoded):
                ids_host[i, : min(128, len(item.ids))] = item.ids[:128]
            report["token_counts"] = [len(v.ids) for v in encoded]
            report["truncated_rows"] = sum(len(v.ids) > 128 for v in encoded)
            report["input_int32_sha256"] = hashlib.sha256(
                ids_host.tobytes()
            ).hexdigest()

            def attention_stats(q, k, segments, index, alpha):
                # Replay each query tile's actual union/padding geometry for YAT.
                length, heads = q.shape[1], q.shape[2]
                local = index % c.global_attn_every_n_layers != 0
                radius = c.local_attention // 2 if local else None
                tile = (
                    min(
                        c.yat_attention_block_size
                        if local
                        else c.yat_global_attention_block_size,
                        length,
                    )
                    if a.role == "yat"
                    else length
                )
                total_entropy = jnp.zeros(heads, jnp.float32)
                total_mass = jnp.zeros(heads, jnp.float32)
                valid = segments >= 0
                q = jnp.where(valid[..., None, None], q, 0)
                k = jnp.where(valid[..., None, None], k, 0)
                for start in range(0, length, tile):
                    end = min(length, start + tile)
                    lo, hi = (
                        (max(0, start - radius), min(length, end + radius))
                        if local
                        else (0, length)
                    )
                    qq = q[:, start:end].transpose(0, 2, 1, 3)
                    kk = k[:, lo:hi].transpose(0, 2, 1, 3)
                    # YAT actual kernel contains halo padding. Preserve reduction length.
                    left = max(0, radius - start) if local else 0
                    right = max(0, end + radius - length) if local else 0
                    if local:
                        kk = jnp.pad(kk, ((0, 0), (0, 0), (left, right), (0, 0)))
                    if a.role == "yat":
                        dots = jnp.matmul(
                            qq, kk.swapaxes(-1, -2), preferred_element_type=jnp.bfloat16
                        )
                        dist = squared_distance_bf16(qq, kk, block=32)
                        logits = (
                            alpha.astype(jnp.float32)
                            * (dots.astype(jnp.float32) + 1) ** 2
                            / (dist.astype(jnp.float32) + 0.01)
                        )
                    else:
                        logits = jnp.matmul(
                            qq.astype(jnp.float32),
                            kk.astype(jnp.float32).swapaxes(-1, -2),
                        ) / jnp.sqrt(q.shape[-1])
                    qpos, kpos = (
                        jnp.arange(start, end),
                        jnp.arange(lo - left, hi + right),
                    )
                    kv = jnp.pad(valid[:, lo:hi], ((0, 0), (left, right)))
                    mask = valid[:, start:end, None] & kv[:, None, :]
                    if local:
                        mask &= jnp.abs(qpos[:, None] - kpos[None, :])[None] <= radius
                    mask |= (~valid[:, start:end, None]) & (
                        jnp.arange(kk.shape[-2]) == 0
                    )[None, None, :]
                    probs = jax.nn.softmax(
                        jnp.where(mask[:, None], logits, -jnp.inf), axis=-1
                    )
                    ent = -jnp.sum(
                        jnp.where(
                            probs > 0, probs * jnp.log(jnp.maximum(probs, 1e-30)), 0
                        ),
                        -1,
                    )
                    total_entropy += jnp.sum(
                        ent * valid[:, None, start:end], axis=(0, 2)
                    )
                    total_mass += jnp.sum(
                        jnp.max(probs, -1) * valid[:, None, start:end], axis=(0, 2)
                    )
                count = jnp.maximum(valid.sum(), 1)
                norms_q = jnp.sqrt(jnp.sum(q.astype(jnp.float32) ** 2, -1))
                norms_k = jnp.sqrt(jnp.sum(k.astype(jnp.float32) ** 2, -1))
                return (
                    total_entropy / count,
                    total_mass / count,
                    jnp.sum(norms_q * valid[..., None], (0, 1)) / count,
                    jnp.sum(norms_k * valid[..., None], (0, 1)) / count,
                )

            def start_hidden(model, ids):
                return model.embedding_norm(model.embedding(ids).astype(jnp.float32))

            def trace_layer(layer, x, segments, positions):
                h = layer.attn_norm(x) if layer.attn_norm is not None else x
                qkv = layer.qkv(h).reshape(*x.shape[:2], 3, c.num_attention_heads, -1)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                local = layer.index % c.global_attn_every_n_layers != 0
                q, k = (
                    _rope(
                        q,
                        positions,
                        c.local_rope_theta if local else c.global_rope_theta,
                    ),
                    _rope(
                        k,
                        positions,
                        c.local_rope_theta if local else c.global_rope_theta,
                    ),
                )
                alpha = (
                    layer.yat_attention_alpha[...]
                    if a.role == "yat"
                    else jnp.array(1.0, jnp.float32)
                )
                attended = bidirectional_attention(
                    q,
                    k,
                    v,
                    segments,
                    radius=c.local_attention // 2 if local else None,
                    backend=c.attention_backend,
                    packed=False,
                    score=c.attention_score,
                    yat_compute_mode=c.yat_compute_mode,
                    yat_local_shards=c.yat_local_shards,
                    yat_softmax_backward=c.yat_softmax_backward,
                    yat_attention_implementation=c.yat_attention_implementation,
                    yat_attention_block_size=c.yat_attention_block_size
                    if local
                    else c.yat_global_attention_block_size,
                    alpha=alpha,
                )
                attn_delta = layer.attn_out(attended.reshape(x.shape))
                y = x + attn_delta
                h = layer.mlp_norm(y)
                sensitive = jnp.array(0, jnp.int32)
                tiles = sensitive
                pairs = sensitive
                valid_sensitive = sensitive
                if a.role == "yat":
                    hb = h.astype(jnp.bfloat16)
                    kernel = layer.wi.kernel[...].astype(jnp.bfloat16)
                    dots = jnp.matmul(hb, kernel, preferred_element_type=jnp.bfloat16)[
                        ..., : c.intermediate_size
                    ]
                    xx = hb.reshape(-1, hb.shape[-1])
                    ww = kernel[:, : c.intermediate_size].T
                    total = (
                        jnp.sum(xx * xx, -1, dtype=jnp.bfloat16)[:, None]
                        + jnp.sum(ww * ww, -1, dtype=jnp.bfloat16)[None]
                    )
                    raw = total - 2 * dots.reshape(-1, c.intermediate_size)
                    zero = jnp.all(xx == 0, -1)[:, None] & jnp.all(ww == 0, -1)[None]
                    selected = (raw <= total * 0.125) & ~zero
                    sensitive = selected.sum()
                    pairs = jnp.asarray(selected.size)
                    valid_sensitive = (selected & (segments >= 0).reshape(-1, 1)).sum()
                    rr, cc = min(16, selected.shape[0]), min(64, selected.shape[1])
                    padded = jnp.pad(
                        selected,
                        (
                            (0, (-selected.shape[0]) % rr),
                            (0, (-selected.shape[1]) % cc),
                        ),
                    )
                    tiles = (
                        padded.reshape(
                            padded.shape[0] // rr, rr, padded.shape[1] // cc, cc
                        )
                        .any((1, 3))
                        .sum()
                    )
                    ffn = yat_glu(
                        hb,
                        layer.wi.kernel[...],
                        alpha=layer.yat_alpha[...],
                        compute_mode=c.yat_ffn_compute_mode or c.yat_compute_mode,
                        local_shards=c.yat_local_shards,
                        distance_backward_block=c.yat_ffn_backward_block,
                    )
                else:
                    features, gate = jnp.split(layer.wi(h), 2, -1)
                    ffn = jax.nn.gelu(features, approximate=False) * gate
                ffn_delta = layer.wo(ffn)
                valid = segments >= 0

                def rms(z):
                    return jnp.sqrt(
                        jnp.sum(
                            jnp.where(valid[..., None], z.astype(jnp.float32) ** 2, 0)
                        )
                        / jnp.maximum(valid.sum() * z.shape[-1], 1)
                    )

                valid_features = jnp.maximum(valid.sum() * ffn.shape[-1], 1)
                zero_fraction = jnp.sum((ffn == 0) & valid[..., None]) / valid_features
                nearzero_fraction = (
                    jnp.sum((jnp.abs(ffn) < 1e-4) & valid[..., None]) / valid_features
                )
                stat = jnp.stack(
                    (
                        rms(x),
                        rms(attn_delta),
                        rms(y),
                        rms(ffn_delta),
                        sensitive,
                        tiles,
                        pairs,
                        valid_sensitive,
                        zero_fraction,
                        nearzero_fraction,
                    )
                )
                stats = attention_stats(q, k, segments, layer.index, alpha)
                return y + ffn_delta, stat, stats

            def finish(model, x, segments):
                hidden = jnp.where((segments >= 0)[..., None], model.final_norm(x), 0)
                return hidden.sum(1) / jnp.maximum(
                    (segments >= 0).sum(1, keepdims=True), 1
                )

            @nnx.jit
            def reference(model, ids):
                return model.pool(ids)

            @nnx.jit
            def trace_full(model, ids):
                segments = jnp.where(ids == c.pad_token_id, -1, 0)
                positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
                x = start_hidden(model, ids)
                stats, heads = [], []
                for layer in model.layers:
                    if c.use_remat:
                        x, stat, head = nnx.remat(
                            lambda block, h, s, p: trace_layer(block, h, s, p)
                        )(layer, x, segments, positions)
                    else:
                        x, stat, head = trace_layer(layer, x, segments, positions)
                    stats.append(stat)
                    heads.append(head)
                return finish(model, x, segments), tuple(stats), tuple(heads)

            pooled = []
            for offset in range(0, len(rows), 8):
                original = np.asarray(
                    reference(
                        model, jax.device_put(ids_host[offset : offset + 8], device)
                    )
                )
                if not np.isfinite(original).all() or np.any(
                    np.linalg.norm(original, axis=-1) <= 1e-12
                ):
                    raise ValueError("Native pooled vectors nonfinite or collapsed")
                pooled.append(original)
                np.savez(npz, pooled=np.concatenate(pooled))
                report["native_pooled_rows"] = offset + 8
                report["pooled_npz_sha256"] = digest(npz)
                write_report(a.output, report)
            vectors = np.concatenate(pooled)
            report["geometry"] = geometry(vectors)
            report["per_domain_geometry"] = {
                domain: geometry(vectors[[r["domain"] == domain for r in rows]])
                for domain in sorted({r["domain"] for r in rows})
            }
            if all("language" in r for r in rows):
                report["per_language_geometry"] = {
                    lang: geometry(vectors[[r["language"] == lang for r in rows]])
                    for lang in sorted({r["language"] for r in rows})
                }
            unit = vectors / np.maximum(
                np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
            )
            report["paired_group_cosines"] = [
                {
                    "left": rows[i]["id"],
                    "right": rows[j]["id"],
                    "group": rows[i]["group"],
                    "cosine": float(unit[i] @ unit[j]),
                }
                for i in range(len(rows))
                for j in range(i + 1, len(rows))
                if rows[i]["group"] == rows[j]["group"]
            ]
            write_report(a.output, report)
            retained_trace = []
            retained_reference = []
            for offset in range(0, len(indices), 8):
                selected_indices = indices[offset : offset + 8]
                begin = time.monotonic()
                ids = jax.device_put(ids_host[selected_indices], device)
                # Reference on identical row grouping isolates tracing from batch-context sensitivity.
                same_batch_native = np.asarray(reference(model, ids))
                initial_native = vectors[selected_indices]
                context_abs = float(np.max(np.abs(same_batch_native - initial_native)))
                context_rel = float(
                    np.linalg.norm(same_batch_native - initial_native)
                    / max(np.linalg.norm(initial_native), 1e-12)
                )
                traced_device, scalar_arrays, head_arrays_by_layer = trace_full(
                    model, ids
                )
                traced_device.block_until_ready()
                layers = []
                for index, (stat, head) in enumerate(
                    zip(scalar_arrays, head_arrays_by_layer, strict=True)
                ):
                    stat = np.asarray(stat)
                    head_arrays = [np.asarray(v) for v in head]
                    if not np.isfinite(stat).all() or any(
                        not np.isfinite(v).all() for v in head_arrays
                    ):
                        raise ValueError("Nonfinite activation diagnostic scalars")
                    head = [v.tolist() for v in head_arrays]
                    layers.append(
                        {
                            "layer": index,
                            "input_rms": float(stat[0]),
                            "attention_delta_rms": float(stat[1]),
                            "post_attention_rms": float(stat[2]),
                            "ffn_delta_rms": float(stat[3]),
                            "adaptive_sensitive_pairs_including_padding": int(stat[4]),
                            "adaptive_sensitive_tiles_including_padding": int(stat[5]),
                            "adaptive_total_pairs_including_padding": int(stat[6]),
                            "adaptive_sensitive_valid_pairs": int(stat[7]),
                            "adaptive_total_tiles_including_padding": (8 * 128 // 16)
                            * ((c.intermediate_size + 63) // 64)
                            if a.role == "yat"
                            else 0,
                            "ffn_feature_zero_fraction_valid": float(stat[8]),
                            "ffn_feature_abs_lt_1e_minus4_fraction_valid": float(
                                stat[9]
                            ),
                            "attention_entropy_nats_per_head": head[0],
                            "attention_top1_mass_per_head": head[1],
                            "mean_q_norm_per_head": head[2],
                            "mean_k_norm_per_head": head[3],
                        }
                    )
                traced = np.asarray(traced_device)
                original = same_batch_native
                drift = float(np.max(np.abs(traced - original)))
                relative = float(
                    np.linalg.norm(traced - original)
                    / max(np.linalg.norm(original), 1e-12)
                )
                retained_trace.append(traced)
                retained_reference.append(original)
                np.savez(
                    trace_npz,
                    traced=np.concatenate(retained_trace),
                    same_batch_native=np.concatenate(retained_reference),
                    initial_native=vectors[indices[: offset + 8]],
                    row_indices=np.asarray(indices[: offset + 8], dtype=np.int32),
                )
                report["trace_npz_sha256"] = digest(trace_npz)
                accepted = bool(
                    np.isfinite(traced).all()
                    and drift <= a.max_reference_absolute_drift
                    and relative <= a.max_reference_relative_l2_drift
                )
                report["batches"].append(
                    {
                        "offset": offset,
                        "row_indices": selected_indices,
                        "rows": 8,
                        "valid_tokens": int(
                            np.sum(ids_host[selected_indices] != c.pad_token_id)
                        ),
                        "reference_scope": "fresh native model.pool on identical diagnostic row grouping",
                        "initial_vs_regrouped_native_max_absolute_drift": context_abs,
                        "initial_vs_regrouped_native_relative_l2_drift": context_rel,
                        "native_batch_context_within_trace_bounds": bool(
                            context_abs <= a.max_reference_absolute_drift
                            and context_rel <= a.max_reference_relative_l2_drift
                        ),
                        "reference_max_absolute_drift": drift,
                        "reference_relative_l2_drift": relative,
                        "trace_native_parity_accepted": accepted,
                        "seconds_including_compile": time.monotonic() - begin,
                        "layers": layers,
                    }
                )
                write_report(a.output, report)
                if not accepted:
                    raise ValueError(
                        "Trace/native parity failed; do not interpret traced activations: "
                        + str(drift)
                        + "/"
                        + str(relative)
                    )
            report["attention_statistics_scope"] = (
                "same tiled YAT score/distance/FP32-softmax replay"
                if a.role == "yat"
                else "FP32 diagnostic dot-product softmax reference; actual output uses XLA attention"
            )
            report["complete"] = True
            write_report(a.output, report)
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_report(a.output, report)
        raise


if __name__ == "__main__":
    main()
