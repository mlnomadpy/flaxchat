"""Pinned multilingual XNLI frozen-encoder probe, not a production certificate.

Fits one English ridge head and evaluates that same head in every language.
The protocol is intentionally distinct from end-to-end fine-tuned XNLI scores.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pair_features(premise, hypothesis):
    if (
        premise.shape != hypothesis.shape
        or premise.ndim != 2
        or not np.isfinite(premise).all()
        or not np.isfinite(hypothesis).all()
    ):
        raise ValueError("Expected matching finite embedding matrices")
    return np.concatenate(
        [premise, hypothesis, np.abs(premise - hypothesis), premise * hypothesis],
        axis=1,
    )


def fit_probe(features, labels, regularization=1.0):
    if (
        len(features) != len(labels)
        or not np.isfinite(features).all()
        or set(np.unique(labels)) != {0, 1, 2}
    ):
        raise ValueError("Require finite training features and all three classes")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Positive regularization required")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-5)
    x = (features - mean) / scale
    x = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    targets = np.eye(3, dtype=x.dtype)[labels]
    # Dual ridge is cheaper for the bounded 1024-example probe.
    coefficients = x.T @ np.linalg.solve(
        x @ x.T + regularization * np.eye(len(x)), targets
    )
    return mean, scale, coefficients


def score_probe(features, labels, probe):
    mean, scale, coefficients = probe
    if (
        len(features) != len(labels)
        or not len(labels)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Require nonempty finite evaluation")
    x = (features - mean) / scale
    predictions = np.concatenate([x, np.ones((len(x), 1))], axis=1) @ coefficients
    correct = predictions.argmax(axis=1) == labels
    n = len(labels)
    accuracy = float(correct.mean())
    z = 1.959963984540054
    center = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z
        * np.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n))
        / (1 + z * z / n)
    )
    return dict(
        accuracy=accuracy,
        examples=n,
        wilson_95=[float(center - half), float(center + half)],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("Positive batch size required")
    import jax
    from flax import nnx
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.common import replicate_on_mesh
    from flaxchat.training import place_host_batch
    from scripts.train_encoder import load_pretrained, pretrained_inventory
    from flaxchat.checkpoint import (
        load_checkpoint_metadata,
        restore_model_from_checkpoint,
    )
    from flaxchat.runtime import runtime_identity

    if jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise ValueError("This bounded probe requires one physical TPU host")
    root = Path(a.benchmark)
    manifest = json.loads(root.with_suffix(".json").read_text())
    if hashlib.sha256(root.read_bytes()).hexdigest() != manifest["npz_sha256"]:
        raise ValueError("Benchmark checksum mismatch")
    if a.checkpoint:
        metadata = load_checkpoint_metadata(a.checkpoint)
        config = EncoderConfig(**metadata["resolved_config"]["encoder"])
    else:
        config = EncoderConfig.from_hf(
            json.loads((Path(a.snapshot) / "config.json").read_text()),
            compute_dtype="bfloat16",
            residual_dtype="float32",
        )
    model = ModernBert(config, rngs=nnx.Rngs(0))
    if a.checkpoint:
        restore_model_from_checkpoint(model, a.checkpoint)
        identity = {"checkpoint": a.checkpoint, "metadata": metadata}
    else:
        identity = {"weights": pretrained_inventory(a.snapshot)}
        if load_pretrained(model, a.snapshot) != identity["weights"]:
            raise ValueError("Weights changed during loading")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    if a.batch_size % jax.device_count():
        raise ValueError("Batch must divide across devices")
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    pool = nnx.jit(lambda m, x: m.pool(x))
    data = np.load(root, allow_pickle=False)
    features = {}
    for split in ("train", "test"):
        for language in ["en"] if split == "train" else manifest["languages"]:
            embedded = []
            for side in ("premise", "hypothesis"):
                tokens = data[f"{split}_{language}_{side}"]
                chunks = []
                for start in range(0, len(tokens), a.batch_size):
                    x = tokens[start : start + a.batch_size]
                    batch = np.pad(
                        x,
                        ((0, a.batch_size - len(x)), (0, 0)),
                        constant_values=config.pad_token_id,
                    )
                    chunks.append(
                        np.asarray(pool(model, place_host_batch(batch, mesh)))[: len(x)]
                    )
                embedded.append(np.concatenate(chunks).astype(np.float64))
            features[f"{split}_{language}"] = pair_features(*embedded)
    probe = fit_probe(features["train_en"], data["train_labels"])
    scores = {
        lang: score_probe(features[f"test_{lang}"], data["test_labels"], probe)
        for lang in manifest["languages"]
    }
    report = dict(
        scope="xnli_frozen_encoder_probe",
        production_quality_qualified=False,
        dataset=manifest,
        model=identity,
        runtime=runtime_identity(),
        scores=scores,
        macro_accuracy=float(np.mean([s["accuracy"] for s in scores.values()])),
        limitations=[
            "Bounded test subset, fixed English linear probe; not published fine-tuned XNLI.",
            "Does not qualify retrieval, token classification, long-context quality, or domain suitability.",
            "No claim of decontamination from the upstream pretrained model.",
        ],
    )
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("model", "dataset")}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
