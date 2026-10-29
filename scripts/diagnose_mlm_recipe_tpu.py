"""TPU-only per-language MLM diagnostic; reporting only, never edits training data.

Uses the same held-out selection/masking policy as evaluate_encoder, regrouped
by language into padded batches of16. Regrouping can change BF16 rounding:
these scores are a diagnostic protocol, not replacements for canonical loss.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import jax
    import numpy as np
    from flax import nnx
    from flaxchat.checkpoint import (
        load_checkpoint_metadata,
        restore_model_from_checkpoint,
    )
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.encoder_data import file_hash, load_prepared_rows
    from flaxchat.common import replicate_on_mesh
    from flaxchat.mlm import mask_tokens
    from flaxchat.training import place_host_batch
    from flaxchat.runtime import runtime_identity
    from scripts.evaluate_encoder import evaluation_order

    if (
        jax.default_backend() != "tpu"
        or jax.device_count() != 8
        or jax.process_count() != 1
    ):
        raise RuntimeError("Diagnostic requires one physical eight-chip TPU host")
    metadata = load_checkpoint_metadata(args.checkpoint, step=args.step)
    identity = metadata["resolved_config"]
    cfg = EncoderConfig(**identity["encoder"])
    train, tm = load_prepared_rows(args.root / "corpus/train", cfg)
    validation, vm = load_prepared_rows(args.root / "corpus/validation", cfg)
    if file_hash(args.root / "corpus/train/manifest.json") != metadata[
        "data_manifest_identity"
    ] or any(m["tokenizer_sha256"] != metadata["tokenizer_identity"] for m in [tm, vm]):
        raise ValueError("Diagnostic data/tokenizer identity mismatch")

    def row_hash(row):
        return hashlib.sha256(
            np.asarray(row[row != cfg.pad_token_id], dtype="<i4").tobytes()
        ).digest()

    training = {row_hash(row) for row in train}
    order, labels = evaluation_order(
        args.root / "corpus/validation",
        len(validation),
        seed=2026,
        balanced_languages=True,
    )
    chosen, seen = [], set()
    for index in order:
        digest = row_hash(validation[index])
        if digest in training or digest in seen:
            continue
        seen.add(digest)
        chosen.append(int(index))
        if len(chosen) == 512:
            break
    if len(chosen) != 512 or labels is None:
        raise ValueError(
            "Diagnostic requires512 authenticated held-out language-labelled rows"
        )
    model = ModernBert(cfg, rngs=nnx.Rngs(0))
    restore_model_from_checkpoint(
        model,
        args.checkpoint,
        step=args.step,
        expected_identity={"resolved_config": identity},
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    loss_fn = nnx.jit(lambda m, x, y: m(x, y))
    slices = []
    for language in sorted({labels[i] for i in chosen}):
        selected = [i for i in chosen if labels[i] == language]
        numerator, denominator = 0.0, 0
        for start in range(0, len(selected), 16):
            ids = np.asarray(selected[start : start + 16])
            x, y = mask_tokens(
                validation[ids],
                seed=2026,
                step=0,
                example_ids=ids,
                vocab_size=cfg.vocab_size,
                mask_token_id=cfg.mask_token_id,
                special_token_ids=identity["special_token_ids"],
                probability=identity["mask_probability"],
            )
            count = int((y >= 0).sum())
            if count == 0:
                continue
            missing = 16 - len(x)
            x = np.pad(x, ((0, missing), (0, 0)), constant_values=cfg.pad_token_id)
            y = np.pad(y, ((0, missing), (0, 0)), constant_values=-1)
            loss = float(
                loss_fn(model, place_host_batch(x, mesh), place_host_batch(y, mesh))
            )
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic loss for " + language)
            numerator += loss * count
            denominator += count
        if denominator == 0:
            raise ValueError("No masked diagnostic tokens for " + language)
        slices.append(
            dict(
                language=language,
                domain="code" if language.startswith("code:") else "human_language",
                rows=len(selected),
                masked_tokens=denominator,
                masked_token_loss=numerator / denominator,
                limited_sample=len(selected) < 16,
            )
        )
    recipe = json.loads((args.root / "corpus/recipe.json").read_text())
    report = dict(
        passed=True,
        backend="tpu",
        devices=8,
        checkpoint_step=args.step,
        runtime=runtime_identity(),
        diagnostic_source_sha256=file_hash(__file__),
        selected_rows_sha256=hashlib.sha256(
            np.asarray(chosen, dtype="<i8").tobytes()
        ).hexdigest(),
        validation_manifest_sha256=file_hash(
            args.root / "corpus/validation/manifest.json"
        ),
        recipe_sha256=file_hash(args.root / "corpus/recipe.json"),
        training_language_counts=tm.get("language_counts"),
        recipe=recipe,
        protocol="same512 balanced heldout rows; language-grouped padded batch16; fixed seed2026 masks",
        limitations=[
            "Small per-language samples; diagnostic only",
            "Batch regrouping may change BF16 rounding",
            "Exact-row exclusion does not prove semantic decontamination",
            "Training source shares differ from temperature-sampled exposure; no reweighting from this report alone",
        ],
        slices=slices,
        masked_token_loss=sum(
            s["masked_token_loss"] * s["masked_tokens"] for s in slices
        )
        / sum(s["masked_tokens"] for s in slices),
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
