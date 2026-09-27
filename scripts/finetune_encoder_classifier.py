"""End-to-end encoder classification with exact-resume checkpoints and split isolation."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
import numpy as np
import optax

from flaxchat.common import replicate_on_mesh
from flaxchat.checkpoint import (
    create_checkpoint_manager,
    load_checkpoint_metadata,
    restore_model_from_checkpoint,
    save_checkpoint,
)
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.encoder_data import EpochRows, file_hash
from flaxchat.encoder_tasks import (
    EncoderClassifier,
    EncoderTokenClassifier,
    classification_statistics,
)
from flaxchat.runtime import runtime_identity
from flaxchat.training import (
    apply_gradients_if_finite,
    place_host_batch,
    gather_process_metadata,
)
from scripts.prepare_encoder_classification import load_rows
from scripts.train_encoder import (
    load_pretrained,
    pretrained_inventory,
    validate_resume_cursor,
)


def learning_rate(peak, steps, warmup):
    if not np.isfinite(peak) or peak <= 0 or steps < 1 or not 0 <= warmup < steps:
        raise ValueError("Invalid classifier learning-rate schedule")
    decay = optax.linear_schedule(peak, 0.0, steps - warmup)
    return (
        optax.join_schedules(
            [optax.linear_schedule(0.0, peak, warmup), decay], [warmup]
        )
        if warmup
        else decay
    )


def source_identity():
    root = Path(__file__).parents[1]
    files = [
        "scripts/finetune_encoder_classifier.py",
        "scripts/prepare_encoder_classification.py",
        "scripts/prepare_encoder_ner.py",
        "scripts/evaluate_encoder_ner.py",
        "flaxchat/ner.py",
        "scripts/train_encoder.py",
        "flaxchat/encoder.py",
        "flaxchat/yat.py",
        "flaxchat/encoder_tasks.py",
        "flaxchat/encoder_data.py",
        "flaxchat/common.py",
        "flaxchat/training.py",
        "flaxchat/checkpoint.py",
    ]
    return hashlib.sha256(
        "".join(file_hash(root / f) for f in files).encode()
    ).hexdigest()


def train(a):
    if (
        a.seed < 0
        or a.save_every < 1
        or a.batch_size < 1
        or a.batch_size % jax.device_count()
    ):
        raise ValueError(
            "Require nonnegative seed, positive save interval and device-divisible batch"
        )
    if a.stop_after is not None and not 0 < a.stop_after <= a.steps:
        raise ValueError("Invalid stop-after horizon")
    if a.shared_local_checkpoints and jax.default_backend() != "cpu":
        raise ValueError("Shared local classifier checkpoints are CPU-test only")
    if (
        jax.process_count() > 1
        and not a.output.startswith("gs://")
        and not a.shared_local_checkpoints
    ):
        raise ValueError("Multi-host classifier checkpoints require GCS")
    if bool(a.encoder_checkpoint) != (a.encoder_step is not None):
        raise ValueError(
            "Candidate encoder checkpoint requires an explicit committed step"
        )
    schedule = learning_rate(a.learning_rate, a.steps, a.warmup_steps)
    raw = json.loads(Path(a.config).read_text())
    overrides = dict(compute_dtype=a.dtype, residual_dtype="float32", use_remat=True)
    config = (
        EncoderConfig.from_hf(raw, **overrides)
        if raw.get("model_type")
        else EncoderConfig(**(raw | overrides))
    )
    token_task = getattr(a, "task", "sequence_classification") == "token_classification"
    if token_task:
        from scripts.prepare_encoder_ner import load_rows as load_ner_rows

        tokens, labels, _, data = load_ner_rows(a.data, config, expected_split="train")
    else:
        tokens, labels, data = load_rows(a.data, config, expected_split="train")
    if len(tokens) < a.batch_size:
        raise ValueError("Training set is smaller than global batch")
    origin = {}
    if a.pretrained:
        if file_hash(Path(a.pretrained) / "tokenizer.json") != data["tokenizer_sha256"]:
            raise ValueError("Training tokenizer differs from pretrained tokenizer")
        snapshot_config = EncoderConfig.from_hf(
            json.loads((Path(a.pretrained) / "config.json").read_text()), **overrides
        )
        if snapshot_config != config:
            raise ValueError("Pretrained architecture mismatch")
        origin["released_weights"] = pretrained_inventory(a.pretrained, config)
    if a.encoder_checkpoint:
        origin["checkpoint"] = dict(
            path=a.encoder_checkpoint,
            step=a.encoder_step,
            metadata=load_checkpoint_metadata(
                a.encoder_checkpoint, step=a.encoder_step
            ),
        )
        source_config = EncoderConfig(
            **origin["checkpoint"]["metadata"]["resolved_config"]["encoder"]
        )
        # Runtime projection settings may differ; architecture/tokenizer must not.
        for field in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "global_attn_every_n_layers",
            "local_attention",
            "global_rope_theta",
            "local_rope_theta",
            "norm_eps",
            "pad_token_id",
            "mask_token_id",
        ):
            if getattr(source_config, field) != getattr(config, field):
                raise ValueError("Candidate encoder architecture mismatch")
        if (
            origin["checkpoint"]["metadata"]["tokenizer_identity"]
            != data["tokenizer_sha256"]
        ):
            raise ValueError("Candidate encoder tokenizer mismatch")
    recipe = dict(
        encoder=asdict(config),
        num_labels=data["num_labels"],
        seed=a.seed,
        steps=a.steps,
        batch_size=a.batch_size,
        learning_rate=a.learning_rate,
        warmup_steps=a.warmup_steps,
        schedule="linear_warmup_linear_decay",
        weight_decay=0.01,
        decay_mask="ndim>1",
        clip_norm=1.0,
        pooling="mean",
        classifier_dropout=0.0,
        shuffle=True,
        origin=origin,
        runtime=runtime_identity(),
    )
    metadata = dict(
        model_family=(
            "modernbert_token_classifier"
            if token_task
            else "modernbert_sequence_classifier"
        ),
        resolved_config=recipe,
        tokenizer_identity=data["tokenizer_sha256"],
        data_manifest_identity=file_hash(Path(a.data) / "manifest.json"),
        source_python_sha256=source_identity(),
        dataset=data["dataset"],
        dataset_revision=data["revision"],
    )
    if token_task:
        recipe.update(
            task="token_classification",
            pooling="none",
            alignment=data["alignment"],
            label_names=data["label_names"],
        )
    expected = dict(
        resolved_config=recipe,
        tokenizer=data["tokenizer_sha256"],
        data_manifest=metadata["data_manifest_identity"],
        source_python_sha256=metadata["source_python_sha256"],
    )
    workers = gather_process_metadata(expected)
    if any(x != workers[0] for x in workers):
        raise ValueError("Classifier workers disagree on inputs or recipe")
    encoder = ModernBert(config, rngs=nnx.Rngs(a.seed))
    if not a.resume:
        if a.encoder_checkpoint:
            restore_model_from_checkpoint(
                encoder, a.encoder_checkpoint, step=a.encoder_step
            )
        elif a.pretrained:
            if load_pretrained(encoder, a.pretrained) != origin["released_weights"]:
                raise ValueError("Snapshot changed after preflight")
    adapter = EncoderTokenClassifier if token_task else EncoderClassifier
    model = adapter(encoder, data["num_labels"], rngs=nnx.Rngs(a.seed + 1))
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                schedule,
                weight_decay=0.01,
                mask=lambda params: jax.tree.map(lambda x: x.ndim > 1, params),
            ),
        ),
        wrt=nnx.Param,
    )
    start = 0
    if a.resume:
        metadata, cursor = restore_model_from_checkpoint(
            model,
            a.output,
            optimizer=optimizer,
            expected_identity=expected,
            load_training_state=True,
        )
        if cursor is None:
            raise ValueError("Missing classifier training cursor")
        start, updates = validate_resume_cursor(
            cursor, horizon=a.steps, stop=a.stop_after or a.steps
        )
        optimizer.step[...] = jnp.asarray(updates, dtype=optimizer.step.dtype)
        if updates != start:
            raise ValueError("Classifier checkpoint contains skipped optimizer updates")
    end = a.stop_after or a.steps
    if start == end:
        return
    nnx.update(optimizer, replicate_on_mesh(nnx.state(optimizer), mesh))
    manager = create_checkpoint_manager(a.output, async_checkpointing=False)
    if not a.resume and manager.latest_step() is not None:
        manager.close()
        raise ValueError("Classifier output already has checkpoints")

    @nnx.jit
    def update(m, o, x, y):
        def objective(m):
            return (
                m.statistics(x, y)[0]
                if token_task
                else classification_statistics(m(x), y)[0]
            )

        loss, grads = nnx.value_and_grad(objective)(m)
        accepted = apply_gradients_if_finite(m, o, grads, loss)
        return loss, accepted

    rows = EpochRows(len(tokens), a.seed, shuffle=True)
    local = a.batch_size // jax.process_count()
    if jax.process_index() == 0:
        print(
            json.dumps(
                dict(
                    event="classifier_run_config",
                    devices=jax.device_count(),
                    processes=jax.process_count(),
                    backend=jax.default_backend(),
                    recipe=recipe,
                )
            ),
            flush=True,
        )
    try:
        for step in range(start, end):
            begin = time.monotonic()
            ids = rows.batch(step, a.batch_size)[
                jax.process_index() * local : (jax.process_index() + 1) * local
            ]
            loss, accepted = update(
                model,
                optimizer,
                place_host_batch(tokens[ids], mesh),
                place_host_batch(labels[ids], mesh),
            )
            if not bool(accepted):
                raise FloatingPointError("Rejected classifier update")
            if jax.process_index() == 0:
                print(
                    json.dumps(
                        dict(
                            event="classifier_train_step",
                            step=step + 1,
                            loss=float(loss),
                            updated=True,
                            examples=a.batch_size,
                            seconds=time.monotonic() - begin,
                            includes_compilation=step == start,
                        )
                    ),
                    flush=True,
                )
            if (step + 1) % a.save_every == 0 or step + 1 == end:
                save_checkpoint(
                    manager,
                    step + 1,
                    model,
                    optimizer,
                    metadata,
                    training_state=dict(
                        completed_steps=np.array([step + 1], np.int32),
                        optimizer_updates=np.array(
                            [int(optimizer.step[...])], np.int32
                        ),
                    ),
                )
                manager.wait_until_finished()
    finally:
        manager.close()


def evaluate(a):
    if jax.process_count() != 1:
        raise ValueError("Classifier evaluation currently requires one host")
    if a.batch_size < 1 or a.batch_size % jax.device_count():
        raise ValueError("Evaluation batch must divide across devices")
    metadata = load_checkpoint_metadata(a.output, step=a.checkpoint_step)
    if metadata.get("model_family") == "modernbert_token_classifier":
        from scripts.evaluate_encoder_ner import evaluate_ner

        return evaluate_ner(a, metadata)
    if metadata.get("model_family") != "modernbert_sequence_classifier":
        raise ValueError("Not a classifier checkpoint")
    recipe = metadata["resolved_config"]
    config = EncoderConfig(**recipe["encoder"])
    inputs = []
    identities = set()
    if not a.eval_data:
        raise ValueError("At least one evaluation dataset is required")
    for directory in a.eval_data:
        tokens, labels, data = load_rows(directory, config)
        if data["split"] == "train":
            raise ValueError("Evaluation requires validation or test split")
        if (
            data["tokenizer_sha256"] != metadata["tokenizer_identity"]
            or data["num_labels"] != recipe["num_labels"]
            or data["dataset"] != metadata["dataset"]
            or data["revision"] != metadata["dataset_revision"]
        ):
            raise ValueError(
                "Evaluation dataset/labels/tokenizer differs from training"
            )
        identity = (data["split"], data["language"])
        if identity in identities:
            raise ValueError("Duplicate evaluation split/language")
        identities.add(identity)
        inputs.append((directory, tokens, labels, data))
    model = EncoderClassifier(
        ModernBert(config, rngs=nnx.Rngs(0)), recipe["num_labels"], rngs=nnx.Rngs(1)
    )
    restore_model_from_checkpoint(model, a.output, step=metadata["step"])
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    fn = nnx.jit(lambda m, x, y: classification_statistics(m(x), y))
    reports = []
    for directory, tokens, labels, data in inputs:
        correct = count = 0
        total_loss = 0.0
        for start in range(0, len(tokens), a.batch_size):
            n = min(a.batch_size, len(tokens) - start)
            x = np.full((a.batch_size, tokens.shape[1]), config.pad_token_id, np.int32)
            x[:n] = tokens[start : start + n]
            y = np.full(a.batch_size, -1, np.int32)
            y[:n] = labels[start : start + n]
            loss, right, valid = fn(
                model, place_host_batch(x, mesh), place_host_batch(y, mesh)
            )
            if not np.isfinite(float(loss)) or int(valid) != n:
                raise ValueError("Invalid classifier evaluation batch")
            correct += int(right)
            count += int(valid)
            total_loss += float(loss) * n
        reports.append(
            dict(
                data_manifest=data,
                data_manifest_sha256=file_hash(Path(directory) / "manifest.json"),
                examples=count,
                correct=correct,
                accuracy=correct / count,
                loss=total_loss / count,
            )
        )
    report = dict(
        scope="end_to_end_encoder_classification",
        production_quality_qualified=False,
        checkpoint=a.output,
        checkpoint_step=metadata["step"],
        training_metadata=metadata,
        evaluator_source_sha256=source_identity(),
        runtime=runtime_identity(),
        datasets=reports,
    )
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                event="classifier_evaluation",
                scores=[
                    {k: r[k] for k in ("examples", "accuracy", "loss")} for r in reports
                ],
            )
        ),
        flush=True,
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "evaluate"], default="train")
    p.add_argument(
        "--task",
        choices=["sequence_classification", "token_classification"],
        default="sequence_classification",
    )
    p.add_argument("--config")
    p.add_argument("--data")
    p.add_argument("--output", required=True)
    p.add_argument("--pretrained")
    p.add_argument("--encoder-checkpoint")
    p.add_argument("--encoder-step", type=int)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--shared-local-checkpoints",
        action="store_true",
        help="Opt in to one shared local filesystem for CPU process tests only",
    )
    p.add_argument("--stop-after", type=int)
    p.add_argument("--eval-data", nargs="+")
    p.add_argument("--checkpoint-step", type=int)
    p.add_argument("--report")
    return p


def main():
    p = parser()
    a = p.parse_args()
    if a.mode == "train":
        if not a.config or not a.data:
            p.error("Training requires --config and --data")
        train(a)
    else:
        if not a.eval_data or not a.report or a.checkpoint_step is None:
            p.error("Evaluation requires --eval-data, --report and --checkpoint-step")
        evaluate(a)


if __name__ == "__main__":
    main()
