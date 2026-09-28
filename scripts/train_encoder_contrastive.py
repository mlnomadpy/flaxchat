"""Bounded end-to-end dual-encoder contrastive stage from a pinned MLM checkpoint.

The parent contributes model weights only. This stage owns a fresh optimizer,
schedule, dataset cursor and exact-resume checkpoint identity. It expects a
verified prepared contrastive dataset with explicit train/dev isolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax

from flaxchat.checkpoint import (
    create_checkpoint_manager, load_checkpoint_metadata,
    restore_model_from_checkpoint, save_checkpoint,
)
from flaxchat.common import replicate_on_mesh
from flaxchat.contrastive import symmetric_infonce
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.encoder_data import EpochRows, file_hash
from flaxchat.runtime import runtime_identity
from flaxchat.training import apply_gradients_if_finite, gather_process_metadata, place_host_batch
from scripts.prepare_encoder_contrastive import load_tokenized
from scripts.train_encoder import validate_resume_cursor


PARENT_STEP = 48236
PARENT_METADATA_SHA256 = "35cff96b97c0a9d5ba401c39545dfeff62e1c73e527d5f5661ba0960042bf4f4"


def metadata_digest(value):
    # load_checkpoint_metadata binds a redundant selected-directory `step` to
    # older artifacts; the committed manifest hashes the stored payload.
    payload = {key: item for key, item in value.items() if key != "step"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = (
        "scripts/train_encoder_contrastive.py",
        "scripts/prepare_encoder_contrastive.py",
        "scripts/train_encoder.py",
        "flaxchat/contrastive.py",
        "flaxchat/encoder.py",
        "flaxchat/yat.py",
        "flaxchat/checkpoint.py",
        "flaxchat/encoder_data.py",
        "flaxchat/training.py",
        "flaxchat/common.py",
        "flaxchat/runtime.py",
    )
    return hashlib.sha256("".join(file_hash(root / name) for name in names).encode()).hexdigest()


def schedule(peak, steps, warmup, final_ratio):
    if (not np.isfinite(peak) or peak <= 0 or not 1 <= steps <= 5000
            or not 0 <= warmup < steps or not 0 <= final_ratio <= 1):
        raise ValueError("Invalid bounded contrastive schedule")
    if warmup:
        return optax.warmup_cosine_decay_schedule(0.0, peak, warmup, steps,
            end_value=peak * final_ratio)
    return optax.cosine_decay_schedule(peak, steps, alpha=final_ratio)


def _same_path(a, b):
    if a.rstrip("/") == b.rstrip("/"):
        return True
    if not a.startswith("gs://") and not b.startswith("gs://"):
        return Path(a).resolve() == Path(b).resolve()
    return False


def expected_checkpoint_identity(recipe, tokenizer_hash, data_manifest_sha, source_hash):
    """The complete immutable stage identity passed to checkpoint restore."""
    return dict(resolved_config=recipe, tokenizer=tokenizer_hash,
                data_manifest=data_manifest_sha, source_python_sha256=source_hash)


def checked_resume_cursor(cursor, *, steps, stop):
    start, updates = validate_resume_cursor(cursor, horizon=steps, stop=stop)
    if updates != start:
        raise ValueError("Contrastive checkpoint has skipped optimizer updates")
    return start


def load_parent_for_preflight(metadata_file, manifest_file):
    """Use a saved metadata copy only when its committed manifest authenticates it."""
    metadata = json.loads(Path(metadata_file).read_text())
    manifest = json.loads(Path(manifest_file).read_text())
    digest = metadata_digest(metadata)
    if (manifest.get("step") != PARENT_STEP or manifest.get("metadata_sha256") != digest
            or digest != PARENT_METADATA_SHA256):
        raise ValueError("Offline parent metadata is not pinned by committed checkpoint manifest")
    return metadata


def run(args):
    if args.distributed and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    if args.parent_step != PARENT_STEP:
        raise ValueError(f"This stage pins final MLM parent step {PARENT_STEP}")
    if args.resume and args.preflight_only:
        raise ValueError("Preflight does not restore live optimizer state")
    if args.output and _same_path(args.parent_checkpoint, args.output):
        raise ValueError("Contrastive checkpoints require a new output location")
    if (args.seed < 0 or args.batch_size < 2 or args.batch_size % jax.device_count()
            or args.save_every < 1 or args.keep_checkpoints < 1
            or not 1 <= args.sequence_length <= 512
            or not np.isfinite(args.temperature) or not 0 < args.temperature <= 1
            or not np.isfinite(args.weight_decay) or not 0 <= args.weight_decay <= 1):
        raise ValueError("Invalid contrastive batch, sequence, temperature or checkpoint policy")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        raise ValueError("stop-after must be inside the fixed training horizon")
    if jax.process_count() > 1 and not args.output.startswith("gs://"):
        raise ValueError("Multi-host checkpoints require GCS")
    if not args.preflight_only and jax.default_backend() != "tpu":
        raise ValueError("Contrastive model training requires a physical TPU")

    learning_rate = schedule(args.learning_rate, args.steps, args.warmup_steps, args.final_lr_ratio)
    if bool(args.parent_metadata_file) != bool(args.parent_manifest_file):
        raise ValueError("Offline preflight requires both parent metadata and manifest files")
    if args.parent_metadata_file and not args.preflight_only:
        raise ValueError("Live training must read parent checkpoint metadata from its source")
    parent = (load_parent_for_preflight(args.parent_metadata_file, args.parent_manifest_file)
              if args.parent_metadata_file else
              load_checkpoint_metadata(args.parent_checkpoint, step=PARENT_STEP))
    if metadata_digest(parent) != PARENT_METADATA_SHA256:
        raise ValueError("Final MLM parent metadata differs from the pinned step-48236 receipt")
    if parent.get("model_family") != "modernbert" or not isinstance(parent.get("resolved_config", {}).get("encoder"), dict):
        raise ValueError("Parent is not a committed ModernBERT MLM checkpoint")
    config = EncoderConfig(**parent["resolved_config"]["encoder"])
    if (config.ffn_type != "yat_glu" or config.attention_score != "yat_softmax"
            or config.yat_bias != 1.0 or config.yat_epsilon != .01 or not config.yat_alpha_trainable):
        raise ValueError("Parent does not satisfy fixed-bias trainable-alpha YAT architecture")
    tokenizer_hash = file_hash(args.tokenizer)
    if parent.get("tokenizer_identity") != tokenizer_hash:
        raise ValueError("Parent and contrastive tokenizer differ")
    if args.sequence_length > config.max_position_embeddings:
        raise ValueError("Contrastive sequence length exceeds parent model")
    arrays, manifest = load_tokenized(args.data, args.tokenizer, split="train",
                                      sequence_length=args.sequence_length)
    if len(arrays["query_tokens"]) < args.batch_size:
        raise ValueError("Fewer training pairs than one global batch")
    data_manifest_sha = file_hash(Path(args.data) / "manifest.json")
    parent_receipt = dict(path=args.parent_checkpoint, step=PARENT_STEP,
        metadata_sha256=PARENT_METADATA_SHA256,
        policy="model_weights_only; fresh_optimizer_schedule_and_data_cursor")
    recipe = dict(encoder=parent["resolved_config"]["encoder"], parent=parent_receipt,
        data_manifest_sha256=data_manifest_sha, seed=args.seed, steps=args.steps,
        global_batch_size=args.batch_size, sequence_length=args.sequence_length,
        learning_rate=args.learning_rate, final_lr_ratio=args.final_lr_ratio,
        warmup_steps=args.warmup_steps, schedule="warmup_cosine",
        temperature=args.temperature, objective="symmetric_infonce_global_in_batch",
        pooling="mean_l2", negatives="global_batch_mask_duplicate_text_and_known_positive_groups",
        shuffle="replayable_epoch_permutation", gradient_clip_norm=1.0,
        weight_decay=args.weight_decay, optimizer="adamw_fp32_state",
        runtime=runtime_identity())
    identity_source = source_identity()
    metadata = dict(model_family="modernbert_contrastive_encoder",
        resolved_config=recipe, tokenizer_identity=tokenizer_hash,
        data_manifest_identity=data_manifest_sha, source_python_sha256=identity_source,
        dataset=manifest.get("dataset"), dataset_revision=manifest.get("revision"))
    expected = expected_checkpoint_identity(recipe, tokenizer_hash,
        data_manifest_sha, identity_source)
    workers = gather_process_metadata(expected)
    if any(worker != workers[0] for worker in workers):
        raise ValueError("Contrastive workers disagree on stage identity")
    if args.preflight_only:
        if jax.process_index() == 0:
            print(json.dumps(dict(event="contrastive_preflight", pairs=len(arrays["query_tokens"]),
                parent=parent_receipt, data_manifest_sha256=data_manifest_sha,
                source_python_sha256=identity_source, devices=jax.device_count())), flush=True)
        return

    model = ModernBert(config, rngs=nnx.Rngs(args.seed))
    if not args.resume:
        pinned = restore_model_from_checkpoint(model, args.parent_checkpoint, step=PARENT_STEP)
        if metadata_digest(pinned) != parent_receipt["metadata_sha256"]:
            raise ValueError("MLM parent metadata changed during model-only restore")
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=args.weight_decay,
            mask=lambda params: jax.tree.map(lambda x: x.ndim > 1, params)),
    ), wrt=nnx.Param)
    start = 0
    stop = args.stop_after or args.steps
    if args.resume:
        _, cursor = restore_model_from_checkpoint(model, args.output, optimizer=optimizer,
            expected_identity=expected, load_training_state=True)
        if cursor is None:
            raise ValueError("Missing contrastive training cursor")
        start = checked_resume_cursor(cursor, steps=args.steps, stop=stop)
        optimizer.step[...] = jnp.asarray(start, dtype=optimizer.step.dtype)
    if start == stop:
        return
    nnx.update(optimizer, replicate_on_mesh(nnx.state(optimizer), mesh))
    manager = create_checkpoint_manager(args.output, max_to_keep=args.keep_checkpoints,
                                        async_checkpointing=False)
    if not args.resume and manager.latest_step() is not None:
        manager.close()
        raise ValueError("Contrastive output already contains checkpoints")

    @nnx.jit
    def update(m, o, query, document, q_id, d_id, group_id):
        def objective(inner):
            return symmetric_infonce(inner.pool(query), inner.pool(document),
                q_id, d_id, group_id, temperature=args.temperature,
                expected_global_batch=args.batch_size)
        loss, grads = nnx.value_and_grad(objective)(m)
        accepted = apply_gradients_if_finite(m, o, grads, loss)
        return loss, accepted

    sampler = EpochRows(len(arrays["query_tokens"]), args.seed, shuffle=True)
    local_batch = args.batch_size // jax.process_count()
    if jax.process_index() == 0:
        print(json.dumps(dict(event="contrastive_run_config", recipe=recipe,
            devices=jax.device_count(), processes=jax.process_count(),
            backend=jax.default_backend(), pairs=len(arrays["query_tokens"]))), flush=True)
    try:
        for step in range(start, stop):
            begun = time.monotonic()
            indices = sampler.batch(step, args.batch_size)
            indices = indices[jax.process_index() * local_batch:(jax.process_index() + 1) * local_batch]
            x = [place_host_batch(arrays[name][indices], mesh) for name in (
                "query_tokens", "document_tokens", "query_text_ids",
                "document_text_ids", "positive_group_ids")]
            loss, accepted = update(model, optimizer, *x)
            if not bool(accepted) or not np.isfinite(float(loss)):
                raise FloatingPointError(f"Rejected contrastive update {step + 1}")
            if jax.process_index() == 0:
                print(json.dumps(dict(event="contrastive_train_step", step=step + 1,
                    loss=float(loss), updated=True, pairs=args.batch_size,
                    seconds=time.monotonic() - begun,
                    includes_compilation=step == start)), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == stop:
                save_checkpoint(manager, step + 1, model, optimizer, metadata,
                    training_state=dict(completed_steps=np.array([step + 1], np.int32),
                                        optimizer_updates=np.array([int(optimizer.step[...])], np.int32)))
                manager.wait_until_finished()
                if jax.process_index() == 0:
                    print(json.dumps(dict(event="contrastive_checkpoint", step=step + 1)), flush=True)
    finally:
        manager.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--parent-step", type=int, default=PARENT_STEP)
    p.add_argument("--parent-metadata-file", type=Path, help="Manifest-authenticated saved metadata for offline preflight only")
    p.add_argument("--parent-manifest-file", type=Path, help="Committed manifest authenticating offline metadata")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--temperature", type=float, default=.05)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--final-lr-ratio", type=float, default=.1)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
