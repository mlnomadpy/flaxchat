"""TPU-only YAT embedding fine-tuning with mined and in-batch negatives.

Each stage has an immutable data/model/optimizer identity and exact Orbax
resume. The public step-12,000 checkpoint supplies the initial model weights;
later stages can load weights from a prior stage without reusing its optimizer.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax

from flaxchat.checkpoint import (create_checkpoint_manager,
                                 restore_model_from_checkpoint, save_checkpoint, load_checkpoint_metadata)
from flaxchat.common import replicate_on_mesh
from flaxchat.contrastive import hard_negative_infonce
from flaxchat.embedding_objective import configure_objective_state, objective_loss_arguments
from flaxchat.embedding_contract import (source_identity, validate_parent_metadata, validate_cursor,
                                        validate_optimizer_cursor, emit_completion_status)
from flaxchat.runtime import runtime_identity
from flaxchat.profiling import TrainingTrace
from flaxchat.embedding_data import ReplayRows, HomogeneousSchedule, ExposureTracker
from flaxchat.embedding_telemetry import InvocationTelemetry, PHASES, reduce_step
from flaxchat.embedding_quality import (retrieval_metrics, quality_gate, sts_metrics, programming_language_slices)
from flaxchat.embedding_gradient_cache import nnx_cached_pool
from flaxchat.embedding_recovery import (evaluation_due, selection_record, reconcile_best,
    require_save_success, validate_resume_evaluation, admit_quality_migration, migration_restore_identity)
from jax.experimental import multihost_utils
from flaxchat.public_encoder import load_public_encoder
from flaxchat.training import (apply_gradients_if_finite, gather_process_metadata,
                               place_host_batch)


from flaxchat.embedding_stage import (TRAIN_ARRAYS, add_stage_arguments, prepare_stage)


def _batch(step: int, split: str, data: dict, counts: dict[str, int],
           seed: int, offsets: dict[str, int], samplers: dict | None = None,
           *, local_slice=None, source_cursor=None):
    if samplers is None:
        raise ValueError("Replayable source samplers are required")
    names = sorted(counts)
    indices = {name: samplers[name].batch(step if source_cursor is None else source_cursor, counts[name]) for name in names}
    total = sum(counts.values())
    order = np.random.default_rng(seed + step * 1_000_003 + 931).permutation(total)
    if local_slice is not None:
        order = order[local_slice]
    result = {}
    boundaries = np.cumsum([0] + [counts[name] for name in names])
    for key in TRAIN_ARRAYS:
        sample = data[names[0]][split][key]
        values = np.empty((len(order),) + sample.shape[1:], dtype=sample.dtype)
        for number, name in enumerate(names):
            selected = (order >= boundaries[number]) & (order < boundaries[number + 1])
            row_indices = indices[name][order[selected] - boundaries[number]]
            values[selected] = data[name][split][key][row_indices]
        result[key] = values
    return result


def run(args):
    telemetry = InvocationTelemetry()
    if args.distributed and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Model training requires a physical TPU")
    admitted = prepare_stage(args, device_count=jax.device_count(), process_count=jax.process_count())
    objective_identity = admitted["admission_receipt"].get("contrastive_objective")
    parent = Path(args.parent_public)
    parent_hashes, encoder_config = admitted['parent_hashes'], admitted['encoder_config']
    data_items, data = admitted['data_items'], admitted['data']
    weights, counts = admitted['weights'], admitted['counts']
    query_lengths, document_lengths = admitted['query_lengths'], admitted['document_lengths']
    mixture_receipt, dev_indices = admitted['mixture_receipt'], admitted['dev_indices']
    sts_data, sts_receipts = admitted['sts_data'], admitted['sts_receipts']
    retrieval_data, retrieval_receipts = admitted['retrieval_data'], admitted['retrieval_receipts']
    offsets = {name: 0 for name in data}
    source_receipt = source_identity(Path(__file__).resolve().parents[1])
    source_hash = source_receipt["sha256"]
    data_manifests = {name: digest for name, _, _, digest in data_items}
    data_digest = hashlib.sha256(json.dumps(data_manifests, sort_keys=True).encode()).hexdigest()
    parent_receipt = {"public_files_sha256": parent_hashes,
                      "policy": "model_weights_only; fresh_optimizer_schedule_and_data_cursor"}
    parent_expected = None
    if args.parent_checkpoint:
        parent_metadata = load_checkpoint_metadata(args.parent_checkpoint, args.parent_step, include_receipt=True)
        parent_receipt.update(validate_parent_metadata(parent_metadata, admitted['parent_encoder_config'],
                                                       parent_hashes["tokenizer.json"]))
        parent_receipt["checkpoint"] = args.parent_checkpoint
        parent_receipt["committed_artifact"] = parent_metadata["committed_receipt"]
        parent_expected = {"resolved_config": parent_metadata["resolved_config"],
                           "tokenizer": parent_metadata["tokenizer_identity"],
                           "source_python_sha256": parent_metadata.get("source_python_sha256", "unavailable"),
                           "data_manifest": parent_metadata.get("data_manifest_identity", "unavailable")}
    recipe = {"encoder": encoder_config, "runtime": runtime_identity(),
              "source_receipt": source_receipt,
              "parent": parent_receipt, "data_manifests": data_manifests,
              "source_weights": weights, "mixture_receipt": mixture_receipt,
              "batch_policy": args.batch_policy, "language_exponent": args.language_exponent,
              "encoder_gradient_cache": {"chunk_size": args.encoder_chunk_size, "policy": "global-denominator-exact-vjp-v1"},
              "sts_development": sts_receipts,
              "independent_retrieval_development": retrieval_receipts,
              "production_quality_plan": admitted['production_quality_plan'],
              "development_policy": {"max_rows": args.dev_max_rows, "max_regression": args.max_dev_regression,
                  "eval_every": args.eval_every, "selection": "seeded-language-stratified-v1",
                  "metrics": ["mrr", "recall_at_1", "recall_at_10"],
                  "metric_policy": "deduplicated-candidate-fractional-recall-v2",
                  "recovery_policy": "authenticated-terminal-development-gate-v2",
                  "selection_policy": "equal-source macro MRR or Spearman; natural/programming-language regression gates",
                  "programming_language_policy": "authenticated-real-label-slices-v1; unknowns excluded; no extra macro votes",
                  "selection_split": "dev"}, "batch_counts": counts, "steps": args.steps,
              "global_batch_size": args.batch_size, "warmup": args.warmup,
              "learning_rate": args.learning_rate, "temperature": args.temperature,
              "weight_decay": args.weight_decay, "seed": args.seed,
              "optimizer": "AdamW, FP32 state, global-norm clip 1",
              "schedule": "warmup cosine to 10% peak",
              "objective": "symmetric global in-batch InfoNCE with mined negatives and duplicate masking",
              "shuffle": "cursor-derived language-temperature draws with replacement; source-local homogeneous cursors",
              "query_length": next(iter(query_lengths)),
              "document_length": next(iter(document_lengths))}
    if objective_identity is not None:
        recipe["contrastive_objective"] = objective_identity
    if encoder_config.get("weight_quantization", "none") != "none":
        recipe["quantization_aware_training"] = {
            "format": "symmetric-int8-output-channel-v1", "range": [-127, 127],
            "rounding": "nearest-even", "scale_dtype": "float32",
            "scale": "max(abs(master))/127; zero row scale 1; floor fp32 tiny",
            "gradient": "identity-ste-including-scale", "master_dtype": "float32",
            "linear_reduce_axis": 0, "embedding_reduce_axis": -1,
            "embedding_policy": "quantize-after-gather",
            "unquantized": ["norms", "biases", "yat_alpha", "unused-mlm-head"],
            "activation_quantization": False, "compressed_execution": False,
            "development_baseline": "quantized-parent; measures stage learning only",
            "additional_release_gate": "original-parent versus exported-QAT quality and physical loader parity"}
    regression_action = getattr(args, 'quality_regression_action', 'stop')
    if regression_action != 'stop':
        recipe['development_policy']['regression_action'] = regression_action
    schedule_steps = getattr(args, 'schedule_steps', None)
    schedule_steps = args.steps if schedule_steps is None else schedule_steps
    if schedule_steps != args.steps:
        recipe['schedule_steps'] = schedule_steps
    migration = None
    if getattr(args, 'resume_quality_migration', None) is not None:
        migration, lineage = admit_quality_migration(args.resume_quality_migration,
            {'resolved_config': recipe, 'tokenizer': parent_hashes['tokenizer.json'],
             'data_manifest': data_digest, 'source_python_sha256': source_hash}, args.output)
        recipe['resume_policy_migration'] = lineage
    metadata = {"model_family": "yat_embedding_finetune", "resolved_config": recipe,
                "tokenizer_identity": parent_hashes["tokenizer.json"],
                "data_manifest_identity": data_digest,
                "source_python_sha256": source_hash}
    expected = {"resolved_config": recipe, "tokenizer": parent_hashes["tokenizer.json"],
                "data_manifest": data_digest, "source_python_sha256": source_hash}
    if any(worker != expected for worker in gather_process_metadata(expected)):
        raise ValueError("Training hosts disagree on stage identity")

    model = load_public_encoder(parent)
    if (model.config.yat_bias != 1.0 or model.config.yat_epsilon != 0.01
            or not model.config.yat_alpha_trainable):
        raise ValueError("YAT architecture identity mismatch")
    if args.parent_checkpoint and not args.resume:
        configure_objective_state(model, parent_metadata["resolved_config"].get("contrastive_objective"))
        restore_model_from_checkpoint(model, args.parent_checkpoint, step=args.parent_step,
                                      expected_identity=parent_expected)
    # A new stage resets training-only alpha; exact resume restores it below.
    configure_objective_state(model, objective_identity)
    # Restore with the original parent config, then enter the new stage policy.
    # FP32 parameter leaves and trainable alpha retain their original identity.
    config = replace(model.config, weight_quantization=encoder_config.get("weight_quantization", "none"))
    model.config = config
    for layer in model.layers:
        layer.config = config
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    lr = optax.warmup_cosine_decay_schedule(0.0, args.learning_rate,
        args.warmup, schedule_steps, end_value=args.learning_rate * 0.1)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr, weight_decay=args.weight_decay,
            mask=lambda params: jax.tree.map(lambda value: value.ndim > 1, params))),
        wrt=nnx.Param)
    start = 0
    bootstrap_resume = False
    if args.resume:
        probe = create_checkpoint_manager(args.output, max_to_keep=999, async_checkpointing=False)
        try:
            latest = probe.latest_step()
        finally:
            probe.close()
        resume_source = args.output
        if latest is None:
            # Initial best step 0 can commit before the first recovery snapshot.
            # It is a complete authenticated model/optimizer/cursor snapshot.
            resume_source = args.output.rstrip("/") + "/best"
            restored_metadata = load_checkpoint_metadata(resume_source, 0, include_receipt=True)
            bootstrap_resume = True
        else:
            restored_metadata = load_checkpoint_metadata(resume_source, latest, include_receipt=True)
        restore_expected = migration_restore_identity(restored_metadata, expected, migration)
        _, state = restore_model_from_checkpoint(model, resume_source, step=restored_metadata["step"],
            optimizer=optimizer, expected_identity=restore_expected, load_training_state=True)
        start = validate_cursor(state, horizon=args.steps, stop=args.stop_after or args.steps,
                                committed_step=restored_metadata["step"])
        validate_optimizer_cursor(np.asarray(optimizer.step[...]), start)
        if migration is not None and jax.process_index() == 0:
            print(json.dumps({'event': 'embedding_resume_policy_migration', 'step': start,
                              'lineage': recipe['resume_policy_migration'],
                              'restored_previous_identity': restore_expected != expected}), flush=True)
    nnx.update(optimizer, replicate_on_mesh(nnx.state(optimizer), mesh))
    manager = create_checkpoint_manager(args.output,
        max_to_keep=args.keep_checkpoints, async_checkpointing=False)
    if not args.resume and manager.latest_step() is not None:
        manager.close()
        raise ValueError("Output already contains checkpoints")

    trace = TrainingTrace(args.profile_dir, skip=args.profile_skip, steps=args.profile_steps)

    @partial(nnx.jit, static_argnames=("pair_only",))
    def update(m, o, q, p, n, qid, pid, nid, gid, valid, known, *, pair_only=False):
        def objective(inner):
            pool = (lambda tokens: nnx_cached_pool(inner, tokens, args.encoder_chunk_size)) if args.encoder_chunk_size else inner.pool
            qvectors, pvectors = pool(q), pool(p)
            nvectors = jnp.zeros_like(pvectors) if pair_only else pool(n)
            return hard_negative_infonce(qvectors, pvectors, nvectors,
                qid, pid, nid, gid, valid, temperature=args.temperature,
                expected_global_batch=args.batch_size, known_positive_text_ids=known, return_metrics=True,
                **objective_loss_arguments(inner, objective_identity))
        (loss, masking_metrics), grads = nnx.value_and_grad(objective, has_aux=True)(m)
        accepted = apply_gradients_if_finite(m, o, grads, loss)
        return loss, accepted, optax.global_norm(grads), masking_metrics

    @nnx.jit
    def dev_loss(m, q, p, n, qid, pid, nid, gid, valid, known):
        return hard_negative_infonce(m.pool(q), m.pool(p), m.pool(n),
            qid, pid, nid, gid, valid, temperature=args.temperature,
            expected_global_batch=args.batch_size, known_positive_text_ids=known,
            **objective_loss_arguments(m, objective_identity))

    samplers = {name: ReplayRows(data[name]["train"]["language_ids"],
                                args.seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16),
                                exponent=args.language_exponent) for name in data}
    local = args.batch_size // jax.process_count()
    schedule = HomogeneousSchedule(sorted(data), weights, args.seed)
    pair_sources = {name: not bool(np.any(data[name]["train"]["negative_valid"])) for name in data}
    local_slice = slice(jax.process_index() * local, (jax.process_index() + 1) * local)

    def place(batch):
        return [place_host_batch(batch[name], mesh) for name in TRAIN_ARRAYS]

    @nnx.jit
    def encode(m, tokens):
        return m.pool(tokens)

    exposure = ExposureTracker(data, encoder_config["pad_token_id"])
    if args.resume:
        exposure.restore(state)

    baseline = None
    last_evaluation = None
    best_score, best_step, best_receipt = -math.inf, None, None
    best_output = args.output.rstrip("/") + "/best"
    best_manager = create_checkpoint_manager(best_output,
        max_to_keep=1, async_checkpointing=False)
    if args.resume:
        quality_state = state.get("quality_state")
        if quality_state is None:
            raise ValueError("Resume lacks development quality state; use an explicit new stage")
        quality_state = json.loads(bytes(np.asarray(quality_state, np.uint8)).decode())
        try:
            last_evaluation = validate_resume_evaluation(quality_state, cursor=start,
                horizon=args.steps, every=args.eval_every, max_regression=args.max_dev_regression,
                regression_action=regression_action)
        except Exception:
            best_manager.close()
            manager.close()
            raise
        best_metadata = load_checkpoint_metadata(best_output, include_receipt=True)
        best_expected = migration_restore_identity(best_metadata, expected, migration, best=True)
        quality_state = reconcile_best(quality_state, best_metadata, best_expected,
            horizon=args.steps, every=args.eval_every, bootstrap=bootstrap_resume)
        baseline, best_score = quality_state["baseline"], quality_state["best_score"]
        best_step, best_receipt = quality_state["best_step"], quality_state["best_receipt"]
        quality_gate(baseline, baseline, max_regression=args.max_dev_regression)
        if jax.process_index() == 0:
            print(json.dumps({"event": "embedding_selection_reconciled", "cursor": start,
                              "best_step": best_step, "best_score": best_score,
                              "best_ahead_of_cursor": best_step > start,
                              "bootstrap_resume": bootstrap_resume}), flush=True)
    elif best_manager.latest_step() is not None:
        best_manager.close()
        manager.close()
        raise ValueError("Best checkpoint output already exists; use authenticated --resume")

    def dev_embeddings(rows, indices):
        count = len(indices)
        embeddings = {key: [] for key in ("query_tokens", "positive_tokens")}
        for first in range(0, count, args.batch_size):
            take = min(args.batch_size, count - first)
            for key in embeddings:
                tokens = np.asarray(rows[key][indices[first:first + take]])
                if take < args.batch_size:
                    tokens = np.concatenate([tokens, np.repeat(tokens[-1:], args.batch_size - take, axis=0)])
                selected = slice(jax.process_index() * local, (jax.process_index() + 1) * local)
                encoded = encode(model, place_host_batch(tokens[selected], mesh))
                values = np.asarray(multihost_utils.process_allgather(encoded, tiled=True))
                embeddings[key].append(values[:take])
        return tuple(np.concatenate(embeddings[key]) for key in embeddings)

    def evaluate_dev_impl(step):
        nonlocal baseline, best_score, last_evaluation
        result = {}
        for name in sorted(data):
            rows, indices = data[name]["dev"], dev_indices[name]
            q, p = dev_embeddings(rows, indices)
            def probe(selected, q=q, p=p, rows=rows, indices=indices):
                return retrieval_metrics(q[selected], p[selected], rows["query_text_ids"][indices[selected]],
                    rows["positive_text_ids"][indices[selected]], rows["positive_group_ids"][indices[selected]])
            result[name] = probe(np.arange(len(indices)))
            result[name]["available_rows"] = len(rows["query_tokens"])
            result[name]["probe_policy"] = "seeded-language-stratified-v1"
            languages = rows["language_ids"][indices]
            for language in sorted(set(languages.tolist())):
                result[f"{name}/language/{language}"] = probe(np.flatnonzero(languages == language))
            for language, selected in programming_language_slices(rows['programming_languages'], indices).items():
                result[f'{name}/programming-language/{language}'] = probe(selected)
        for name, (rows, indices) in sorted(sts_data.items()):
            q, p = dev_embeddings(rows, indices)
            result[f"sts/{name}"] = sts_metrics(q, p, rows["scores"][indices])
            languages = rows["languages"][indices]
            for language in sorted(set(languages.tolist())):
                take = np.flatnonzero(languages == language)
                result[f"sts/{name}/language/{language}"] = sts_metrics(q[take], p[take], rows["scores"][indices[take]])
        for name, (rows, indices) in sorted(retrieval_data.items()):
            q, p = dev_embeddings(rows, indices)
            def independent_probe(selected, q=q, p=p, rows=rows, indices=indices):
                return retrieval_metrics(q[selected], p[selected], rows['query_text_ids'][indices[selected]],
                    rows['positive_text_ids'][indices[selected]], rows['positive_group_ids'][indices[selected]])
            key = f'independent/{name}'
            result[key] = independent_probe(np.arange(len(indices)))
            result[key]['available_rows'] = len(rows['query_tokens'])
            result[key]['probe_policy'] = 'seeded-language-stratified-v1'
            languages = rows['languages'][indices]
            for language in sorted(set(languages.tolist())):
                result[f'{key}/language/{language}'] = independent_probe(np.flatnonzero(languages == language))
            for language, selected in programming_language_slices(rows['programming_languages'], indices).items():
                result[f'{key}/programming-language/{language}'] = independent_probe(selected)
        if baseline is None:
            baseline = result
        gate = quality_gate(result, baseline, max_regression=args.max_dev_regression)
        last_evaluation = {"step": step, "metrics": result, "gate": gate}
        improved = gate["passed"] and gate["score"] > best_score
        if improved:
            if best_step is not None and step <= best_step:
                raise RuntimeError("Replay changed an earlier quality decision; preserve durable best and qualify migration")
            best_score = gate["score"]
        if jax.process_index() == 0:
            print(json.dumps({"event": "embedding_dev", "step": step,
                              "source_metrics": result, "gate": gate, "exposure": exposure.report(),
                              "regression_action": regression_action,
                              "best_score": best_score}), flush=True)
        return improved, gate

    def evaluate_dev(step):
        with telemetry.phase("evaluation"):
            return evaluate_dev_impl(step)

    def recovery_state(step):
        quality = json.dumps({"baseline": baseline, "best_score": best_score,
                             "last_evaluation": last_evaluation,
                             "best_step": best_step, "best_receipt": best_receipt},
                             sort_keys=True, allow_nan=False).encode()
        return {**exposure.state(), "completed_steps": np.array([step], np.int32),
                "optimizer_updates": np.array([int(optimizer.step[...])], np.int32),
                "quality_state": np.frombuffer(quality, dtype=np.uint8).copy()}

    def commit_recent_impl(step):
        saved = save_checkpoint(manager, step, model, optimizer, metadata,
                                training_state=recovery_state(step))
        require_save_success(saved, step)
        manager.wait_until_finished()
        committed = load_checkpoint_metadata(args.output, step, include_receipt=True)
        if committed["committed_receipt"]["step"] != step:
            raise RuntimeError("Recovery checkpoint did not commit requested cursor")

    def commit_recent(step):
        with telemetry.phase("recovery_checkpoint"):
            return commit_recent_impl(step)

    def commit_best_impl(step):
        nonlocal best_step, best_receipt
        best_step, best_receipt = step, None
        selected_metadata = {**metadata, "quality_selection": selection_record(step, best_score, baseline)}
        saved = save_checkpoint(best_manager, step, model, optimizer, selected_metadata,
                                training_state=recovery_state(step))
        require_save_success(saved, step)
        best_manager.wait_until_finished()
        committed = load_checkpoint_metadata(best_output, step, include_receipt=True)
        selected = reconcile_best(None, committed, expected, horizon=args.steps, every=args.eval_every)
        best_receipt = selected["best_receipt"]

    def commit_best(step):
        with telemetry.phase("best_checkpoint"):
            return commit_best_impl(step)

    # Bound count transport before any update, identically on every host.
    max_tokens_per_row = max(sum(rows["train"][key].shape[1] for key in
                                ("query_tokens", "positive_tokens", "negative_tokens"))
                             for rows in data.values())
    compiled_variants = set()
    telemetry.seconds["setup"] = time.monotonic() - telemetry.started
    if jax.process_index() == 0:
        print(json.dumps({"event": "embedding_run", "recipe": recipe,
                          "devices": jax.device_count(), "processes": jax.process_count(),
                          "start": start, "backend": jax.default_backend()}), flush=True)
    stop = args.stop_after or args.steps
    try:
        if local * max_tokens_per_row >= 2**31:
            raise ValueError("Per-host token accounting exceeds int32 transport capacity")
        if jax.process_index() == 0:
            emit_completion_status(start, stop)
        if start == 0 and not args.resume:
            improved, _ = evaluate_dev(0)
            if improved:
                commit_best(0)
                commit_recent(0)
        elif bootstrap_resume:
            commit_recent(0)
        for step in range(start, stop):
            begun = time.monotonic()
            source_cursor = None
            step_counts = counts
            if args.batch_policy == "homogeneous":
                source, source_cursor = schedule.at(step)
                step_counts = {source: args.batch_size}
            batch = _batch(step, "train", data, step_counts, args.seed, offsets, samplers,
                           local_slice=local_slice, source_cursor=source_cursor)
            input_seconds = time.monotonic() - begun
            # Static variant eliminates the third encoder for pair-only batches.
            pair_only = all(pair_sources[name] for name in step_counts)
            first_variant_call = pair_only not in compiled_variants
            update_started = time.monotonic()
            with trace.step(step - start):
                loss, accepted, grad_norm, masking_metrics = update(model, optimizer, *place(batch), pair_only=pair_only)
                jax.block_until_ready(loss)
            update_seconds = time.monotonic() - update_started
            compiled_variants.add(pair_only)
            if not bool(accepted) or not math.isfinite(float(loss)):
                raise FloatingPointError(f"Rejected update {step + 1}")
            for name, count in step_counts.items():
                indices = samplers[name].batch(step if source_cursor is None else source_cursor, count)
                exposure.record(name, indices)
            # Fixed cadence on every host: never put a collective under host-zero logging.
            positive_tokens = sum(np.count_nonzero(batch[key] != encoder_config["pad_token_id"])
                                  for key in ("query_tokens", "positive_tokens"))
            processed_tokens = useful_tokens = positive_tokens
            if not pair_only:
                negative_counts = np.count_nonzero(batch["negative_tokens"] != encoder_config["pad_token_id"], axis=1)
                processed_tokens += int(negative_counts.sum())
                useful_tokens += int(negative_counts[batch["negative_valid"]].sum())
            gathered = multihost_utils.process_allgather({
                "counts": np.asarray([local, processed_tokens, useful_tokens], np.int32),
                "timing": np.asarray([input_seconds, update_seconds, time.monotonic() - begun,
                                      int(first_variant_call)], np.float32)}, tiled=False)
            # Keep counts integer through transport even when JAX x64 is disabled.
            host_rows = [list(counts_row) + list(timing_row) for counts_row, timing_row in zip(
                np.asarray(gathered["counts"]).reshape(-1, 3),
                np.asarray(gathered["timing"]).reshape(-1, 4), strict=True)]
            step_telemetry = reduce_step(host_rows)
            telemetry.record_step(step_telemetry)
            if jax.process_index() == 0:
                print(json.dumps({"event": "embedding_step", "step": step + 1,
                    "loss": float(loss), "seconds": time.monotonic() - begun,
                    "global_pairs": args.batch_size, "input_seconds": input_seconds,
                    "telemetry": step_telemetry,
                    "gradient_norm": float(grad_norm), "pair_only": pair_only,
                    "candidate_masking": {key: int(value) for key, value in masking_metrics.items() if not key.startswith("yat_")},
                    **({"contrastive_metrics": {key: float(value) for key, value in masking_metrics.items()
                        if key.startswith("yat_")}} if objective_identity is not None else {}),
                    "nonpadding_tokens_local": int(processed_tokens)}), flush=True)
            if evaluation_due(step + 1, args.eval_every, args.steps):
                improved, gate = evaluate_dev(step + 1)
                if improved:
                    commit_best(step + 1)
                if not gate["passed"] and regression_action == 'stop':
                    commit_recent(step + 1)
                    raise RuntimeError(f"Development representation gate failed: {gate['regressions']}")
            if (step + 1) % args.save_every == 0 or step + 1 == stop:
                checkpoint_started = time.monotonic()
                commit_recent(step + 1)
                if jax.process_index() == 0:
                    print(json.dumps({"event": "embedding_checkpoint",
                                      "step": step + 1, "seconds": time.monotonic() - checkpoint_started}), flush=True)
    finally:
        with telemetry.phase("finalization"):
            trace.close()
            best_manager.close()
            manager.close()
    # Only successful completion reaches this collective. No failure-path barrier.
    timing_rows = np.asarray(multihost_utils.process_allgather(
        np.asarray(telemetry.timing_vector(), np.float32), tiled=False)).reshape(-1, len(PHASES) + 1)
    if jax.process_index() == 0:
        print(json.dumps({"event": "embedding_stage_telemetry", "start": start, "stop": stop,
                          **telemetry.report(timing_rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
