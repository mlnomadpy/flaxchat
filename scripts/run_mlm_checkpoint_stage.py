"""Bounded new-data MLM stage from a pinned trained checkpoint; TPU models only.

The supervisor never imports JAX. Data authentication/selection uses NumPy;
every forward/backward/restore executes in an exclusive physical-TPU child.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def authenticated_parent(args):
    from flaxchat.checkpoint_metadata import read_committed_metadata

    metadata = read_committed_metadata(
        args.parent_checkpoint, args.parent_step, include_receipt=True
    )
    receipt = metadata.pop("committed_receipt")
    if (
        receipt["metadata_sha256"] != args.parent_metadata_sha256
        or receipt["manifest_sha256"] != args.parent_manifest_sha256
        or metadata.get("model_family") != "modernbert"
    ):
        raise ValueError("Pinned committed MLM parent identity differs")
    encoder = metadata["resolved_config"]["encoder"]
    if (
        encoder.get("attention_score") != "goat_input"
        or encoder.get("ffn_type") != "yat_glu"
        or encoder.get("yat_bias") != 1
        or encoder.get("yat_epsilon") != 0.01
        or encoder.get("yat_alpha_trainable") is not True
    ):
        raise ValueError("Expected existing trained GOAT-input/YAT architecture")
    return metadata, receipt


def row_digest(row, pad):
    import numpy as np

    return hashlib.sha256(np.asarray(row[row != pad], dtype="<i4").tobytes()).digest()


def validate_device_inventory(backend, kinds, processes):
    if (
        backend != "tpu"
        or len(kinds) != 8
        or processes != 1
        or any(
            "v5" not in kind.lower()
            or not ("lite" in kind.lower() or "v5e" in kind.lower())
            for kind in kinds
        )
    ):
        raise RuntimeError(
            "This stage qualification requires one physical eight-device v5e host"
        )


def balanced_selection(train, validation, documents, *, pad, seed, count=512):
    """Keep only validation hashes in RAM; scan training rows exactly once."""
    import numpy as np

    if len(validation) > 2_000_000 or count < 1:
        raise ValueError("Validation selection exceeds bounded inventory")
    hashes = [row_digest(row, pad) for row in validation]
    candidates = set(hashes)
    exposed = set()
    for row in train:
        digest = row_digest(row, pad)
        if digest in candidates:
            exposed.add(digest)
    groups, cursor, seen = {}, 0, set()
    with Path(documents).open() as stream:
        for line in stream:
            document = json.loads(line)
            start, end, label = (
                document["first_row"],
                document["end_row"],
                document["language_script"],
            )
            if (
                type(start) is not int
                or type(end) is not int
                or start != cursor
                or not start < end <= len(validation)
                or not isinstance(label, str)
                or not label
            ):
                raise ValueError("Invalid contiguous validation document provenance")
            for index in range(start, end):
                digest = hashes[index]
                if digest not in exposed and digest not in seen:
                    groups.setdefault(label, []).append(index)
                    seen.add(digest)
            cursor = end
    if cursor != len(validation):
        raise ValueError("Incomplete validation document provenance")
    rng = np.random.default_rng(seed)
    pools = {
        label: rng.permutation(indices).tolist()
        for label, indices in sorted(groups.items())
    }
    chosen, labels = [], []
    for ordinal in range(max(map(len, pools.values()), default=0)):
        for label, indices in pools.items():
            if ordinal < len(indices):
                chosen.append(indices[ordinal])
                labels.append(label)
                if len(chosen) == count:
                    return np.asarray(chosen, dtype="<i8"), labels, len(exposed)
    raise ValueError("Insufficient independent exact-token validation rows")


def admit_data(args, parent):
    """Full actual array/provenance hashes once; repeated eval uses sealed masks."""
    import numpy as np
    from flaxchat.encoder_data import load_prepared_rows
    from flaxchat.mlm import mask_tokens

    corpus = args.root / "corpus"
    complete = read(corpus / "COMPLETE.json")
    for field, relative in [
        ("recipe_sha256", "recipe.json"),
        ("train_manifest_sha256", "train/manifest.json"),
        ("validation_manifest_sha256", "validation/manifest.json"),
    ]:
        if complete.get(field) != sha(corpus / relative):
            raise ValueError("Completed corpus identity differs: " + relative)
    if complete.get("training_qualified") is not True:
        raise ValueError("Completed corpus coverage is not qualified")
    encoder, policy = parent["resolved_config"]["encoder"], parent["resolved_config"]
    arrays, manifests = [], []
    for split in ("train", "validation"):
        directory = corpus / split
        tokens, manifest = load_prepared_rows(
            directory,
            SimpleNamespace(**encoder),
            minimum_rows=64 if split == "train" else 512,
        )
        if (
            manifest.get("split") != split
            or manifest["tokenizer_sha256"] != parent["tokenizer_identity"]
            or manifest["special_token_ids"] != policy["special_token_ids"]
            or tokens.shape[1] != 512
            or sha(directory / "documents.jsonl") != manifest.get("documents_sha256")
        ):
            raise ValueError("Actual corpus schema/tokenizer/provenance differs")
        arrays.append(tokens)
        manifests.append(manifest)
    selected, languages, overlap = balanced_selection(
        arrays[0],
        arrays[1],
        corpus / "validation/documents.jsonl",
        pad=encoder["pad_token_id"],
        seed=2026,
    )
    x, y = mask_tokens(
        arrays[1][selected],
        seed=2026,
        step=0,
        example_ids=selected,
        vocab_size=encoder["vocab_size"],
        mask_token_id=encoder["mask_token_id"],
        special_token_ids=policy["special_token_ids"],
        probability=policy["mask_probability"],
    )
    if not np.count_nonzero(y >= 0):
        raise ValueError("Validation has no masked targets")
    cache = args.root / "new-stage-evaluation"
    cache.mkdir(exist_ok=True)
    for name, value in [("inputs", x), ("targets", y), ("indices", selected)]:
        np.save(cache / (name + ".npy"), value, allow_pickle=False)
    selection = {
        "protocol": "new-corpus-exact-token-balanced-fixed-masks-v1",
        "train_manifest_sha256": complete["train_manifest_sha256"],
        "validation_manifest_sha256": complete["validation_manifest_sha256"],
        "corpus_complete_sha256": sha(corpus / "COMPLETE.json"),
        "recipe_sha256": complete["recipe_sha256"],
        "train_tokens_sha256": manifests[0]["tokens_sha256"],
        "validation_tokens_sha256": manifests[1]["tokens_sha256"],
        "parent_tokenizer_sha256": parent["tokenizer_identity"],
        "files": {
            name + ".npy": sha(cache / (name + ".npy"))
            for name in ("inputs", "targets", "indices")
        },
        "selected_rows_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "rows": len(selected),
        "masked_tokens": int(np.count_nonzero(y >= 0)),
        "seed": 2026,
        "mask_probability": policy["mask_probability"],
        "special_token_ids": policy["special_token_ids"],
        "language_rows": {
            label: languages.count(label) for label in sorted(set(languages))
        },
        "excluded_distinct_validation_token_rows": overlap,
        "exposure_scope": "new-training exact unpadded token rows; historical/semantic exposure unresolved",
    }
    write(cache / "selection.json", selection)
    return selection


def training_argv(args, output, stop, *, resume=False, save_every=250):
    encoder = read(args.root / "stage-encoder.json")
    flags = [
        "--config",
        str(args.root / "stage-encoder.json"),
        "--data",
        str(args.root / "corpus/train"),
        "--output",
        output,
        "--steps",
        str(args.steps),
        "--stop-after",
        str(stop),
        "--batch-size",
        "64",
        "--accumulation-steps",
        "4",
        "--local-gradient-accumulation",
        "--learning-rate",
        "0.0001",
        "--lr-schedule",
        "cosine",
        "--warmup-steps",
        "500",
        "--final-lr-ratio",
        ".05",
        "--seed",
        "1006",
        "--mask-probability",
        ".15",
        "--language-exponent",
        ".5",
        "--coverage-sampling",
        "--save-every",
        str(save_every),
        "--keep-checkpoints",
        "3",
        "--compile-diagnostics",
    ]
    mapping = {
        "compute_dtype": "dtype",
        "residual_dtype": "residual-dtype",
        "attention_backend": "attention-backend",
    }
    for name in (
        "loss_chunk_size",
        "mlm_projection",
        "mlm_projection_capacity",
        "mlm_loss_backend",
        "mlm_vocab_tile",
        "ffn_type",
        "yat_epsilon",
        "yat_alpha",
        "yat_attention_alpha",
        "attention_score",
        "yat_compute_mode",
        "yat_ffn_compute_mode",
        "yat_ffn_backward_block",
        "yat_attention_block_size",
        "yat_global_attention_block_size",
        "yat_softmax_backward",
        "yat_attention_implementation",
    ):
        mapping[name] = name.replace("_", "-")
    for name, flag in mapping.items():
        if encoder.get(name) is not None:
            flags += ["--" + flag, str(encoder[name])]
    if encoder.get("yat_local_shards"):
        flags += ["--yat-local-shards"]
    if encoder.get("use_remat") is False:
        flags += ["--no-remat"]
    if resume:
        flags += ["--resume"]
    else:
        flags += [
            "--initialize-from-checkpoint",
            args.parent_checkpoint,
            "--initialize-step",
            str(args.parent_step),
        ]
    return flags


def physical_evaluate(args, checkpoint, step):
    import jax
    import numpy as np
    from flax import nnx
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.checkpoint import restore_model_from_checkpoint
    from flaxchat.checkpoint_metadata import read_committed_metadata
    from flaxchat.common import replicate_on_mesh
    from flaxchat.training import place_host_batch
    from flaxchat.runtime import runtime_identity

    validate_device_inventory(
        jax.default_backend(),
        [d.device_kind for d in jax.devices()],
        jax.process_count(),
    )
    metadata = read_committed_metadata(checkpoint, step, include_receipt=True)
    committed = metadata.pop("committed_receipt")
    if checkpoint == args.parent_checkpoint and (
        committed["manifest_sha256"] != args.parent_manifest_sha256
        or committed["metadata_sha256"] != args.parent_metadata_sha256
        or step != args.parent_step
    ):
        raise ValueError("Physical baseline parent commit differs from admitted pins")
    encoder = read(args.root / "stage-encoder.json")
    if metadata["resolved_config"]["encoder"] != encoder:
        # Current EncoderConfig may add default, nonarchitectural dataclass fields.
        if EncoderConfig(**metadata["resolved_config"]["encoder"]) != EncoderConfig(
            **encoder
        ):
            raise ValueError(
                "Checkpoint architecture differs from pinned trained parent"
            )
    selection = authenticate_selection(args)
    cache = args.root / "new-stage-evaluation"
    for relative, digest in selection["files"].items():
        if sha(cache / relative) != digest:
            raise ValueError("Fixed evaluation masks/rows changed")
    inputs, targets = (
        np.load(cache / (name + ".npy"), allow_pickle=False)
        for name in ("inputs", "targets")
    )
    if inputs.shape != (512, 512) or targets.shape != inputs.shape:
        raise ValueError("Fixed validation requires512rows x512tokens")
    model = ModernBert(EncoderConfig(**encoder), rngs=nnx.Rngs(0))
    restore_model_from_checkpoint(
        model,
        checkpoint,
        step=step,
        expected_identity={
            "resolved_config": metadata["resolved_config"],
            "tokenizer": metadata["tokenizer_identity"],
            "data_manifest": metadata["data_manifest_identity"],
            "source_python_sha256": metadata["source_python_sha256"],
        },
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    loss_fn = nnx.jit(lambda m, x, y: m(x, y))
    numerator, denominator = 0.0, 0
    for first in range(0, 512, 16):
        count = int(np.count_nonzero(targets[first : first + 16] >= 0))
        loss = float(
            loss_fn(
                model,
                place_host_batch(inputs[first : first + 16], mesh),
                place_host_batch(targets[first : first + 16], mesh),
            )
        )
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite fixed-mask heldout MLM loss")
        numerator += loss * count
        denominator += count
    from scripts.representation_run import source_snapshot, source_tree_digest

    return {
        "backend": "tpu",
        "devices": 8,
        "processes": 1,
        "device_kinds": [d.device_kind for d in jax.devices()],
        "checkpoint": checkpoint,
        "checkpoint_step": step,
        "masked_token_loss": numerator / denominator,
        "masked_tokens": denominator,
        "evaluation_batch_size": 16,
        "evaluated_rows": 512,
        "selection_sha256": sha(cache / "selection.json"),
        "selected_rows_sha256": selection["selected_rows_sha256"],
        "runtime": runtime_identity(),
        "source_tree_sha256": source_tree_digest(
            source_snapshot(Path(__file__).parents[1])
        ),
        "protocol": selection["protocol"],
    }


def validate_report(report, baseline):
    if any(
        report.get(key) != baseline.get(key)
        for key in (
            "backend",
            "devices",
            "processes",
            "masked_tokens",
            "evaluation_batch_size",
            "evaluated_rows",
            "selection_sha256",
            "selected_rows_sha256",
            "protocol",
        )
    ):
        raise ValueError("Heldout evaluation identities differ")
    if (
        report.get("backend") != "tpu"
        or report.get("devices") != 8
        or report.get("evaluated_rows") != 512
    ):
        raise ValueError("Physical heldout coverage differs")
    losses = [report.get("masked_token_loss"), baseline.get("masked_token_loss")]
    if (
        any(
            type(value) not in (float, int) or not math.isfinite(value) or value < 0
            for value in losses
        )
        or losses[0] > 1.05 * losses[1]
    ):
        raise ValueError("Heldout loss exceeds matched parent baseline by5%")


def validate_updates(log, start, stop):
    events = []
    for line in Path(log).read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("event") == "train_step":
            events.append(item)
    if [item.get("step") for item in events] != list(range(start + 1, stop + 1)) or any(
        item.get("updated") is not True
        or item.get("masked_tokens", 0) < 1
        or not math.isfinite(item.get("loss", float("nan")))
        or not math.isfinite(item.get("gradient_norm_before_clip", float("nan")))
        for item in events
    ):
        raise ValueError("Finite applied optimizer update sequence differs")
    return events


def execute(
    command, log, deadline, *, graceful_seconds=120, environment=None, progress=None
):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Stage process deadline exhausted")
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Stage child deadline exhausted; retain only verified commits"
                    )
                try:
                    code = process.wait(timeout=min(60, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if progress is not None:
                        progress()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=graceful_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)


def child_args(args, action, *, checkpoint=None, step=None):
    values = [
        sys.executable,
        "-m",
        "scripts.run_mlm_checkpoint_stage",
        "--root",
        str(args.root),
        "--output",
        checkpoint or args.output,
        "--parent-checkpoint",
        args.parent_checkpoint,
        "--parent-step",
        str(args.parent_step),
        "--parent-metadata-sha256",
        args.parent_metadata_sha256,
        "--parent-manifest-sha256",
        args.parent_manifest_sha256,
        "--steps",
        str(args.steps),
        "--max-seconds",
        str(args.max_seconds),
        "--evidence-prefix",
        args.evidence_prefix,
        "--action",
        action,
    ]
    if step is not None:
        values += ["--checkpoint-step", str(step)]
    return values


def upload(args, paths):
    for path in paths:
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                str(path),
                args.evidence_prefix.rstrip("/") + "/" + Path(path).name,
            ],
            check=True,
            timeout=90,
        )


def authenticate_selection(args):
    selection = read(args.root / "new-stage-evaluation/selection.json")
    for relative, digest in selection["files"].items():
        if sha(args.root / "new-stage-evaluation" / relative) != digest:
            raise ValueError("Cached heldout token/mask identity differs")
    if (
        sha(args.root / "corpus/COMPLETE.json") != selection["corpus_complete_sha256"]
        or sha(args.root / "corpus/recipe.json") != selection["recipe_sha256"]
        or sha(args.root / "corpus/train/manifest.json")
        != selection["train_manifest_sha256"]
        or sha(args.root / "corpus/validation/manifest.json")
        != selection["validation_manifest_sha256"]
    ):
        raise ValueError("Cached heldout corpus identity differs")
    return selection


def latest_committed(output):
    """A listing discovers candidates; committed reader authenticates selection."""
    if output.startswith("gs://"):
        process = subprocess.run(
            ["gcloud", "storage", "ls", output.rstrip("/") + "/"],
            text=True,
            capture_output=True,
            timeout=60,
        )
        if process.returncode:
            if "matched no objects" in process.stderr.lower():
                return None
            raise RuntimeError("Cannot observe owned stage checkpoint inventory")
        names = [
            line.rstrip("/").rsplit("/", 1)[-1] for line in process.stdout.splitlines()
        ]
    else:
        directory = Path(output)
        names = (
            [path.name for path in directory.iterdir()] if directory.exists() else []
        )
    return max((int(name) for name in names if name.isdecimal()), default=None)


def segment_boundary(
    step,
    stop,
    remaining,
    previous_seconds=None,
    previous_updates=None,
    *,
    interval=5000,
):
    """Warm first250, then conservatively fit checkpoint multiples in lease."""
    size = min(250 if previous_seconds is None else interval, stop - step)
    if previous_seconds is None:
        return step + size if remaining > 1800 else None
    per_update = previous_seconds / previous_updates
    affordable = int(max(0, remaining - 600) / (1.2 * per_update))
    size = min(size, affordable // 250 * 250)
    if size < 250 and stop - step > size:
        return None
    return step + size if size else None


def physical_test_environment():
    return os.environ | {"FLAXCHAT_PHYSICAL_TPU": "1"}


def newstage_metadata(output, step, args):
    from flaxchat.checkpoint_metadata import read_committed_metadata

    metadata = read_committed_metadata(output, step, include_receipt=True)
    initialization = metadata.get("initialization", {})
    if (
        initialization.get("checkpoint") != args.parent_checkpoint
        or initialization.get("step") != args.parent_step
    ):
        raise ValueError(
            "New-stage checkpoint lost trained-parent initialization lineage"
        )
    admitted_parent = read(args.root / "parent-admitted.json")
    if (
        initialization.get("metadata_sha256")
        != hashlib.sha256(
            json.dumps(admitted_parent, sort_keys=True).encode()
        ).hexdigest()
    ):
        raise ValueError("New-stage initialization metadata pin differs")
    admission = read(args.root / "new-stage-evaluation/selection.json")
    if (
        metadata["resolved_config"].get("steps") != args.steps
        or metadata["data_manifest_identity"] != admission["train_manifest_sha256"]
    ):
        raise ValueError("New-stage horizon/corpus identity differs")
    return metadata


def run(args):
    deadline = time.monotonic() + args.max_seconds
    parent, parent_receipt = authenticated_parent(args)
    args.root.mkdir(parents=True, exist_ok=True)
    write(args.root / "stage-encoder.json", parent["resolved_config"]["encoder"])
    write(args.root / "parent-admitted.json", parent)
    spec = {
        "parent": {
            "checkpoint": args.parent_checkpoint,
            "step": args.parent_step,
            **{k: v for k, v in parent_receipt.items() if k != "model_state"},
        },
        "steps": args.steps,
        "output": args.output,
        "policy": "model_weights_only; fresh_optimizer_schedule_and_data_cursor",
        "training_argv": training_argv(args, args.output, args.steps),
    }
    write(args.root / "stage-spec.json", spec)
    if args.qualify_only:
        admit_data(args, parent)
    else:
        authenticate_selection(args)

    def evaluation(checkpoint, step):
        execute(
            child_args(args, "evaluate", checkpoint=checkpoint, step=step),
            args.root / f"evaluate-{step}.log",
            deadline,
        )

    if args.qualify_only:
        qualification_output = args.output.rstrip("/") + "-qualification"
        xml = args.root / "goat-physical-tests.xml"
        execute(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_goat_physical_tpu.py",
                "--junitxml=" + str(xml),
            ],
            args.root / "goat-physical-tests.log",
            deadline,
            environment=physical_test_environment(),
        )
        from scripts.run_yat_mlm_continuation import validate_goat_test_report

        validate_goat_test_report(xml, input_scores=True)
        evaluation(args.parent_checkpoint, args.parent_step)
        baseline = read(args.root / f"evaluate-{args.parent_step}.json")
        execute(
            [
                sys.executable,
                "-m",
                "scripts.train_encoder",
                *training_argv(args, qualification_output, 4, save_every=4),
            ],
            args.root / "train-4.log",
            deadline,
        )
        validate_updates(args.root / "train-4.log", 0, 4)
        fourth = newstage_metadata(qualification_output, 4, args)
        evaluation(qualification_output, 4)
        validate_report(read(args.root / "evaluate-4.json"), baseline)
        execute(
            [
                sys.executable,
                "-m",
                "scripts.train_encoder",
                *training_argv(
                    args, qualification_output, 8, resume=True, save_every=4
                ),
            ],
            args.root / "resume-8.log",
            deadline,
        )
        validate_updates(args.root / "resume-8.log", 4, 8)
        eighth = newstage_metadata(qualification_output, 8, args)
        if (
            fourth["resolved_config"] != eighth["resolved_config"]
            or fourth["source_python_sha256"] != eighth["source_python_sha256"]
        ):
            raise ValueError("Qualification exact-resume identity changed")
        evaluation(qualification_output, 8)
        validate_report(read(args.root / "evaluate-8.json"), baseline)
        run_manifest = read(args.root / "run.json")
        tests = [
            "goat-physical-forward-backward",
            "trained-parent-fixed-new-validation",
            "new-stage-four-updates",
            "exact-new-stage-resume-eight-updates",
            "matched-heldout-parent-regression",
        ]
        receipt = {
            "schema_version": 1,
            "passed": True,
            "backend": "tpu",
            "devices": 8,
            "processes": 1,
            "manifest_sha256": sha(args.root / "run.json"),
            "runtime_lock_sha256": run_manifest["runtime_lock"]["sha256"],
            "source_tree_sha256": os.environ["FLAXCHAT_SOURCE_TREE_SHA256"],
            "tests": [{"id": name, "status": "passed"} for name in tests],
            "stage_spec_sha256": sha(args.root / "stage-spec.json"),
            "selection_sha256": sha(args.root / "new-stage-evaluation/selection.json"),
            "parent": spec["parent"],
            "parent_baseline": baseline,
            "goat_physical_xml_sha256": sha(xml),
            "qualification_checkpoint": qualification_output,
            "last_committed_step": 8,
            "resume_scope": "authenticated model/optimizer/cursor restored, applied updates5..8; no continuous-run equivalence claim",
        }
        write(args.root / "mlm-qualification.json", receipt)
        upload(
            args,
            [
                args.root / "mlm-qualification.json",
                args.root / "stage-spec.json",
                xml,
                *args.root.glob("evaluate-*.json"),
                args.root / "train-4.log",
                args.root / "resume-8.log",
            ],
        )
        return receipt
    receipt = read(args.root / "mlm-qualification.json")
    if (
        not receipt.get("passed")
        or receipt.get("stage_spec_sha256") != sha(args.root / "stage-spec.json")
        or receipt.get("selection_sha256")
        != sha(args.root / "new-stage-evaluation/selection.json")
    ):
        raise ValueError(
            "Production requires current qualified stage/selection identity"
        )
    baseline = receipt["parent_baseline"]
    observed = latest_committed(args.output)
    step, stop = observed or 0, args.stop_after or args.steps
    if observed is not None:
        newstage_metadata(args.output, observed, args)
    if step > stop:
        raise ValueError("Requested segment stop precedes existing new-stage cursor")
    previous_seconds, previous_updates = None, None
    # Finite segments stop at real checkpoint boundaries; SIGTERM alone does not
    # guarantee an off-cadence trainer save. The last observed commit is durable.
    while step < stop:
        boundary = segment_boundary(
            step,
            stop,
            deadline - time.monotonic(),
            previous_seconds,
            previous_updates,
            interval=args.eval_every,
        )
        if boundary is None:
            break
        segment_started = time.monotonic()
        log = args.root / f"train-to-{boundary}.log"
        state = {
            "status": "training_segment_running",
            "checkpoint": args.output,
            "last_observed_committed_step": step,
            "target_boundary": boundary,
            "parent": spec["parent"],
            "full_corpus_exposure_claimed": False,
        }
        write(args.root / "stage-status.json", state)
        upload(args, [args.root / "stage-status.json"])

        def progress(state=state, log=log):
            write(
                args.root / "stage-status.json",
                {
                    **state,
                    "observed_unix": time.time(),
                    "worker_log_bytes": log.stat().st_size,
                },
            )
            try:
                upload(args, [args.root / "stage-status.json", log])
            except subprocess.SubprocessError:
                # Failure to upload progress does not imply the owned worker died.
                write(
                    args.root / "progress-upload-error.json",
                    {"failed_unix": time.time()},
                )

        execute(
            [
                sys.executable,
                "-m",
                "scripts.train_encoder",
                *training_argv(args, args.output, boundary, resume=step > 0),
            ],
            log,
            deadline - 300,
            progress=progress,
        )
        validate_updates(log, step, boundary)
        metadata = newstage_metadata(args.output, boundary, args)
        evaluation(args.output, boundary)
        report = read(args.root / f"evaluate-{boundary}.json")
        validate_report(report, baseline)
        previous_seconds, previous_updates = (
            time.monotonic() - segment_started,
            boundary - step,
        )
        step = boundary
        state = {
            "status": "segment_committed",
            "checkpoint": args.output,
            "step": step,
            "committed_receipt": metadata["committed_receipt"],
            "matched_validation": report,
            "prepared_train_nonpadding_tokens": read(
                args.root / "corpus/train/manifest.json"
            )["nonpadding_tokens"],
            "full_corpus_exposure_claimed": False,
        }
        write(args.root / "stage-status.json", state)
        upload(
            args,
            [
                args.root / "stage-status.json",
                log,
                args.root / f"evaluate-{boundary}.json",
            ],
        )
    write(
        args.root / "stage-status.json",
        {
            "status": "completed" if step == stop else "bounded_segment_stopped",
            "checkpoint": args.output,
            "last_observed_committed_step": step,
            "requested_stop": stop,
            "full_corpus_exposure_claimed": False,
        },
    )
    upload(args, [args.root / "stage-status.json"])


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--parent-checkpoint", required=True)
    value.add_argument("--parent-step", type=int, required=True)
    value.add_argument("--parent-metadata-sha256", required=True)
    value.add_argument("--parent-manifest-sha256", required=True)
    value.add_argument("--steps", type=int, default=500000)
    value.add_argument("--stop-after", type=int)
    value.add_argument("--max-seconds", type=int, required=True)
    value.add_argument("--qualify-only", action="store_true")
    value.add_argument("--eval-every", type=int, default=5000)
    value.add_argument("--evidence-prefix", required=True)
    value.add_argument(
        "--action",
        choices=["supervise", "evaluate"],
        default="supervise",
        help=argparse.SUPPRESS,
    )
    value.add_argument("--checkpoint-step", type=int, help=argparse.SUPPRESS)
    return value


def main():
    args = parser().parse_args()
    if (
        args.steps <= 500
        or args.max_seconds < 1
        or args.parent_step < 1
        or args.eval_every < 250
        or args.eval_every % 250
        or args.action == "supervise"
        and args.output.rstrip("/") == args.parent_checkpoint.rstrip("/")
        or args.stop_after is not None
        and not 0 < args.stop_after <= args.steps
    ):
        raise ValueError(
            "Require finite bounded new stage distinct from trained parent"
        )

    def interrupted(_signum, _frame):
        raise InterruptedError(
            "Owned MLM stage interrupted; retain verified commits only"
        )

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.action == "evaluate":
        write(
            args.root / f"evaluate-{args.checkpoint_step}.json",
            physical_evaluate(args, args.output, args.checkpoint_step),
        )
    else:
        try:
            run(args)
        except BaseException as error:
            committed = None
            try:
                candidate = latest_committed(args.output)
                if candidate is not None:
                    newstage_metadata(args.output, candidate, args)
                    committed = candidate
            except Exception:
                pass
            write(
                args.root / "stage-status.json",
                {
                    "status": "qualification_failed"
                    if args.qualify_only
                    else "interrupted_or_failed",
                    "error_type": type(error).__name__,
                    "last_observed_committed_step": committed,
                    "off_cadence_save_guaranteed": False,
                    "full_corpus_exposure_claimed": False,
                },
            )
            try:
                upload(args, [args.root / "stage-status.json"])
            except Exception:
                pass
            raise


if __name__ == "__main__":
    main()
