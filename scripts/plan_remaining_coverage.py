"""Plan a pinned, full-state sampler transition without importing a model backend.

Run near the prepared corpus, after selecting the final committed cutover step.
This only writes a reviewable plan; it never provisions hardware or trains.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from flaxchat.checkpoint_metadata import read_committed_metadata
from flaxchat.coverage_transition import admit_coverage_transition
from flaxchat.encoder_data import (
    RemainingCoverageMixtureRows,
    file_hash,
    load_prepared_rows,
)


def transition_commands(
    original, *, data, config, output, parent, step, completion, pins
):
    """Keep original training flags; replace only stage location and sampler admission."""
    argv = list(original)
    for flag in ("--initialize-from-checkpoint", "--initialize-step", "--stop-after"):
        while flag in argv:
            at = argv.index(flag)
            del argv[at : at + 2]
    if any(
        flag in argv
        for flag in (
            "--resume",
            "--pretrained",
            "--initialize-from-public",
            "--migrate-coverage-from-checkpoint",
            "--remaining-coverage-from-step",
        )
    ):
        raise ValueError("Expected the original legacy stage initialization arguments")
    for flag, value in (("--data", data), ("--config", config), ("--output", output)):
        if argv.count(flag) != 1:
            raise ValueError("Original stage arguments require exactly one " + flag)
        argv[argv.index(flag) + 1] = str(value)
    common = argv + [
        "--remaining-coverage-from-step",
        str(step),
        "--stop-after",
        str(completion),
    ]
    migration = common + [
        "--migrate-coverage-from-checkpoint",
        parent,
        "--coverage-parent-manifest-sha256",
        pins["manifest_sha256"],
        "--coverage-parent-metadata-sha256",
        pins["metadata_sha256"],
        "--coverage-parent-source-sha256",
        pins["source_python_sha256"],
    ]
    return migration, common + ["--resume"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--parent-step", type=int, required=True)
    p.add_argument("--parent-manifest-sha256", required=True)
    p.add_argument("--parent-metadata-sha256", required=True)
    p.add_argument("--parent-source-sha256", required=True)
    p.add_argument("--original-stage-spec", type=Path, required=True)
    p.add_argument("--original-stage-spec-sha256", required=True)
    p.add_argument("--output-checkpoints", required=True)
    p.add_argument("--plan", type=Path, required=True)
    args = p.parse_args()
    if args.output_checkpoints.rstrip("/") == args.parent_checkpoint.rstrip("/"):
        raise ValueError("A sampler transition requires a separate output namespace")
    if file_hash(args.original_stage_spec) != args.original_stage_spec_sha256:
        raise ValueError("Original stage specification pin differs")
    parent = read_committed_metadata(
        args.parent_checkpoint, args.parent_step, include_receipt=True
    )
    identity = copy.deepcopy(parent["resolved_config"])
    identity.update(
        coverage_sampling="remaining_pool_epoch_v1",
        coverage_transition_step=args.parent_step,
        coverage_transition_batch_size=identity["batch_size"],
    )
    expected, lineage = admit_coverage_transition(
        parent,
        identity,
        parent["tokenizer_identity"],
        parent["data_manifest_identity"],
        args.parent_step,
        args.parent_manifest_sha256,
        args.parent_metadata_sha256,
        args.parent_source_sha256,
    )
    if file_hash(args.data / "manifest.json") != parent["data_manifest_identity"]:
        raise ValueError("Plan must use the exact same prepared corpus")
    tokens, manifest = load_prepared_rows(
        args.data, SimpleNamespace(**identity["encoder"])
    )
    rows = RemainingCoverageMixtureRows(
        args.data,
        tokens,
        manifest,
        identity["seed"],
        identity["language_exponent"],
        args.parent_step,
        identity["batch_size"],
    )
    if rows.completion_step > identity["steps"]:
        raise ValueError("Coverage would exceed the preserved learning-rate schedule")
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    config = args.plan.parent / "coverage-encoder.json"
    config.write_text(json.dumps(identity["encoder"], indent=2) + "\n")
    pins = dict(
        manifest_sha256=args.parent_manifest_sha256,
        metadata_sha256=args.parent_metadata_sha256,
        source_python_sha256=args.parent_source_sha256,
    )
    original = json.loads(args.original_stage_spec.read_text())
    migration, resume = transition_commands(
        original["training_argv"],
        data=args.data,
        config=config,
        output=args.output_checkpoints,
        parent=args.parent_checkpoint,
        step=args.parent_step,
        completion=rows.completion_step,
        pins=pins,
    )
    plan = dict(
        schema_version=1,
        status="planned_not_TPU_qualified",
        parent_checkpoint=args.parent_checkpoint,
        transition=lineage,
        expected_parent_identity=expected,
        new_identity=identity,
        corpus_rows=len(tokens),
        remaining_rows=rows.total_unseen_rows,
        remaining_nonpadding_tokens=rows.total_unseen_tokens,
        completion_step=rows.completion_step,
        additional_updates=rows.completion_step - args.parent_step,
        migration_argv=migration,
        resume_argv=resume,
        final_batch_padding=True,
        coverage_scope="Prepared row IDs, not semantic or historical-text deduplication",
    )
    payload = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    args.plan.write_text(payload)
    print(
        json.dumps(
            dict(
                plan=str(args.plan),
                sha256=hashlib.sha256(payload.encode()).hexdigest(),
                remaining_rows=rows.total_unseen_rows,
                completion_step=rows.completion_step,
                physical_qualification_passed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
