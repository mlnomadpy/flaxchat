"""Choose the smallest safe GitHub Actions validation scope for a change set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess


FULL_TRIGGERS = {
    "pyproject.toml",
    "pixi.toml",
    "pixi.lock",
    ".github/workflows/cpu-tests.yml",
}

TEST_GROUPS = {
    "tests/test_encoder_training.py": ("tests/test_yat_backward_training.py",),
    "tests/yat_attention_oracle.py": ("tests/test_yat_gradient_oracle.py", "tests/test_yat_tiled_oracle.py"),
    "scripts/summarize_jax_trace.py": ("tests/test_jax_trace_summary.py",),
    "scripts/summarize_encoder_gradient_health.py": ("tests/test_encoder_gradient_health.py",),
    "scripts/launcher_topology.py": ("tests/test_launcher_topology.py",),
    "scripts/numerical_evidence.py": ("tests/test_numerical_evidence.py", "tests/test_mlm_accumulation_replay.py"),
    "scripts/replay_mlm_accumulation.py": ("tests/test_mlm_accumulation_replay.py", "tests/test_local_mlm_accumulation.py", "tests/test_tied_embedding_diagnostic.py"),
    "scripts/diagnose_tied_embedding.py": ("tests/test_tied_embedding_diagnostic.py",),
    "scripts/diagnostic_boundaries.py": ("tests/test_diagnostic_boundaries.py", "tests/test_mlm_accumulation_replay.py"),
    "scripts/diagnose_encoder_batching.py": ("tests/test_encoder_batching_diagnostic.py",),
    "scripts/campaign_barrier.py": ("tests/test_campaign_barrier.py", "tests/test_yat_pilot.py"),
    "scripts/validate_yat_mmbert.py": ("tests/test_yat_pilot.py", "tests/test_encoder_training.py", "tests/test_yat_mmbert_recipe.py", "tests/test_yat_bf16.py", "tests/test_yat_windowed.py"),
    "scripts/train_yat_mmbert.py": ("tests/test_yat_backward_training.py", "tests/test_yat_mmbert_recipe.py", "tests/test_yat_encoder.py", "tests/test_yat_bf16.py", "tests/test_encoder_training.py"),
    "scripts/calibrate_encoder_classifier.py": ("tests/test_classifier_calibration.py", "tests/test_encoder_classification_plan.py"),
    "scripts/stage_progress.py": ("tests/test_stage_progress.py", "tests/test_encoder_scale.py"),
    "scripts/validate_encoder_v6.py": ("tests/test_encoder_v6_worker.py",),
    "scripts/validate_encoder_quality_pilot.py": ("tests/test_encoder_quality_pilot.py", "tests/test_encoder_quality_plan.py",),
    "scripts/plan_encoder_quality.py": ("tests/test_encoder_quality_plan.py",),
    "scripts/validate_encoder_endurance.py": ("tests/test_encoder_endurance_plan.py", "tests/test_encoder_scale.py", "tests/test_gcp_tpu_run.py"),
    "scripts/export_encoder_retrieval.py": ("tests/test_encoder_retrieval_export.py", "tests/test_retrieval.py", "tests/test_encoder_bitext_campaign.py"),
    "scripts/export_encoder_bitext_campaign.py": ("tests/test_encoder_bitext_campaign.py", "tests/test_encoder_retrieval_export.py", "tests/test_encoder_bitext_evaluation.py"),
    "scripts/compare_encoder_bitext_campaign.py": ("tests/test_encoder_bitext_campaign.py", "tests/test_encoder_bitext_evaluation.py"),
    "scripts/evaluate_retrieval_embeddings.py": ("tests/test_retrieval.py", "tests/test_encoder_bitext_evaluation.py", "tests/test_encoder_retrieval_export.py"),
    "scripts/audit_encoder_documents.py": ("tests/test_encoder_document_overlap.py", "tests/test_contamination.py"),
    "scripts/audit_encoder_overlap.py": ("tests/test_encoder_overlap.py", "tests/test_contamination.py"),
    "scripts/prepare_encoder_corpus.py": ("tests/test_encoder_corpus.py",),
    "scripts/plan_encoder_classification.py": ("tests/test_encoder_classification_plan.py",),
    "scripts/finetune_encoder_classifier.py": ("tests/test_encoder_classification.py", "tests/test_encoder_ner_finetuning.py"),
    "scripts/evaluate_encoder_ner.py": ("tests/test_encoder_ner_finetuning.py", "tests/test_ner.py"),
    "scripts/prepare_encoder_ner.py": ("tests/test_encoder_ner_preparation.py", "tests/test_ner.py", "tests/test_encoder_ner_finetuning.py"),
    "scripts/prepare_encoder_classification.py": ("tests/test_encoder_classification.py",),
    "scripts/workload_deadline.py": ("tests/test_workload_deadline.py", "tests/test_encoder_projection_benchmark.py"),
    "scripts/validate_test_suite.py": ("tests/test_tpu_validation.py",),
    "scripts/validate_encoder_scale.py": ("tests/test_encoder_scale.py", "tests/test_stage_progress.py"),
    "scripts/evaluate_encoder_xnli.py": ("tests/test_encoder_scale.py",),
    "scripts/compare_encoder_xnli.py": ("tests/test_encoder_scale.py",),
    "scripts/summarize_encoder_scale.py": ("tests/test_encoder_scale.py",),
    "scripts/preflight_encoder_archive.py": ("tests/test_encoder_archive.py",),
    "scripts/summarize_encoder_validation.py": ("tests/test_encoder_qualification.py",),
    "scripts/validate_encoder_tpu.py": ("tests/test_encoder_qualification.py", "tests/test_gcp_tpu_run.py"),
    "scripts/validate_encoder_interruption.py": ("tests/test_encoder_qualification.py",),
    "scripts/diagnose_encoder_projection.py": ("tests/test_encoder_projection_benchmark.py",),
    "scripts/validate_projection_campaign.py": ("tests/test_encoder_projection_benchmark.py",),
    "scripts/prepare_projection_fixture.py": ("tests/test_encoder_training.py", "tests/test_encoder_projection_benchmark.py"),
    "scripts/benchmark_encoder_projection.py": ("tests/test_encoder_projection_benchmark.py", "tests/test_fused_cross_entropy.py"),
    "scripts/benchmark_yat.py": ("tests/test_yat_bf16.py", "tests/test_yat_encoder.py", "tests/test_numerical_evidence.py"),
    "scripts/compare_yat_training.py": ("tests/test_yat_training_comparison.py", "tests/test_encoder_scale.py"),
    "scripts/compare_encoder_projection.py": ("tests/test_encoder_projection_benchmark.py",),
    "scripts/validate_encoder_projection.py": ("tests/test_fused_cross_entropy.py", "tests/test_encoder.py", "tests/test_yat_encoder.py", "tests/test_yat_bf16.py", "tests/test_encoder_projection_benchmark.py"),
    "scripts/validate_encoder_reference.py": ("tests/test_encoder_reference.py", "tests/test_encoder.py"),
    "scripts/plan_encoder_endurance.py": ("tests/test_encoder_endurance_plan.py", "tests/test_encoder_scale.py"),
    "scripts/convert_encoder_checkpoint.py": ("tests/test_encoder.py",),
    "scripts/validate_encoder_checkpoint.py": ("tests/test_encoder.py",),
    "flaxchat/profiling.py": ("tests/test_profiling.py",),
    "flaxchat/yat_attention.py": ("tests/test_yat_windowed.py", "tests/test_yat_bf16.py"),
    "scripts/train_encoder.py": ("tests/test_yat_backward_training.py", "tests/test_profiling.py", "tests/test_encoder_training.py", "tests/test_encoder_stage_initialization.py", "tests/test_encoder.py", "tests/test_encoder_snapshot.py"),
    "scripts/prepare_encoder_data.py": ("tests/test_encoder_training.py",),
    "scripts/prepare_encoder_bitext.py": ("tests/test_encoder_bitext.py", "tests/test_bitext.py", "tests/test_encoder_retrieval_export.py"),
    "scripts/evaluate_encoder_bitext.py": ("tests/test_encoder_bitext_evaluation.py", "tests/test_bitext.py", "tests/test_encoder_retrieval_export.py"),
    "scripts/validate_encoder_mlm_pair.py": ("tests/test_encoder_mlm_pair_worker.py", "tests/test_encoder_mlm_comparison.py",),
    "scripts/compare_encoder_mlm.py": ("tests/test_encoder_mlm_comparison.py",),
    "scripts/evaluate_encoder.py": ("tests/test_encoder_training.py", "tests/test_encoder_initial_evaluation.py", "tests/test_encoder_evaluation_sampling.py"),
    "scripts/compare_tpu_slices.py": ("tests/test_slice_comparison.py",),
    "scripts/audit_token_overlap.py": ("tests/test_contamination.py",),
    "scripts/evaluate_prepared_checkpoint.py": ("tests/test_prepared_evaluation.py", "tests/test_contamination.py"),
    "scripts/train_gpt2.py": ("tests/test_token_pool.py", "tests/test_finetuning_resume.py", "tests/test_training_quality_gate.py"),
    "scripts/validate_training_quality.py": ("tests/test_training_quality_gate.py",),
    "scripts/gcp_spot_supervisor.py": ("tests/test_operations.py", "tests/test_gcp_cleanup_guard.py"),
    "scripts/gcp_cleanup_guard.py": ("tests/test_gcp_cleanup_guard.py",),
    "scripts/gcp_tpu_run.py": ("tests/test_gcp_tpu_run.py",),
    "scripts/gcp_tpu_preflight.py": ("tests/test_gcp_tpu_preflight.py",),
    "scripts/tpu_scale_plan.py": ("tests/test_tpu_scale_plan.py",),
    "infra/tpu/spot_watchdog.py": ("tests/test_spot_watchdog.py",),
    "benchmarks/": (
        "tests/test_benchmark_compare.py",
        "tests/test_benchmark_protocol.py",
        "tests/test_matched_benchmark.py",
        "tests/test_training_scaling.py",
    ),
    "accelerators/kaggle/": ("tests/test_kaggle_launcher.py",),
    "scripts/kaggle_tpu_tests.py": ("tests/test_kaggle_launcher.py",),
    "scripts/kaggle_matched_benchmarks.py": ("tests/test_kaggle_launcher.py",),
    "scripts/run_matched_benchmarks.py": ("tests/test_matched_benchmark.py",),
    "scripts/multihost_acceptance.py": ("tests/test_multihost_acceptance.py",),
    "scripts/check_docs.py": (),
    "scripts/check_coverage.py": ("tests/test_quality_policy.py",),
    "scripts/ci_scope.py": ("tests/test_ci_scope.py",),
    "scripts/checkpoint_demo.py": ("tests/test_published_artifact.py",),
    "scripts/checkpoint_portability.py": ("tests/test_checkpoint_topology.py",),
    "scripts/verify_artifact.py": ("tests/test_published_artifact.py",),
    ".github/workflows/release.yml": ("tests/test_quality_policy.py",),
    ".github/workflows/deploy.yaml": ("tests/test_quality_policy.py",),
    ".github/workflows/kaggle-tpu.yml": ("tests/test_quality_policy.py",),
    ".github/workflows/macos-compatibility.yml": ("tests/test_quality_policy.py",),
    "infra/tpu/": ("tests/test_quality_policy.py",),
}


def _is_test(path: str) -> bool:
    candidate = PurePosixPath(path)
    return candidate.parent == PurePosixPath("tests") and candidate.name.startswith("test_") and candidate.suffix == ".py"


def select_scope(changed_paths: list[str], *, force_full: bool = False) -> dict[str, object]:
    """Return deterministic workflow outputs for the supplied repository paths."""
    paths = sorted({path.strip() for path in changed_paths if path.strip()})
    forced = force_full or not paths
    full = forced or any(
        path in FULL_TRIGGERS or path.startswith("flaxchat/") or path.startswith("tasks/")
        for path in paths
    )
    selected: set[str] = set()
    if not full:
        for path in paths:
            if _is_test(path):
                selected.add(path)
            for prefix, tests in TEST_GROUPS.items():
                if path == prefix or path.startswith(prefix):
                    selected.update(tests)
            if path.startswith("scripts/") and path not in TEST_GROUPS:
                selected.update(("tests/test_pipeline.py", "tests/test_stage_functions.py"))

    multidevice = forced or any(
        path.startswith(("flaxchat/sharding", "flaxchat/checkpoint", "flaxchat/training", "flaxchat/common", "flaxchat/gpt", "flaxchat/encoder", "flaxchat/yat", "flaxchat/mlm", "flaxchat/fused_cross_entropy"))
        or path in {"tests/test_sharding.py", "tests/test_checkpoint_topology.py", "scripts/train_gpt2.py",
                    "tests/test_token_pool.py", "tests/test_distributed_cpu.py", "tests/test_encoder_training.py", "scripts/train_encoder.py",
                    "tests/test_fused_cross_entropy.py", "tests/test_encoder.py", "tests/test_yat_encoder.py", "tests/test_yat_bf16.py", "tests/test_yat_local_shards.py", "tests/test_local_mlm_accumulation.py",
                    "scripts/finetune_encoder_classifier.py", "tests/test_encoder_classification.py",
                    "scripts/evaluate_encoder_ner.py", "tests/test_encoder_ner_finetuning.py",
                    "scripts/replay_mlm_accumulation.py", "scripts/diagnose_tied_embedding.py",
                    "tests/test_tied_embedding_diagnostic.py"}
        or path in FULL_TRIGGERS
        for path in paths
    )
    e2e = forced or any(
        path.startswith(("flaxchat/engine", "flaxchat/gpt", "flaxchat/stages/"))
        or path.startswith("tasks/")
        or path in FULL_TRIGGERS
        for path in paths
    )
    return {
        "mode": "full" if full else "targeted",
        "tests": sorted(selected),
        "run_audit": forced or (full and any(path in {"pyproject.toml", "pixi.toml", "pixi.lock"} for path in paths)),
        "run_build": forced or (full and any(path == "pyproject.toml" or path.startswith("flaxchat/") for path in paths)),
        "run_multidevice": multidevice,
        "run_e2e": e2e,
    }


def changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    paths = [] if args.full else changed_paths(args.base, args.head) if args.base else []
    scope = select_scope(paths, force_full=args.full)
    if args.github_output:
        lines = [
            f"mode={scope['mode']}",
            f"tests={json.dumps(scope['tests'], separators=(',', ':'))}",
            *(f"{key}={str(scope[key]).lower()}" for key in (
                "run_audit", "run_build", "run_multidevice", "run_e2e"
            )),
        ]
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write("\n".join(lines) + "\n")
    else:
        print(json.dumps(scope, indent=2))


if __name__ == "__main__":
    main()
