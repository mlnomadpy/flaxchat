"""Run one full matched XNLI seed pair on a leased four-device TPU host."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts.stage_progress import run_stage
from scripts.validate_encoder_quality_pilot import file_hash
from scripts.workload_deadline import WorkloadDeadline


def compare(baseline, candidate, frozen, seed):
    """Require identical full development coverage and training protocol."""
    for report in (baseline, candidate):
        if (
            report.get("scope") != "end_to_end_encoder_classification"
            or report["checkpoint_step"] != frozen["steps"]
        ):
            raise ValueError("Incomplete or wrong classifier evaluation")
        metadata = report["training_metadata"]
        if metadata["step"] != frozen["steps"]:
            raise ValueError("Incomplete training checkpoint")
        recipe = metadata["resolved_config"]
        if recipe["seed"] != seed or any(
            recipe[k] != frozen[k] for k in ("steps", "batch_size", "warmup_steps")
        ):
            raise ValueError("Unmatched full training schedule")
    for key in ("evaluator_source_sha256", "runtime"):
        if not baseline.get(key) or baseline[key] != candidate.get(key):
            raise ValueError("Unmatched evaluator identity")
    left, right = baseline["training_metadata"], candidate["training_metadata"]
    for key in (
        "source_python_sha256",
        "data_manifest_identity",
        "tokenizer_identity",
        "dataset",
        "dataset_revision",
    ):
        if not left.get(key) or left[key] != right.get(key):
            raise ValueError("Unmatched training provenance")
    lrecipe, rrecipe = left["resolved_config"], right["resolved_config"]
    if {k: v for k, v in lrecipe.items() if k != "origin"} != {
        k: v for k, v in rrecipe.items() if k != "origin"
    }:
        raise ValueError("Unmatched training recipe")
    if "checkpoint" in lrecipe["origin"] or not lrecipe["origin"].get(
        "released_weights"
    ):
        raise ValueError("Baseline must originate from released weights")
    if lrecipe["origin"]["released_weights"] != rrecipe["origin"].get(
        "released_weights"
    ):
        raise ValueError("Unmatched initial weight inventory")
    argv = frozen["commands"][f"candidate-seed-{seed}"]["train"]
    origin = rrecipe["origin"]["checkpoint"]
    if origin["path"] != argv[argv.index("--encoder-checkpoint") + 1] or origin[
        "step"
    ] != int(argv[argv.index("--encoder-step") + 1]):
        raise ValueError("Wrong candidate checkpoint origin")
    scores = []
    for report in (baseline, candidate):
        by_language = {}
        for row in report["datasets"]:
            manifest = row["data_manifest"]
            language = manifest["language"]
            if (
                language in by_language
                or language not in frozen["languages"]
                or manifest["split"] != "validation"
            ):
                raise ValueError(
                    "Missing, duplicate, or unexpected development language"
                )
            expected = next(
                x
                for x in frozen["dataset_audit"]
                if x["split"] == "validation" and x["language"] == language
            )
            if (
                row["data_manifest_sha256"]
                != frozen["input_sha256"][expected["path"] + "/manifest.json"]
                or row["examples"] != expected["rows"]
            ):
                raise ValueError("Development data or coverage changed")
            if (
                type(row["correct"]) is not int
                or not 0 <= row["correct"] <= row["examples"]
                or not math.isclose(
                    row["accuracy"], row["correct"] / row["examples"], abs_tol=1e-12
                )
                or not math.isfinite(row["loss"])
                or row["loss"] < 0
            ):
                raise ValueError("Invalid development metric")
            by_language[language] = row["accuracy"]
        if set(by_language) != set(frozen["languages"]):
            raise ValueError("Incomplete language coverage")
        scores.append(by_language)
    deltas = {
        lang: 100 * (scores[1][lang] - scores[0][lang]) for lang in frozen["languages"]
    }
    return dict(
        scope="matched_full_xnli_development",
        seed=seed,
        protocol_matched=True,
        language_delta_percentage_points=deltas,
        macro_delta_percentage_points=sum(deltas.values()) / len(deltas),
        baseline_macro_accuracy=sum(scores[0].values()) / len(deltas),
        candidate_macro_accuracy=sum(scores[1].values()) / len(deltas),
        production_quality_qualified=False,
    )


def validate_training(path, *, first, last):
    records = [
        json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")
    ]
    configs = [r for r in records if r.get("event") == "classifier_run_config"]
    if len(configs) != 1 or any(
        configs[0].get(k) != v
        for k, v in dict(devices=4, processes=1, backend="tpu").items()
    ):
        raise ValueError("Expected four physical TPU devices")
    steps = [r for r in records if r.get("event") == "classifier_train_step"]
    if [r["step"] for r in steps] != list(range(first, last + 1)):
        raise ValueError("Incomplete training horizon")
    for r in steps:
        if (
            r["updated"] is not True
            or r["examples"] != 32
            or not math.isfinite(r["loss"])
            or not math.isfinite(r["seconds"])
            or r["seconds"] <= 0
            or r["includes_compilation"] is not (r["step"] == first)
        ):
            raise ValueError("Invalid update evidence")


def run(a):
    if (
        not a.prefix.startswith("gs://")
        or int(os.environ.get("JAX_PROCESS_COUNT", "1")) != 1
    ):
        raise ValueError("Require one host and persistent evidence prefix")
    frozen = json.loads(a.plan.read_text())
    if a.seed not in frozen["seeds"]:
        raise ValueError("Seed outside frozen protocol")
    for field in ("input_sha256", "source_sha256"):
        for path, digest in frozen[field].items():
            if file_hash(path) != digest:
                raise ValueError(f"Frozen file changed: {path}")
    estimate = json.loads(a.calibration_estimate.read_text())
    if (
        estimate["steps"] != frozen["steps"]
        or estimate["evaluation_examples"] != frozen["evaluation_examples"]
        or estimate["calibration_updates"] != 100
    ):
        raise ValueError("Unmatched calibration estimate")
    deadline = WorkloadDeadline(
        min(6500, int(os.environ["FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS"]) - 90)
    )
    if estimate["estimated_total_seconds"] * 2 + 180 > deadline.remaining():
        raise ValueError("Both complete recipes must fit lease")
    a.output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = dict(
        passed=False,
        quality_qualified=False,
        seed=a.seed,
        stages=[],
        plan_sha256=file_hash(a.plan),
        worker_sha256=file_hash(__file__),
    )
    env = os.environ | {
        "JAX_PLATFORMS": "tpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    }

    def persist():
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                *map(str, a.output.iterdir()),
                a.prefix + "/evidence/",
            ],
            check=True,
            timeout=deadline.remaining(60),
        )

    def stage(name, argv):
        argv = list(argv)
        argv[0] = sys.executable

        def publish(path, seconds):
            subprocess.run(
                ["gcloud", "storage", "cp", str(path), a.prefix + "/progress/"],
                check=True,
                timeout=min(30, seconds),
                capture_output=True,
            )

        log = a.output / f"{name}.log"
        code = run_stage(
            argv, log=log, env=env, timeout=deadline.remaining() - 65, publish=publish
        )
        summary["stages"].append(dict(name=name, returncode=code))
        persist()
        if code:
            raise RuntimeError(f"{name} failed: {code}")
        return log

    try:
        reports = []
        for role in ("baseline", "candidate"):
            commands = frozen["commands"][f"{role}-seed-{a.seed}"]
            train = list(commands["train"])
            resumed = role == "baseline" and a.seed == 17
            if resumed:
                train += ["--resume"]
            log = stage(role + "-train", train)
            validate_training(log, first=101 if resumed else 1, last=frozen["steps"])
            evaluate = list(commands["evaluate"])
            report = a.output / f"{role}.json"
            evaluate[evaluate.index("--report") + 1] = str(report)
            stage(role + "-evaluate", evaluate)
            reports.append(json.loads(report.read_text()))
        result = compare(reports[0], reports[1], frozen, a.seed)
        (a.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        summary["passed"] = True
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        persist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--calibration-estimate", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output", type=Path, default=Path("/tmp/encoder-classifier-pair"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
