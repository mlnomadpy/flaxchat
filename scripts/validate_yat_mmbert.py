"""Validate physical YAT training/recovery, then launch a budgeted adaptation pilot.

Runs inside an externally bounded single- or multi-host TPU allocation.
No JAX imports here: each child owns the TPU runtime.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from scripts.stage_progress import run_stage
from scripts.validate_encoder_scale import validate_updates
from scripts.validate_encoder_tpu import validate_hardware
from scripts.workload_deadline import WorkloadDeadline
from scripts.campaign_barrier import exchange


def validate_rank_mapping(process_indices, count):
    """TPU runtime ranks need not equal SSH endpoint indices."""
    if (len(process_indices) != count or any(type(i) is not int for i in process_indices)
            or sorted(process_indices) != list(range(count))):
        raise ValueError("TPU process ranks must form a complete permutation")
    return process_indices.index(0)


def primary_writer(claims):
    if any(type(claim) is not bool for claim in claims) or sum(claims) != 1:
        raise ValueError("Expected exactly one global training log writer")
    return claims.index(True)


def summarize_source_exposure(rows, target_shares):
    if target_shares is None:
        return None
    totals = dict.fromkeys(target_shares, 0)
    for row in rows:
        counts = row.get("source_nonpadding_tokens")
        if (not isinstance(counts, dict) or set(counts) != set(totals)
                or any(type(n) is not int or n < 0 for n in counts.values())
                or sum(counts.values()) != row["nonpadding_tokens"]):
            raise ValueError("Invalid source token exposure accounting")
        for name, count in counts.items():
            totals[name] += count
    total = sum(totals.values())
    if not total:
        raise ValueError("No mixture token exposure")
    return dict(nonpadding_tokens=totals, realized_token_shares={k: v / total for k, v in totals.items()},
                target_token_shares=target_shares)


def choose_steps(rows, remaining_seconds, *, target_tokens=50_000_000):
    measured = [r for r in rows if r.get("includes_compilation") is False
                and r.get("includes_profiling", False) is False]
    if len(measured) < 21:
        raise ValueError("At least 21 steady-state updates required")
    if any(
        r.get("updated") is not True
        or r.get("projection_dense_fallback") is not False
        or not math.isfinite(r["seconds"])
        or r["seconds"] <= 0
        or not math.isfinite(r["loss"])
        or r["nonpadding_tokens"] <= 0
        for r in measured
    ):
        raise ValueError("Invalid calibration")
    p90 = sorted(r["seconds"] for r in measured)[math.ceil(0.9 * len(measured)) - 1]
    mean_tokens = statistics.mean(r["nonpadding_tokens"] for r in measured)
    affordable = math.floor((remaining_seconds - 1200) / (1.5 * p90))
    steps = min(math.ceil(target_tokens / mean_tokens), affordable)
    if steps < 256:
        raise ValueError("Insufficient lease for a useful pilot and evaluation reserve")
    return dict(
        steps=steps,
        p90_seconds=p90,
        estimated_useful_tokens=steps * mean_tokens,
        target_useful_tokens=target_tokens,
        evaluation_reserve_seconds=1200,
        rate_is_measured=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--snapshot", default="artifacts/mmbert-base")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--expected-devices", type=int, default=4)
    p.add_argument("--expected-processes", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--local-gradient-accumulation", action="store_true")
    p.add_argument("--calibration-timeout-seconds", type=int, default=900)
    p.add_argument("--calibration-only", action="store_true",
                   help="Stop after bounded timing diagnostics; does not qualify recovery or launch a pilot")
    p.add_argument("--yat-local-shards", action="store_true")
    p.add_argument("--yat-attention-block-size", type=int, default=0)
    p.add_argument("--yat-global-attention-block-size", type=int, default=0)
    p.add_argument("--yat-softmax-backward", choices=["factored", "max_centered"], default="factored")
    p.add_argument("--yat-attention-alpha", type=float)
    p.add_argument("--donate-state", action="store_true")
    p.add_argument("--profile-calibration", action="store_true")
    p.add_argument("--validation-only", action="store_true", help="Complete recovery and evaluation, then stop before pilot")
    a = p.parse_args()
    if a.yat_attention_alpha is not None and (
            not math.isfinite(a.yat_attention_alpha) or a.yat_attention_alpha <= 0):
        p.error("Attention alpha must be finite and positive")
    if a.validation_only and a.calibration_only:
        p.error("Choose validation-only or calibration-only")
    if min(a.yat_attention_block_size, a.yat_global_attention_block_size) < 0:
        p.error("Attention block sizes must be nonnegative")
    rank = int(os.environ.get("JAX_PROCESS_INDEX", "0"))
    if (a.calibration_timeout_seconds < 60 or a.expected_devices < 1 or a.expected_processes < 1 or a.accumulation_steps < 1
            or not 0 <= rank < a.expected_processes or a.batch_size < 1
            or a.batch_size % (a.expected_devices * a.accumulation_steps)
            or int(os.environ.get("JAX_PROCESS_COUNT", "1")) != a.expected_processes):
        p.error("Invalid batch, accumulation, or launcher topology")
    if not a.prefix.startswith("gs://"):
        raise ValueError("Require durable GCS checkpoints")
    a.output.mkdir(parents=True, exist_ok=False)
    deadline = WorkloadDeadline(
        int(os.environ["FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS"]) - 90
    )
    env = os.environ | {
        "JAX_PLATFORMS": "tpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    }
    report: dict[str, Any] = dict(
        passed=False, physical_validated=False, training_started=False, stages=[]
    )
    target_shares = json.loads((Path(a.data) / "manifest.json").read_text()).get("source_token_shares")
    report["target_source_token_shares"] = target_shares
    evidence_prefix = a.prefix + (f"/worker-{rank}" if a.expected_processes > 1 else "")

    def persist():
        (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                *map(str, a.output.iterdir()),
                evidence_prefix + "/evidence/",
            ],
            check=True,
            timeout=60,
            capture_output=True,
        )

    def stage(name, argv, expected=0, limit=None):
        log = a.output / (name + ".log")

        def publish(path, seconds):
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(path),
                    str(path.with_suffix(".status.json")),
                    evidence_prefix + "/progress/",
                ],
                check=True,
                timeout=min(30, seconds),
                capture_output=True,
            )

        code = run_stage(
            argv, log=log, env=env, timeout=min(deadline.remaining() - 65, limit or math.inf), publish=publish
        )
        report["stages"].append(dict(name=name, returncode=code))
        persist()
        if code != expected:
            raise RuntimeError(f"{name} returned {code}, expected {expected}")
        exchange(a.prefix, a.output, name, rank, a.expected_processes, {}, deadline)
        return log

    def primary_log(name, path):
        claims = exchange(a.prefix, a.output, name + "-writer", rank, a.expected_processes,
                          any(r.get("event") == "train_step" for r in records(path)), deadline)
        writer = primary_writer(claims)
        if rank != writer:
            primary = a.output / (name + "-global.log")
            subprocess.run(["gcloud", "storage", "cp",
                a.prefix + f"/worker-{writer}/evidence/{name}.log", str(primary)], check=True,
                capture_output=True, timeout=min(60, deadline.remaining()))
            return primary
        return path

    def records(path):
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.startswith("{")
        ]

    profile_directory = a.output.parent / (a.output.name + f"-profiles-{rank}")

    def training(name, steps, *, module="scripts.train_yat_mmbert", extra=()):
        return [
            sys.executable,
            "-m",
            module,
            "--config",
            a.snapshot + "/config.json",
            "--pretrained",
            a.snapshot,
            "--data",
            a.data,
            "--output",
            a.prefix + "/" + name,
            "--steps",
            str(steps),
            "--batch-size",
            str(a.batch_size),
            "--accumulation-steps",
            str(a.accumulation_steps),
            "--save-every",
            "16" if steps == 32 else "256",
            "--compile-diagnostics",
            "--keep-checkpoints",
            "0",
            "--lr-schedule",
            "cosine",
            "--warmup-steps",
            str(min(100, max(1, steps // 20))),
            "--learning-rate",
            "2e-5",
            "--mask-probability",
            ".15",
            "--seed",
            "17",
            "--language-exponent",
            ".5",
            "--dtype",
            "bfloat16",
            "--residual-dtype",
            "bfloat16",
            "--ffn-type",
            "yat_glu",
            "--attention-score",
            "yat_softmax",
            "--yat-compute-mode",
            "bf16_adaptive",
            "--mlm-projection",
            "masked",
            "--mlm-loss-backend",
            "pallas",
            *(["--yat-local-shards"] if a.yat_local_shards else []),
            "--yat-attention-block-size", str(a.yat_attention_block_size),
            "--yat-global-attention-block-size", str(a.yat_global_attention_block_size),
            "--yat-softmax-backward", a.yat_softmax_backward,
            *(["--yat-attention-alpha", str(a.yat_attention_alpha)]
              if a.yat_attention_alpha is not None else []),
            *(["--local-gradient-accumulation"] if a.local_gradient_accumulation else []),
            *(["--donate-state"] if a.donate_state else []),
            *(["--profile-dir", str(profile_directory)] if a.profile_calibration and name == "calibration" else []),
            *extra,
        ]

    def evaluate(name, checkpoint):
        stage(
            name,
            [
                sys.executable,
                "-m",
                "scripts.evaluate_encoder",
                "--checkpoint",
                checkpoint,
                "--data",
                a.validation,
                "--train-data",
                a.data,
                "--batch-size",
                "8",
                "--max-rows",
                "256",
                "--balanced-languages",
                "--output",
                str(a.output / (name + ".json")),
            ],
        )
        return json.loads((a.output / (name + ".json")).read_text())

    try:
        log = stage(
            "hardware",
            [
                sys.executable,
                "-c",
                "import json,jax,flaxchat.common; print(json.dumps(dict(backend=jax.default_backend(),process_count=jax.process_count(),device_count=jax.device_count(),process_index=jax.process_index())))",
            ],
        )
        hardware = records(log)[0]
        validate_hardware(hardware, "multi" if a.expected_processes > 1 else "single",
                          a.expected_processes, a.expected_devices)
        mapping = exchange(a.prefix, a.output, "runtime-ranks", rank, a.expected_processes,
                           hardware["process_index"], deadline)
        validate_rank_mapping(mapping, a.expected_processes)
        report["runtime_process_indices_by_launcher_rank"] = mapping
        report["hardware"] = hardware
        if a.expected_processes == 1:
            stage(
                "yat_tests",
                [sys.executable, "-m", "pytest", "-q",
                 "tests/test_yat_encoder.py", "tests/test_yat_bf16.py"],
            )
        # Run on every physical worker, before allocating the full model.
        # These adversarial cases caught dispatch-dependent document leakage
        # that ordinary random attention comparisons did not exercise.
        stage("yat_numerical_regressions", [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_yat_bf16.py::test_repair_dispatch_does_not_change_an_unchanged_pair",
            "tests/test_yat_bf16.py::test_sensitive_distance_cotangents_preserve_zeros_and_small_nonzeros",
            "tests/test_yat_windowed.py::test_near_collision_dispatch_preserves_packed_document_isolation",
        ])
        stage("preflight", training("calibration", 32, extra=["--preflight-only"]))
        log = stage("calibration", training("calibration", 32), limit=a.calibration_timeout_seconds)
        if a.profile_calibration:
            traces = list(profile_directory.rglob("*.xplane.pb"))
            if not traces:
                raise ValueError("Calibration produced no JAX trace")
            subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(profile_directory),
                            evidence_prefix + "/profiles"], check=True, timeout=min(120, deadline.remaining() - 65))
            report["profile_files"] = [str(p.relative_to(profile_directory)) for p in traces]
        log = primary_log("calibration", log)
        report["calibration"] = validate_updates(
            log, first=1, last=32, batch=a.batch_size, length=512
        )
        calibration_rows = [r for r in records(log) if r.get("event") == "train_step"]
        report["calibration_source_exposure"] = summarize_source_exposure(calibration_rows, target_shares)
        if a.calibration_only:
            report["calibration_passed"] = True
            report["qualification_complete"] = False
            return
        interrupted = stage(
            "interrupt",
            training(
                "recovery",
                32,
                module="scripts.validate_encoder_interruption",
                extra=["--fault-step", "16"],
            ),
            -9,
        )
        if (
            "FAULT_INJECTION: encoder SIGKILL after committed step 16"
            not in interrupted.read_text()
        ):
            raise ValueError("Missing committed interruption marker")
        resumed = stage("resume", training("recovery", 32, extra=["--resume"]))
        resumed = primary_log("resume", resumed)
        validate_updates(resumed, first=17, last=32, batch=a.batch_size, length=512)
        manifests = []
        for name in ("calibration", "recovery"):
            raw = subprocess.check_output(
                [
                    "gcloud",
                    "storage",
                    "cat",
                    a.prefix + "/" + name + "/32/manifest/metadata",
                ],
                text=True,
                timeout=60,
            )
            (a.output / (name + "-manifest.json")).write_text(raw)
            manifests.append(json.loads(raw))
        for key in ("model_state", "optimizer_state", "training_state"):
            if manifests[0][key] != manifests[1][key]:
                raise ValueError(f"Recovery mismatch: {key}")
        report["physical_validated"] = True
        report["calibration_evaluation"] = evaluate(
            "calibration-evaluation", a.prefix + "/calibration"
        )
        if a.validation_only:
            report["passed"] = True
            report["production_quality_qualified"] = False
            return
        budgets = exchange(a.prefix, a.output, "pilot-budget", rank, a.expected_processes,
                           deadline.remaining() - 65, deadline)
        plan = choose_steps(calibration_rows, min(budgets))
        report["pilot_plan"] = plan
        report["training_started"] = True
        persist()
        trained = stage("pilot", training("pilot", plan["steps"]))
        trained = primary_log("pilot", trained)
        report["pilot"] = validate_updates(
            trained, first=1, last=plan["steps"], batch=a.batch_size, length=512
        )
        report["pilot_source_exposure"] = summarize_source_exposure(
            [r for r in records(trained) if r.get("event") == "train_step"], target_shares)
        report["pilot_evaluation"] = evaluate("pilot-evaluation", a.prefix + "/pilot")
        report["heldout_loss_improved"] = (
            report["pilot_evaluation"]["masked_token_loss"]
            < report["calibration_evaluation"]["masked_token_loss"]
        )
        report["passed"] = True
        report["production_quality_qualified"] = False
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
