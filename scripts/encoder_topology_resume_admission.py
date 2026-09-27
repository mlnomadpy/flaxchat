"""Read-only admission for changing the TPU count of a frozen encoder stage.

This checks the exact-resume invariants before a separate, guarded TPU run.
It never provisions hardware, rewrites checkpoint metadata, or changes batch size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(plan: dict, draft: dict, report: dict, identity: dict, checkpoint: dict,
          *, target_devices: int, target_processes: int) -> dict:
    if target_devices not in (8, 16) or target_processes < 1:
        raise ValueError("Only explicitly reviewed 8- or 16-chip v5e targets are accepted")
    if target_processes != 1:
        raise ValueError("Multi-host stage supervisor needs rank-aware evidence ownership")
    if report.get("complete") is not True or report.get("status") != "segment_finished":
        raise ValueError("Prior segment did not finish")
    if report.get("end_decision", {}).get("resume_admitted_stage") is not True:
        raise ValueError("Prior quality gate rejected continuation")
    step = report.get("end_step")
    if checkpoint.get("complete") is not True or checkpoint.get("step") != step:
        raise ValueError("No integrity-checked exact-boundary checkpoint receipt")
    if checkpoint != report.get("end_checkpoint") or checkpoint.get("exact_cursor_verified") is not True:
        raise ValueError("Checkpoint receipt differs from prior report")
    if report.get("checkpoint_prefix") is None:
        raise ValueError("Missing checkpoint prefix")
    segments = plan.get("segments", [])
    next_segments = [s for s in segments if s.get("start_step") == step]
    if len(next_segments) != 1 or next_segments[0].get("segment") != report.get("segment", 0) + 1:
        raise ValueError("No unique next segment at the checkpoint boundary")
    if identity.get("event") != "input_preflight" or identity.get("passed") is not True:
        raise ValueError("Missing prior frozen input identity")
    frozen = identity.get("input_identity", {})
    if frozen.get("resolved_config") != identity.get("config"):
        raise ValueError("Prior input identity/config disagree")
    if frozen.get("source_python_sha256") is None or frozen.get("data_manifest") is None:
        raise ValueError("Incomplete core-source or data identity")
    argv = draft.get("argv_without_output", [])

    def arg(name: str) -> str:
        if argv.count(name) != 1:
            raise ValueError(f"Missing or duplicate {name}")
        return argv[argv.index(name) + 1]

    batch = int(arg("--batch-size"))
    accumulation = int(arg("--accumulation-steps"))
    if (int(arg("--fsdp")) != 1 or int(arg("--steps")) != plan.get("horizon_steps")
            or int(arg("--seed")) != plan.get("seed")
            or frozen["resolved_config"].get("batch_size") != batch
            or frozen["resolved_config"].get("accumulation_steps") != accumulation
            or frozen["resolved_config"].get("steps") != plan.get("horizon_steps")):
        raise ValueError("Frozen optimizer/sampler/schedule recipe differs")
    if batch % (accumulation * target_devices):
        raise ValueError("Global batch is not divisible by accumulation and devices")
    if target_devices == 16:
        raise ValueError("16-chip v5e requires multi-host worker and evidence coordination")
    return {
        "static_preflight_passed": True,
        "physical_tpu_validation_required": True,
        "next_segment": next_segments[0]["segment"],
        "exact_resume_step": step,
        "checkpoint_prefix": report["checkpoint_prefix"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "core_source_sha256": frozen["source_python_sha256"],
        "global_batch": batch,
        "accumulation_steps": accumulation,
        "target_devices": target_devices,
        "target_processes": target_processes,
        "per_device_microbatch": batch // (accumulation * target_devices),
        "batch_or_schedule_change_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "draft", "report", "identity", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-devices", type=int, required=True)
    parser.add_argument("--target-processes", type=int, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in
             ("plan", "draft", "report", "identity", "checkpoint")}
    evidence = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if digest(paths["identity"]) != evidence["report"].get("identity_sha256"):
        raise ValueError("Prior input identity hash differs from report")
    result = check(**evidence, target_devices=args.target_devices,
                   target_processes=args.target_processes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
