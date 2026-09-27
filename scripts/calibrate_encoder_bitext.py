"""Time two full subsets without computing quality scores or provisioning TPU."""

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import jax

from flaxchat.checkpoint import load_checkpoint_metadata
from flaxchat.encoder import EncoderConfig
from scripts.export_encoder_bitext_campaign import verify_plan
from scripts.export_encoder_retrieval import EmbeddingSession
from scripts.workload_deadline import WorkloadDeadline


def estimate(
    entries, timings, *, batch_size, load_seconds, verification_seconds
) -> dict[str, Any]:
    """Keep first-call compilation and conservatively floor each subset at warm cost."""
    selected = sorted(entries, key=lambda e: (-e["pairs"], e["subset"]))[:2]
    if (
        len(selected) != 2
        or len(timings) != 2
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError(
            "Two distinct full calibration subsets and positive batch size required"
        )
    if [e["subset"] for e in selected] != [t["subset"] for t in timings]:
        raise ValueError("Calibration subsets differ from deterministic selection")
    if any(not math.isfinite(v) or v < 0 for v in (load_seconds, verification_seconds)):
        raise ValueError("Invalid setup timing")
    for e, t in zip(selected, timings, strict=True):
        if (
            t["pairs"] != e["pairs"]
            or not math.isfinite(t["seconds"])
            or t["seconds"] <= 0
        ):
            raise ValueError("Incomplete subset or invalid timing")
    warm_batches = 2 * math.ceil(selected[1]["pairs"] / batch_size)
    warm_seconds = timings[1]["seconds"]
    steady_seconds = sum(
        warm_seconds * max(1, 2 * math.ceil(e["pairs"] / batch_size) / warm_batches)
        for e in entries
    )
    return dict(
        subsets=len(entries),
        pairs=sum(e["pairs"] for e in entries),
        batch_size=batch_size,
        full_export_seconds=load_seconds
        + 2 * verification_seconds
        + timings[0]["seconds"]
        + steady_seconds,
        method="Retain complete first export for compilation; every full subset costs at least one measured warm subset; load and two verifications included. Pair planner adds margin and scoring/upload reserve.",
        measured_subsets=timings,
        scores_computed=False,
        production_quality_qualified=False,
    )


def run(plan_path, role, output, *, batch_size=32, max_seconds=1500):
    if role not in ("baseline", "candidate"):
        raise ValueError("Invalid model role")
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing existing calibration output")
    deadline = WorkloadDeadline(max_seconds)
    before = time.monotonic()
    plan, identity, entries, directories, snapshot = verify_plan(plan_path)
    verification_seconds = time.monotonic() - before
    selected = sorted(entries, key=lambda e: (-e["pairs"], e["subset"]))[:2]
    if len(selected) != 2:
        raise ValueError("Need at least two subsets for calibration")
    metadata = load_checkpoint_metadata(
        plan["candidate"]["checkpoint"], step=plan["candidate"]["step"]
    )
    if (
        metadata.get("step") != plan["candidate"]["step"]
        or metadata.get("model_family") != "modernbert"
    ):
        raise ValueError("Checkpoint identity mismatch")
    config = EncoderConfig(**metadata["resolved_config"]["encoder"])
    before = time.monotonic()
    session = (
        EmbeddingSession.from_pretrained(plan["baseline"]["snapshot"], config=config)
        if role == "baseline"
        else EmbeddingSession(
            plan["candidate"]["checkpoint"], step=plan["candidate"]["step"]
        )
    )
    load_seconds = time.monotonic() - before
    timings = []
    for entry in selected:
        deadline.remaining()
        root = directories[entry["subset"]]
        before = time.monotonic()
        session.export(
            root / "queries",
            root / "corpus",
            root / "judgments.json",
            output / entry["subset"],
            batch_size=batch_size,
        )
        timings.append(
            dict(
                subset=entry["subset"],
                pairs=entry["pairs"],
                seconds=time.monotonic() - before,
            )
        )
        print(json.dumps(timings[-1]), flush=True)
    after, after_identity, _, _, after_snapshot = verify_plan(plan_path)
    if after != plan or after_identity != identity or snapshot != after_snapshot:
        raise ValueError("Frozen inputs changed during calibration")
    deadline.remaining()
    result = estimate(
        entries,
        timings,
        batch_size=batch_size,
        load_seconds=load_seconds,
        verification_seconds=verification_seconds,
    )
    result.update(
        role=role,
        plan_sha256=identity,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        device_count=jax.device_count(),
        processes=jax.process_count(),
        jax_version=jax.__version__,
    )
    (output / "estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=1500)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.plan,
                args.role,
                args.output,
                batch_size=args.batch_size,
                max_seconds=args.max_seconds,
            )
        )
    )


if __name__ == "__main__":
    main()
