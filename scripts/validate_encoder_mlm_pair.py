"""Bounded, single-host TPU comparison of initial and trained encoder weights.

The coordinator never imports JAX. It provisions no resources; the caller must
provide a lease, cleanup guard, and an immutable source/data recipe.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts.compare_encoder_mlm import compare
from scripts.validate_encoder_quality_pilot import file_hash
from scripts.workload_deadline import WorkloadDeadline


def commands(plan, output):
    result = {}
    for name in ("baseline", "candidate"):
        argv = list(plan["commands_in_order"][name])
        if argv[1:3] != ["-m", "scripts.evaluate_encoder"]:
            raise ValueError("Expected encoder evaluator entrypoint")
        for flag, value in (
            ("--checkpoint", plan["checkpoint"]),
            ("--checkpoint-step", str(plan["checkpoint_step"])),
            ("--max-rows", str(plan["validation_rows"])),
            ("--seed", str(plan["seed"])),
        ):
            if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
                raise ValueError(f"Unmatched planned evaluation: {flag}")
        if ("--initial-pretrained" in argv) != (name == "baseline"):
            raise ValueError("Wrong parameter source in planned evaluation")
        if argv.count("--output") != 1:
            raise ValueError("Expected one evaluation output")
        for flag in ("--data", "--train-data"):
            directory = Path(argv[argv.index(flag) + 1])
            if any(
                str(directory / file) not in plan["input_sha256"]
                for file in ("manifest.json", "tokens.npy")
            ):
                raise ValueError("Evaluation datasets must be pinned")
        argv[0] = sys.executable
        argv[argv.index("--output") + 1] = str(output / f"{name}.json")
        result[name] = argv
    return result


def run(args):
    if (
        not args.prefix.startswith("gs://")
        or int(os.environ.get("JAX_PROCESS_COUNT", "1")) != 1
    ):
        raise ValueError("Requires GCS evidence prefix and one TPU host")
    plan = json.loads(args.plan.read_text())
    if plan["checkpoint_step"] < 0 or plan["validation_rows"] <= 0:
        raise ValueError("Invalid planned checkpoint step or row count")
    planned = commands(plan, args.output)
    for name, digest in plan["input_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = WorkloadDeadline(
        min(1800, int(os.environ["FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS"]) - 90)
    )
    env = os.environ | {
        "JAX_PLATFORMS": "tpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    }
    summary: dict[str, Any] = dict(
        passed=False,
        quality_qualified=False,
        stages=[],
        checkpoint_step=plan["checkpoint_step"],
    )

    def persist():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                *map(str, args.output.iterdir()),
                args.prefix + "/evidence/",
            ],
            check=True,
            timeout=deadline.remaining(60),
        )

    try:
        probe = "import jax,json; print(json.dumps([dict(platform=d.platform,kind=d.device_kind,process=d.process_index) for d in jax.devices()]))"
        inventory = json.loads(
            subprocess.check_output(
                [sys.executable, "-c", probe],
                env=env,
                text=True,
                timeout=deadline.remaining(120),
            )
        )
        if (
            len(inventory) != 4
            or len({d["process"] for d in inventory}) != 1
            or any(d["platform"] != "tpu" for d in inventory)
        ):
            raise ValueError("Expected four physical TPU devices on one host")
        (args.output / "inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        for name, argv in planned.items():
            remaining = deadline.remaining() - 65
            if remaining <= 0:
                raise TimeoutError(
                    "Insufficient time for evaluation and evidence upload"
                )
            with (args.output / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    argv,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                )
            summary["stages"].append(dict(name=name, returncode=result.returncode))
            persist()
            if result.returncode:
                raise RuntimeError(f"{name} evaluation failed: {result.returncode}")
        baseline, candidate = [
            json.loads((args.output / f"{name}.json").read_text()) for name in planned
        ]
        comparison = compare(baseline, candidate)
        if (
            candidate["evaluated_rows"] != plan["validation_rows"]
            or candidate["checkpoint_step"] != plan["checkpoint_step"]
            or candidate["seed"] != plan["seed"]
            or candidate["checkpoint_context"] != plan["checkpoint"]
        ):
            raise ValueError(
                "Evaluation did not cover the full planned rows/checkpoint"
            )
        if candidate["backend"] != "tpu" or len(candidate["devices"]) != 4:
            raise ValueError("Evaluation report did not execute on four TPUs")
        (args.output / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        summary["passed"] = True
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mlm-pair"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
