"""Current-source v6e qualification with durable per-stage evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.workload_deadline import WorkloadDeadline
from scripts.stage_progress import run_stage


def preflight_imports():
    # Import checks must run in a short-lived child. Importing flaxchat in
    # this coordinator initializes JAX and monopolizes the TPU for later stages.
    modules = (
        "flaxchat.yat",
        "flaxchat.encoder",
        "scripts.train_encoder",
        "scripts.validate_encoder_projection",
        "scripts.validate_encoder_scale",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; [importlib.import_module(m) for m in "
            + repr(modules)
            + "]",
        ],
        env=os.environ | {"JAX_PLATFORMS": "cpu"},
        check=True,
        timeout=120,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    preflight_imports()
    required = [
        "tests/test_yat_bf16.py",
        "tests/test_yat_encoder.py",
        "tests/test_encoder_training.py",
        "artifacts/encoder-qualify-0923/validate_v6.py",
        "scripts/benchmark_yat.py",
    ]
    for name in required:
        if not Path(name).is_file():
            raise ValueError("Missing workload file: " + name)
    if args.preflight_only:
        print(json.dumps(dict(import_preflight_passed=True)))
        return 0
    if not args.prefix:
        parser.error("--prefix required")
    probe = "import jax,json; print(json.dumps([dict(id=d.id,process=d.process_index,kind=d.device_kind,platform=d.platform) for d in jax.devices()]))"
    inventory = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", probe],
            text=True,
            timeout=120,
            env=os.environ | {"JAX_PLATFORMS": "tpu"},
        )
    )
    if (
        len(inventory) != 4
        or len({d["process"] for d in inventory}) != 1
        or any(
            d["platform"] != "tpu" or "v6" not in d["kind"].lower() for d in inventory
        )
    ):
        raise ValueError(
            "Expected four physical v6e chips on one host: " + str(inventory)
        )
    root = Path("/tmp/v6-current-evidence")
    root.mkdir(exist_ok=False)
    (root / "inventory.json").write_text(json.dumps(inventory, indent=2))
    deadline = WorkloadDeadline(
        int(os.environ["FLAXCHAT_WORKLOAD_TIMEOUT_SECONDS"]) - 90
    )
    env = os.environ | {
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "JAX_PLATFORMS": "tpu",
    }
    stages = []
    commands = [
        (
            "yat",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_yat_bf16.py",
                "tests/test_yat_encoder.py",
                "tests/test_encoder_training.py",
                "-k",
                "yat",
                "-q",
                f"--junitxml={root}/yat-tests.xml",
            ],
            900,
        ),
        (
            "benchmark",
            [
                sys.executable,
                "-m",
                "scripts.benchmark_yat",
                "--output",
                str(root / "benchmark.json"),
                "--collision-stress",
            ],
            300,
        ),
        (
            "backend",
            [
                sys.executable,
                "artifacts/encoder-qualify-0923/validate_v6.py",
                args.prefix + "/backend",
            ],
            1900,
        ),
    ]
    for name, argv, limit in commands:
        try:

            def publish(path, seconds):
                subprocess.run(
                    ["gcloud", "storage", "cp", str(path), args.prefix + "/progress/"],
                    check=True,
                    timeout=min(30, seconds),
                    capture_output=True,
                )

            code = run_stage(
                argv,
                log=root / f"{name}.log",
                env=env,
                timeout=min(limit, deadline.remaining() - 65),
                publish=publish,
            )
        except Exception as exc:
            code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 1
            (root / f"{name}-error.txt").write_text(repr(exc))
        stages.append(dict(name=name, returncode=code, passed=code == 0))
        (root / "summary.json").write_text(
            json.dumps(
                dict(stages=stages, production_quality_qualified=False), indent=2
            )
        )
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                *map(str, root.iterdir()),
                args.prefix + "/evidence/",
            ],
            check=True,
            timeout=60,
        )
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
