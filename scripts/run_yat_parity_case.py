"""Run one bounded physical parity case with durable evidence and separate runtimes.

This is a worker command, not a cloud allocator. Aggregate independently
qualified case reports with validate_yat_torch_parity matrix for publication.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
from scripts.evaluation_contract import write_atomic
from scripts.release_contract import digest, validate_metrics


def run_case(
    source,
    target,
    jax_python,
    torch_python,
    evidence,
    length,
    batch_size,
    timeout_seconds,
):
    if length < 32 or batch_size < 1 or not 60 <= timeout_seconds <= 1800:
        raise ValueError("Invalid bounded parity case")
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    stem = f"length-{length}-batch-{batch_size}"
    report = evidence / f"{stem}-report.json"
    expected = {
        "conversion_sha256": digest(Path(target) / "conversion.json"),
        "sequence_length": length,
        "batch_size": batch_size,
    }
    if report.exists():
        saved = json.loads(report.read_text())
        validate_metrics(saved)
        if (
            saved["conversion_sha256"] != expected["conversion_sha256"]
            or saved["scope"]["sequence_length"] != length
            or saved["scope"]["batch_size"] != batch_size
        ):
            raise ValueError("Existing case belongs to another artifact/scope")
        # Source and current environments must be matched before reusing a
        # measurement. This worker deliberately refuses silent result reuse.
        raise ValueError(
            "Case already exists; merge its durable receipt or choose a fresh evidence directory"
        )
    inputs = evidence / f"{stem}-inputs.npy"
    left, right = evidence / f"{stem}-jax.npz", evidence / f"{stem}-torch.npz"
    deadline = time.monotonic() + timeout_seconds

    def execute(python, argv):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("Parity case deadline exhausted")
        subprocess.run(
            [str(python), "-m", "scripts.validate_yat_torch_parity", *argv],
            check=True,
            timeout=remaining,
        )

    status = evidence / f"{stem}-status.json"
    write_atomic(
        status, {"state": "running", **expected, "full_matrix_qualified": False}
    )
    try:
        execute(
            jax_python,
            [
                "inputs",
                str(Path(target) / "tokenizer.json"),
                str(inputs),
                "--sequence-length",
                str(length),
            ],
        )
        execute(
            jax_python,
            [
                "jax",
                str(source),
                str(inputs),
                str(left),
                "--batch-size",
                str(batch_size),
            ],
        )
        execute(
            torch_python,
            [
                "torch",
                str(target),
                str(inputs),
                str(right),
                "--batch-size",
                str(batch_size),
            ],
        )
        execute(jax_python, ["compare", str(left), str(right), str(report)])
        saved = json.loads(report.read_text())
        validate_metrics(saved)
        write_atomic(
            status,
            {
                "state": "case_passed",
                **expected,
                "report_sha256": digest(report),
                "full_matrix_qualified": False,
            },
        )
    except Exception as error:
        write_atomic(
            status,
            {
                "state": "failed",
                **expected,
                "error": f"{type(error).__name__}: {error}",
                "full_matrix_qualified": False,
            },
        )
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "jax-python", "torch-python", "evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    print(
        run_case(
            args.source,
            args.target,
            args.jax_python,
            args.torch_python,
            args.evidence,
            args.length,
            args.batch_size,
            args.timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
