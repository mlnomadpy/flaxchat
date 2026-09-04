"""Enforce risk-based per-module coverage floors from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FLOORS = {
    "flaxchat/chat.py": 55.0,
    "flaxchat/checkpoint.py": 75.0,
    "flaxchat/common.py": 65.0,
    "flaxchat/engine.py": 65.0,
    "flaxchat/eval.py": 60.0,
    # Spawned child lines are not attributed to the parent coverage process;
    # adversarial behavior is asserted directly in test_execution.py.
    "flaxchat/execution.py": 45.0,
    "flaxchat/sft.py": 55.0,
    "flaxchat/tokenizer.py": 55.0,
}


def module_coverage_deltas(
    report: dict, floors: dict[str, float] = FLOORS
) -> list[tuple[str, float, float, float]]:
    """Return module, actual, floor, and signed headroom for reported modules."""
    files = report.get("files", {})
    deltas = []
    for module, floor in floors.items():
        summary = files.get(module, {}).get("summary")
        if summary and "percent_covered" in summary:
            actual = float(summary["percent_covered"])
            deltas.append((module, actual, floor, actual - floor))
    return deltas


def check_coverage(report: dict, floors: dict[str, float] = FLOORS) -> list[str]:
    failures = []
    files = report.get("files", {})
    for module, floor in floors.items():
        summary = files.get(module, {}).get("summary")
        if not summary or "percent_covered" not in summary:
            failures.append(f"{module}: missing from coverage report")
            continue
        actual = float(summary["percent_covered"])
        if actual < floor:
            failures.append(f"{module}: {actual:.2f}% is below {floor:.2f}%")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=Path("coverage.json"))
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    for module, actual, floor, delta in module_coverage_deltas(report):
        print(
            f"{module}: {actual:.2f}% (floor {floor:.2f}%, "
            f"delta {delta:+.2f}pp)"
        )
    failures = check_coverage(report)
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")
    print("Risk-based module coverage floors passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
