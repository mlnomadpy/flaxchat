"""Merge disjoint split-level MTEB receipts after strict coverage checks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from scripts.evaluation_contract import write_atomic as _write_atomic, validate_result


def merge(paths: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not paths:
        raise ValueError("No split receipts")
    receipts = [json.loads(path.read_text()) for path in paths]
    first = receipts[0]
    common = ("identity", "task_name", "dataset_revision", "full_splits", "full_task_inventory")
    for item in receipts:
        if any(item.get(key) != first.get(key) for key in common):
            raise ValueError("Model, task, revision, or split inventory mismatch")
        if item.get("result"):
            validate_result(item["task_name"], {"category": item["task_inventory"]["category"], "result": item["result"]}, item["task_inventory"])
        if not item.get("result"):
            raise ValueError(f"Incomplete split receipt: {item.get('split')}")
    full = first["full_splits"]
    observed = [item["split"] for item in receipts]
    if len(full) != len(set(full)) or sorted(observed) != sorted(full):
        raise ValueError(f"Split coverage mismatch: expected {full}, got {observed}")
    template = copy.deepcopy(first["result"])
    template["scores"] = {}
    template["evaluation_phases"] = []
    template["evaluation_time"] = 0.0
    for item in receipts:
        row = item["result"]
        if row["task_name"] != first["task_name"] or row["dataset_revision"] != first["dataset_revision"]:
            raise ValueError("Result identity mismatch")
        if set(row["scores"]) != {item["split"]}:
            raise ValueError("Result split mismatch")
        template["scores"].update(row["scores"])
        template["evaluation_phases"].extend(row.get("evaluation_phases", []))
        template["evaluation_time"] += row.get("evaluation_time", 0.0)
    template["scores"] = {split: template["scores"][split] for split in full}
    merged = {
        "identity": first["identity"],
        "task_name": first["task_name"],
        "dataset_revision": first["dataset_revision"],
        "full_splits": full,
        "task_inventory": first["full_task_inventory"],
        "split_receipts": [str(path) for path in paths],
        "result": template,
        "encoded_texts": sum(item.get("encoded_texts", 0) for item in receipts),
        "truncated_texts": sum(item.get("truncated_texts", 0) for item in receipts),
    }
    validate_result(first["task_name"], {"category": merged["task_inventory"]["category"], "result": template}, merged["task_inventory"])
    _write_atomic(output, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    result = merge(args.receipts, args.output)
    print(json.dumps({"task": result["task_name"],
                      "splits": list(result["result"]["scores"])}))


if __name__ == "__main__":
    main()
