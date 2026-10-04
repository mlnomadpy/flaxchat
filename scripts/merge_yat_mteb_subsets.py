"""Merge disjoint language-subset TPU receipts into one complete MTEB task."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from scripts.evaluation_contract import aggregate as _aggregate, write_atomic as _write_atomic, validate_result


def merge(inputs: list[Path], inventory_receipt: Path, output: Path) -> dict:
    if not inputs or output.exists():
        raise ValueError("Require subset receipts and a fresh output path")
    inventory_source = json.loads(inventory_receipt.read_text())
    merged = None
    scores = {}
    for path in inputs:
        part = json.loads(path.read_text())
        if part.get("result") is None:
            raise ValueError(f"Incomplete subset receipt: {path}")
        if merged is None:
            merged = {"identity": part["identity"], "task_name": part["task_name"],
                      "dataset_revision": part["dataset_revision"],
                      "full_subsets": part["full_subsets"],
                      "result": copy.deepcopy(part["result"]), "sources": []}
            merged["result"]["scores"] = {}
        for field in ("identity", "task_name", "dataset_revision", "full_subsets"):
            if part[field] != merged[field]:
                raise ValueError(f"Mismatched {field} in {path}")
        validate_result(part["task_name"], {"category": part["task_inventory"]["category"], "result": part["result"]}, part["task_inventory"])
        rows = [(split, row) for split, split_rows in part["result"]["scores"].items()
                for row in split_rows]
        if set(row["hf_subset"] for _, row in rows) != set(part["subsets"]):
            raise ValueError(f"Subset row mismatch in {path}")
        for split, row in rows:
            subset = row["hf_subset"]
            if (split, subset) in scores:
                raise ValueError(f"Duplicate subset {subset}")
            scores[(split, subset)] = row
            merged["result"]["scores"].setdefault(split, []).append(row)
        merged["sources"].append(str(path))
    if {subset for split, subset in scores} != set(merged["full_subsets"]):
        raise ValueError(f"Incomplete subset coverage: got {len(scores)} of {len(merged['full_subsets'])}")
    if merged["identity"] != inventory_source["identity"]:
        raise ValueError("Model/benchmark identity differs from inventory")
    name = merged["task_name"]
    inventory = inventory_source["inventory"]
    if merged["dataset_revision"] != inventory[name]["dataset_revision"]:
        raise ValueError("Dataset revision differs from inventory")
    for rows in merged["result"]["scores"].values():
        rows.sort(key=lambda row: row["hf_subset"])
    receipt = {"identity": merged["identity"], "inventory": inventory,
               "selected_tasks": [name],
               "results": {name: {"category": inventory[name]["category"],
                                   "result": merged["result"]}},
               "failures": {}, "sources": merged["sources"],
               "aggregate": {}}
    receipt["aggregate"] = _aggregate(receipt)
    _write_atomic(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--inventory-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = merge(args.input, args.inventory_receipt, args.output)
    print(json.dumps({"task": next(iter(receipt["results"])),
                      "subsets": sum(len(rows) for rows in next(iter(receipt["results"].values()))["result"]["scores"].values())}))


if __name__ == "__main__":
    main()
