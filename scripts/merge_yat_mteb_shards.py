"""Merge per-chip YAT MTEB receipts, rejecting mismatched or duplicate results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation_contract import aggregate as _aggregate, write_atomic as _write_atomic


def merge(inputs: list[Path], output: Path) -> dict:
    if not inputs or output.exists():
        raise ValueError("Require input receipts and a fresh output path")
    merged = None
    seen_selections = set()
    for path in inputs:
        receipt = json.loads(path.read_text())
        _aggregate(receipt)
        if merged is None:
            merged = {"identity": receipt["identity"],
                      "inventory": receipt["inventory"],
                      "selected_tasks": sorted(receipt["inventory"]),
                      "results": {}, "failures": {}, "sources": []}
        elif (receipt["identity"] != merged["identity"]
              or receipt["inventory"] != merged["inventory"]):
            raise ValueError(f"Mismatched benchmark/model in {path}")
        selection = frozenset(receipt.get("selected_tasks", receipt["inventory"]))
        if not selection <= receipt["inventory"].keys():
            raise ValueError(f"Unknown selected task in {path}")
        seen_selections.update(selection)
        for name, item in receipt["results"].items():
            if name not in selection:
                raise ValueError(f"Unselected result in {path}: {name}")
            if name in merged["results"] and merged["results"][name] != item:
                raise ValueError(f"Conflicting result for {name}")
            merged["results"][name] = item
            merged["failures"].pop(name, None)
        for name, item in receipt["failures"].items():
            if name not in selection:
                raise ValueError(f"Unselected failure in {path}: {name}")
            if name not in merged["results"]:
                merged["failures"][name] = item
        merged["sources"].append(str(path))
    merged["assigned_tasks"] = sorted(seen_selections)
    merged["aggregate"] = _aggregate(merged)
    _write_atomic(output, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.input, args.output)
    print(json.dumps({"complete": result["aggregate"]["complete"],
                      "done": len(result["results"]),
                      "total": len(result["inventory"]),
                      "failures": sorted(result["failures"])}, sort_keys=True))


if __name__ == "__main__":
    main()
