"""Assemble complete, hash-matched YAT embedding MTEB receipts.

Run only after every TPU worker and WebLINX split has finished. The existing
merge helpers reject missing, duplicate, and mismatched model/dataset rows.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from scripts.evaluation_contract import aggregate as _aggregate, write_atomic as _write_atomic
from scripts.merge_yat_mteb_shards import merge as merge_shards
from scripts.merge_yat_mteb_splits import merge as merge_splits


def _weblinx_shard(root: Path) -> Path:
    receipts = sorted((root / "weblinx-splits").glob("weblinx-*.json"))
    split_path = root / "weblinx-complete.json"
    merged = json.loads(split_path.read_text()) if split_path.exists() else merge_splits(receipts, split_path)
    observed = [json.loads(path.read_text()) for path in receipts]
    if sorted(item["split"] for item in observed) != sorted(merged["full_splits"]):
        raise ValueError("WebLINX split receipt coverage changed")
    for item in observed:
        if (not item["result"] or item["identity"] != merged["identity"]
                or item["dataset_revision"] != merged["dataset_revision"]
                or item["result"]["scores"][item["split"]]
                   != merged["result"]["scores"][item["split"]]):
            raise ValueError(f"WebLINX split differs from merge: {item['split']}")
    reference = json.loads((root / "shards/chip-5-multilingual-extra.json").read_text())
    name = merged["task_name"]
    if merged["identity"] != reference["identity"]:
        raise ValueError("WebLINX model/protocol differs from suite inventory")
    if merged["dataset_revision"] != reference["inventory"][name]["dataset_revision"]:
        raise ValueError("WebLINX dataset revision differs from suite inventory")
    shard = {
        "identity": merged["identity"],
        "inventory": reference["inventory"],
        "selected_tasks": [name],
        "results": {name: {"category": reference["inventory"][name]["category"],
                           "result": merged["result"]}},
        "failures": {},
    }
    shard["aggregate"] = _aggregate(shard)
    path = root / "weblinx-shard.json"
    _write_atomic(path, shard)
    return path


def assemble(root: Path) -> dict:
    shards = root / "shards"
    full = root / "full-mteb-reports"
    weblinx = _weblinx_shard(root)
    english_inputs = sorted(shards.glob("*-english*.json"))
    english_inputs += sorted(full.glob("chip-*-english.json"))
    multilingual_inputs = sorted(shards.glob("*-multilingual*.json"))
    multilingual_inputs += sorted(full.glob("*multilingual*.json"))
    multilingual_inputs += [root / "miracl-complete.json", weblinx]
    for suite, inputs, expected in (
        ("english", english_inputs, 41),
        ("multilingual", multilingual_inputs, 131),
    ):
        observed = set()
        failed = set()
        for path in inputs:
            part = json.loads(path.read_text())
            observed.update(part["results"])
            failed.update(part["failures"])
        unresolved = failed - observed
        if unresolved:
            raise ValueError(f"{suite} has unresolved task failures: {sorted(unresolved)}")
        if len(observed) != expected:
            raise ValueError(f"{suite} incomplete: {len(observed)}/{expected}")
    output = {}
    for suite, inputs, expected in (
        ("english", english_inputs, 41),
        ("multilingual", multilingual_inputs, 131),
    ):
        receipt_path = root / f"{suite}-complete.json"
        receipt = (json.loads(receipt_path.read_text()) if receipt_path.exists()
                   else merge_shards(inputs, receipt_path))
        for path in inputs:
            part = json.loads(path.read_text())
            if part["identity"] != receipt["identity"] or part["inventory"] != receipt["inventory"]:
                raise ValueError(f"{suite} receipt identity/inventory differs from {path}")
            if any(receipt["results"].get(name) != item for name, item in part["results"].items()):
                raise ValueError(f"{suite} receipt differs from task results in {path}")
        if set(receipt["results"]) != set(receipt["inventory"]):
            raise ValueError(f"{suite} receipt does not cover its inventory")
        receipt["aggregate"] = _aggregate(receipt)
        if not receipt["aggregate"]["complete"] or len(receipt["results"]) != expected:
            raise ValueError(f"{suite} incomplete: {len(receipt['results'])}/{expected}")
        if receipt["failures"]:
            raise ValueError(f"{suite} has failed tasks: {sorted(receipt['failures'])}")
        output[suite] = receipt
    english_identity = output["english"]["identity"].copy()
    multilingual_identity = output["multilingual"]["identity"].copy()
    english_identity.pop("benchmark")
    multilingual_identity.pop("benchmark")
    if english_identity != multilingual_identity:
        raise ValueError("English/multilingual model or protocol differs")
    summaries = {}
    for suite, receipt in output.items():
        categories = receipt["aggregate"]["category_scores"]
        summaries[suite] = {
            "tasks": len(receipt["results"]),
            "category_scores": categories,
            "mean_category_score": receipt["aggregate"]["mean_category_score"],
        }
        if suite == "multilingual":
            common = {k: v for k, v in categories.items() if k != "InstructionReranking"}
            if len(common) != 8:
                raise ValueError(f"Expected 8 paper-reported multilingual categories, got {sorted(common)}")
            summaries[suite]["eight_common_category_mean"] = sum(common.values()) / len(common)
    _write_atomic(root / "full-mteb-summary.json", summaries)
    return summaries


def assemble_receipts(suites: dict[str, list[Path]], output: Path) -> dict:
    """Assemble arbitrary frozen suites without registry-dependent task counts.

    All input paths are explicit; the receipt inventories are the frozen manifest.
    Legacy inventories without exact split/subset rows must be regenerated.
    """
    if not suites:
        raise ValueError("No benchmark suites supplied")
    summaries = {}
    for suite, paths in suites.items():
        receipt_path = output.parent / f"{suite}-complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            with tempfile.TemporaryDirectory() as temporary:
                expected = merge_shards(paths, Path(temporary) / "expected.json")
            if any(receipt.get(key) != expected[key] for key in ("identity", "inventory", "selected_tasks", "results", "failures")):
                raise ValueError(f"Cached suite differs from current source receipts: {suite}")
        else:
            receipt = merge_shards(paths, receipt_path)
        receipt["aggregate"] = _aggregate(receipt)
        if not receipt["aggregate"]["complete"]:
            raise ValueError(f"Incomplete suite {suite}")
        summaries[suite] = {"identity": receipt["identity"], "inventory": receipt["inventory"],
                            "tasks": len(receipt["results"]), **receipt["aggregate"]}
    _write_atomic(output, summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suite-inputs", type=Path,
                        help="JSON mapping suite names to explicit receipt path arrays; generic frozen-manifest mode")
    args = parser.parse_args()
    if args.suite_inputs:
        manifest = json.loads(args.suite_inputs.read_text())
        if not isinstance(manifest, dict) or any(not isinstance(paths, list) or not paths for paths in manifest.values()):
            raise ValueError("Expected nonempty suite-to-receipt-path mapping")
        if any(not name or Path(name).name != name or name in {".", ".."} for name in manifest):
            raise ValueError("Unsafe suite name")
        result = assemble_receipts({name: [Path(path) for path in paths] for name, paths in manifest.items()},
                                   args.root / "full-mteb-summary.json")
    else:
        result = assemble(args.root)  # Explicit historical 41/131-task layout only.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
