"""Audit both full bitext exports and compare the frozen matched model pair."""

import argparse
import json
from pathlib import Path

from flaxchat.encoder_data import file_hash
from scripts.evaluate_encoder_bitext import evaluate_campaign
from scripts.export_encoder_bitext_campaign import verify_plan


def compare(plan_path, baseline, candidate):
    frozen, identity, entries, directories, snapshot = verify_plan(plan_path)
    reports = []
    common = None
    inputs = {}
    for role, root in (("baseline", Path(baseline)), ("candidate", Path(candidate))):
        path = root / "report.json"
        before = file_hash(path)
        report = json.loads(path.read_text())
        if (
            report.get("complete") is not True
            or report.get("role") != role
            or report.get("plan_sha256") != identity
            or report.get("snapshot_sha256") != snapshot
            or report.get("candidate") != frozen["candidate"]
        ):
            raise ValueError("Incomplete or wrong model campaign")
        recorded = report["manifests"]
        if [r["subset"] for r in recorded] != [e["subset"] for e in entries]:
            raise ValueError("Incomplete manifest inventory")
        for row in recorded:
            manifest_path = root / "embeddings" / row["subset"] / "manifest.json"
            if file_hash(manifest_path) != row["sha256"]:
                raise ValueError("Changed exported manifest")
            manifest = json.loads(manifest_path.read_text())
            directory = directories[row["subset"]]
            expected_inputs = {
                k: v
                for k, v in frozen["input_sha256"].items()
                if Path(k).parent == directory or Path(k).parent.parent == directory
            }
            if (
                manifest["encoder_config"] != report["encoder_config"]
                or manifest["pooling"] != frozen["pooling"]
                or manifest["input_sha256"] != expected_inputs
            ):
                raise ValueError("Unmatched export inputs or configuration")
            origin = manifest["origin"]
            if role == "baseline":
                if (
                    manifest["checkpoint"] is not None
                    or manifest["checkpoint_step"] is not None
                    or origin.get("released_weights", {}).get("model.safetensors")
                    != frozen["baseline"]["weights_sha256"]
                ):
                    raise ValueError("Baseline must use frozen released weights")
            elif (
                manifest["checkpoint"] != frozen["candidate"]["checkpoint"]
                or manifest["checkpoint_step"] != frozen["candidate"]["step"]
                or origin != frozen["candidate"]
            ):
                raise ValueError("Candidate checkpoint origin mismatch")
            match = dict(
                encoder=manifest["encoder_config"],
                pooling=manifest["pooling"],
                tokenizer=manifest["tokenizer_sha256"],
                source=manifest["export_source_sha256"],
            )
            if common is None:
                common = match
            elif match != common:
                raise ValueError("Unmatched inference provenance")
        scores = evaluate_campaign(root / "embeddings", frozen["inventory_path"])
        if (
            scores != report["scores"]
            or scores["subset_count"] != frozen["subsets"]
            or scores["pair_count"] != frozen["pairs"]
        ):
            raise ValueError("Recorded scores disagree with full recomputation")
        if file_hash(path) != before:
            raise ValueError("Campaign report changed during audit")
        inputs[role] = before
        reports.append(report)
    left, right = reports
    for key in (
        "source_sha256",
        "encoder_config",
        "batch_size",
        "jax_version",
        "backend",
    ):
        if not left.get(key) or left[key] != right.get(key):
            raise ValueError("Unmatched campaign runtime or source")
    per_subset = [
        dict(
            subset=a["subset"],
            pairs=a["pairs"],
            baseline=a["metrics"],
            candidate=b["metrics"],
            delta={k: b["metrics"][k] - v for k, v in a["metrics"].items()},
        )
        for a, b in zip(
            left["scores"]["per_subset"], right["scores"]["per_subset"], strict=True
        )
    ]
    return dict(
        complete=True,
        plan_sha256=identity,
        input_report_sha256=inputs,
        subsets=frozen["subsets"],
        pairs=frozen["pairs"],
        per_subset=per_subset,
        baseline=left["scores"]["subset_macro"],
        candidate=right["scores"]["subset_macro"],
        delta={
            k: right["scores"]["subset_macro"][k] - v
            for k, v in left["scores"]["subset_macro"].items()
        },
        official_mteb_parity=False,
        production_quality_qualified=False,
        limitations=frozen.get("limitations", []),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.plan, args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_subset"}))


if __name__ == "__main__":
    main()
