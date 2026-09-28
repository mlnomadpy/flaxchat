"""Audit both full bitext exports and compare the frozen matched model pair."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from flaxchat.encoder import EncoderConfig
from flaxchat.encoder_data import file_hash
from scripts.evaluate_encoder_bitext import evaluate_campaign
from scripts.export_encoder_bitext_campaign import verify_plan


def compare(plan_path, baseline, candidate):
    frozen, identity, entries, directories, snapshot = verify_plan(plan_path)
    reports = []
    common = None
    inputs = {}
    role_configs = {}
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
            if role == "baseline" and "encoder_config" in frozen["baseline"] and (
                manifest["encoder_config"] != frozen["baseline"]["encoder_config"]
            ):
                raise ValueError("Baseline encoder differs from frozen plan")
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
                or origin != {key: frozen["candidate"][key] for key in ("checkpoint", "step")}
            ):
                raise ValueError("Candidate checkpoint origin mismatch")
            match = dict(
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
        role_configs[role] = report["encoder_config"]
        reports.append(report)
    left, right = reports
    if frozen.get("require_physical_tpu") and (
        left.get("backend") != "tpu" or right.get("backend") != "tpu"
    ):
        raise ValueError("Paired benchmark reports require physical TPU inference")
    if "encoder_config" not in frozen["baseline"]:
        if role_configs["baseline"] != role_configs["candidate"]:
            raise ValueError("Legacy matched campaign requires identical encoders")
    else:
        candidate_config = EncoderConfig(**role_configs["candidate"])
        baseline_config = EncoderConfig(**role_configs["baseline"])
        released = EncoderConfig.from_hf(
            json.loads((Path(frozen["baseline"]["snapshot"]) / "config.json").read_text()),
            **{key: getattr(candidate_config, key) for key in (
                "compute_dtype", "residual_dtype", "use_remat", "attention_backend",
                "loss_chunk_size", "mlm_projection", "mlm_loss_backend", "mlm_vocab_tile",
            )},
        )
        if baseline_config != released or asdict(baseline_config) != frozen["baseline"]["encoder_config"]:
            raise ValueError("Baseline must use released mmBERT architecture under matched runtime")
        for field in (
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "max_position_embeddings", "pad_token_id", "mask_token_id",
            "compute_dtype", "residual_dtype", "attention_backend", "use_remat",
        ):
            if getattr(candidate_config, field) != getattr(baseline_config, field):
                raise ValueError(f"Unmatched inference field: {field}")
    for key in (
        "source_sha256",
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
