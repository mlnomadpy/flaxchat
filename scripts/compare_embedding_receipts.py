"""Compare two content-authenticated complete embedding evaluation receipts.

This is model-free receipt validation, not a benchmark runner or paper-score
importer. A frozen plan binds complete inventory and each model's own protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.evaluation_contract import (
    POLICY,
    aggregate,
    validate_inventory_spec,
    write_atomic,
)

FORMAT = "matched-embedding-comparison-plan-v1"
ROLES = ("baseline", "candidate")
MODEL_FIELDS = ("model", "files_sha256", "prompts", "pooling", "normalization")
COMMON_FIELDS = (
    "benchmark",
    "mteb_version",
    "sequence_length",
    "padding_policy",
    "truncation_policy",
    "score_scale",
)


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def read_authenticated(path, expected, *, limit=32 * 1024 * 1024):
    path = Path(path)
    if (
        not _sha(expected)
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > limit
    ):
        raise ValueError(
            "Require bounded regular JSON and independently pinned SHA-256"
        )
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(
            "Comparison input differs from independently pinned content hash"
        )

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate comparison JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite comparison JSON value")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Comparison JSON object required")
    return value


def validate_plan(plan):
    if plan.get("format") != FORMAT or plan.get("aggregation_policy") != POLICY:
        raise ValueError(
            "Frozen matched-comparison plan and aggregation policy required"
        )
    inventory = plan.get("inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Independent full task inventory required")
    for name, spec in inventory.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Named comparison tasks required")
        validate_inventory_spec(spec)
    common = plan.get("common")
    if not isinstance(common, dict) or set(common) != set(COMMON_FIELDS):
        raise ValueError(
            "Explicit common benchmark/length/truncation/scale protocol required"
        )
    for key in COMMON_FIELDS:
        if key == "sequence_length":
            if type(common[key]) is not int or common[key] < 1:
                raise ValueError("Positive shared sequence length required")
        elif not isinstance(common[key], str) or not common[key]:
            raise ValueError("Nonempty common comparison protocol required")
    if common["score_scale"] != "native-mteb-main-score/no-rescaling":
        raise ValueError("Explicit known score scale required; no implicit rescaling")
    models = plan.get("models")
    if not isinstance(models, dict) or set(models) != set(ROLES):
        raise ValueError("Exactly two declared comparison models required")
    for role in ROLES:
        model = models[role]
        if not isinstance(model, dict) or not _sha(model.get("receipt_sha256")):
            raise ValueError("Each model requires an independently pinned receipt hash")
        for key in MODEL_FIELDS:
            value = model.get(key)
            if key == "files_sha256":
                if (
                    not isinstance(value, dict)
                    or not value
                    or not all(
                        isinstance(name, str) and name and _sha(digest)
                        for name, digest in value.items()
                    )
                ):
                    raise ValueError(
                        "Each model requires complete declared artifact hashes"
                    )
                if not any(
                    ("model" in Path(name).name or "weight" in Path(name).name)
                    and Path(name).suffix
                    in {".safetensors", ".bin", ".msgpack", ".npz"}
                    for name in value
                ):
                    raise ValueError("Each model requires a weight artifact hash")
                if "config.json" not in value or not any(
                    Path(name).name.startswith(
                        ("tokenizer.", "vocab.", "sentencepiece.")
                    )
                    for name in value
                ):
                    raise ValueError(
                        "Each model requires config and tokenizer artifact hashes"
                    )
            elif key == "prompts":
                if not isinstance(value, (str, dict)) or not value:
                    raise ValueError("Explicit model-specific prompt policy required")
            elif not isinstance(value, str) or not value:
                raise ValueError(
                    "Explicit model identity/pooling/normalization required"
                )
    scoring = plan.get("scoring_source_sha256")
    if (
        not isinstance(scoring, dict)
        or not scoring
        or not all(
            isinstance(name, str) and name and _sha(value)
            for name, value in scoring.items()
        )
    ):
        raise ValueError("Shared scoring implementation hashes required")
    if "scripts/evaluation_contract.py" not in scoring:
        raise ValueError("Shared aggregation implementation hash required")


def validate_receipt(record, plan, role):
    if record.get("inventory") != plan["inventory"]:
        raise ValueError(
            "Receipt inventory differs from independently frozen complete inventory"
        )
    calculated = aggregate(record)
    if not calculated["complete"] or not calculated["selected_complete"]:
        raise ValueError(
            "Incomplete/sharded/failed receipts cannot produce matched scores"
        )
    if record.get("aggregate") != calculated:
        raise ValueError("Stored aggregate differs from recomputed complete scores")
    identity = record.get("identity", {})
    for key in COMMON_FIELDS:
        if identity.get(key) != plan["common"][key]:
            raise ValueError(f"Mismatched or missing shared protocol: {key}")
    for key in MODEL_FIELDS:
        if identity.get(key) != plan["models"][role][key]:
            raise ValueError(f"Undeclared model-specific protocol: {role}/{key}")
    runtime = identity.get("protocol")
    if not isinstance(runtime, dict):
        raise ValueError("Complete runtime provenance required")
    for key in (
        "source_sha256",
        "runtime_packages",
        "interpreter",
        "numeric_environment",
        "effective_jax_config",
        "effective_torch_config",
        "deployment_receipts_sha256",
    ):
        if not isinstance(runtime.get(key), dict):
            raise ValueError(f"Missing runtime provenance: {key}")
    if runtime["runtime_packages"].get("mteb") != plan["common"]["mteb_version"]:
        raise ValueError("Scoring package differs from frozen MTEB version")
    interpreter = runtime["interpreter"]
    if (
        not interpreter.get("version")
        or not interpreter.get("implementation")
        or not isinstance(interpreter.get("build"), list)
        or not interpreter["build"]
        or not _sha(interpreter.get("executable_sha256"))
    ):
        raise ValueError("Incomplete interpreter identity")
    if not isinstance(runtime.get("devices"), list) or not runtime["devices"]:
        raise ValueError("Actual runtime device inventory required")
    if identity.get("backend") not in ("physical TPU", "tpu", "physical GPU", "cuda"):
        raise ValueError("Physical accelerator receipt required")
    expected_platform = (
        "tpu" if identity["backend"] in ("physical TPU", "tpu") else "gpu"
    )
    if any(
        not isinstance(device, dict) or device.get("platform") != expected_platform
        for device in runtime["devices"]
    ):
        raise ValueError("Backend differs from recorded accelerator devices")
    for name, expected in plan["scoring_source_sha256"].items():
        matches = [
            value
            for path, value in runtime["source_sha256"].items()
            if path == name or path.endswith("/" + name)
        ]
        if matches != [expected]:
            raise ValueError(f"Mismatched or ambiguous scoring source: {name}")
    return calculated


def compare(plan_path, baseline_path, candidate_path, *, plan_sha256):
    plan = read_authenticated(plan_path, plan_sha256, limit=1024 * 1024)
    validate_plan(plan)
    paths = {"baseline": Path(baseline_path), "candidate": Path(candidate_path)}
    records, aggregates = {}, {}
    for role in ROLES:
        records[role] = read_authenticated(
            paths[role], plan["models"][role]["receipt_sha256"]
        )
        aggregates[role] = validate_receipt(records[role], plan, role)
    rows = []
    for name, spec in sorted(plan["inventory"].items()):
        scores = {}
        for role in ROLES:
            result = records[role]["results"][name]["result"]
            scores[role] = {
                (split, row["hf_subset"]): row["main_score"]
                for split, values in result["scores"].items()
                for row in values
            }
        for split, subset in sorted(map(tuple, spec["expected_rows"])):
            before, after = (
                scores["baseline"][split, subset],
                scores["candidate"][split, subset],
            )
            rows.append(
                dict(
                    task=name,
                    category=spec["category"],
                    split=split,
                    subset=subset,
                    dataset_revision=spec["dataset_revision"],
                    baseline=before,
                    candidate=after,
                    delta=after - before,
                )
            )
    categories = {
        name: dict(
            baseline=aggregates["baseline"]["category_scores"][name],
            candidate=aggregates["candidate"]["category_scores"][name],
            delta=aggregates["candidate"]["category_scores"][name]
            - aggregates["baseline"]["category_scores"][name],
        )
        for name in sorted(aggregates["baseline"]["category_scores"])
    }
    # Re-read all inputs at the original pins before declaring the comparison.
    read_authenticated(plan_path, plan_sha256, limit=1024 * 1024)
    for role in ROLES:
        read_authenticated(paths[role], plan["models"][role]["receipt_sha256"])
    return dict(
        format="matched-embedding-comparison-v1",
        matched=True,
        evidence_scope="content-authenticated recorded measurements; no model execution",
        architecture_effect_isolated=False,
        paper_scores_included=False,
        plan_sha256=plan_sha256,
        common=plan["common"],
        aggregation_policy=POLICY,
        models={role: records[role]["identity"] for role in ROLES},
        model_specific_differences={
            key: {role: records[role]["identity"].get(key) for role in ROLES}
            for key in set(records["baseline"]["identity"])
            | set(records["candidate"]["identity"])
            if records["baseline"]["identity"].get(key)
            != records["candidate"]["identity"].get(key)
        },
        receipt_sha256={role: plan["models"][role]["receipt_sha256"] for role in ROLES},
        per_row=rows,
        per_category=categories,
        aggregates=aggregates,
        mean_category_delta=aggregates["candidate"]["mean_category_score"]
        - aggregates["baseline"]["mean_category_score"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() in {
        path.resolve() for path in (args.plan, args.baseline, args.candidate)
    }:
        raise ValueError(
            "Fresh comparison output distinct from input evidence required"
        )
    report = compare(
        args.plan, args.baseline, args.candidate, plan_sha256=args.plan_sha256
    )
    write_atomic(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
