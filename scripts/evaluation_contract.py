"""Model-free, versioned MTEB receipt validation and aggregation."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

POLICY = "finite-main-score/row-macro/task-macro/category-macro-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_inventory_spec(spec: dict) -> None:
    if not isinstance(spec.get("category"), str) or not spec["category"]:
        raise ValueError("Task category required")
    if not isinstance(spec.get("dataset_revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", spec["dataset_revision"]):
        raise ValueError("Task dataset revision must be an immutable HF commit")
    for key in ("eval_splits", "hf_subsets"):
        values = spec.get(key)
        if (not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"Invalid task {key}")
    expected = spec.get("expected_rows")
    product = {(split, subset) for split in spec["eval_splits"] for subset in spec["hf_subsets"]}
    if (not isinstance(expected, list) or any(not isinstance(row, list) or len(row) != 2 for row in expected)
            or len(expected) != len(product) or {tuple(row) for row in expected} != product):
        raise ValueError("Frozen rows must exactly match declared splits/subsets")
    if spec.get("aggregation_policy") != POLICY or spec.get("metric") != "main_score":
        raise ValueError("Unsupported metric/aggregation policy")


def freeze_task_revision(task) -> None:
    """Resolve the registry's revision before dataset loading; no model work."""
    dataset = task.metadata.dataset
    revision = dataset.get("revision")
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        return
    from huggingface_hub import HfApi
    if not isinstance(dataset.get("path"), str):
        raise ValueError("Task requires a resolvable dataset repository path")
    resolved = HfApi().dataset_info(dataset["path"], revision=revision).sha
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Could not resolve an immutable dataset revision")
    dataset["revision"] = resolved


def task_inventory(task) -> dict:
    splits, subsets = list(task.eval_splits), list(task.hf_subsets)
    if not splits or not subsets or len(splits) != len(set(splits)) or len(subsets) != len(set(subsets)):
        raise ValueError("Task must declare unique nonempty splits/subsets")
    spec = {"category": task.metadata.type,
            "dataset_revision": task.metadata.dataset.get("revision"),
            "eval_splits": splits, "hf_subsets": subsets,
            "expected_rows": [[split, subset] for split in splits for subset in subsets],
            "metric": "main_score", "aggregation_policy": POLICY}
    validate_inventory_spec(spec)
    return spec


def validate_result(name: str, item: dict, spec: dict) -> list[float]:
    validate_inventory_spec(spec)
    raw = item["result"]
    if (item["category"] != spec["category"] or raw.get("task_name") != name
            or raw.get("dataset_revision") != spec["dataset_revision"]):
        raise ValueError(f"Task/category/revision mismatch: {name}")
    expected = spec.get("expected_rows")
    if not expected or len(expected) != len({tuple(row) for row in expected}):
        raise ValueError(f"Missing or duplicated frozen row inventory: {name}")
    if spec.get("aggregation_policy") != POLICY:
        raise ValueError(f"Unsupported aggregation policy: {name}")
    scores = raw.get("scores", {})
    if set(scores) != set(spec["eval_splits"]):
        raise ValueError(f"Split mismatch: {name}")
    observed, values = [], []
    for split, rows in scores.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty score rows: {name}/{split}")
        for row in rows:
            observed.append((split, row.get("hf_subset")))
            value = row.get("main_score")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Nonfinite or absent score: {name}/{split}")
            values.append(value)
    if len(observed) != len(set(observed)) or set(observed) != {tuple(row) for row in expected}:
        raise ValueError(f"Subset row coverage mismatch: {name}")
    return values


def aggregate(record: dict) -> dict:
    inventory, results = record["inventory"], record["results"]
    for spec in inventory.values():
        validate_inventory_spec(spec)
    selected = record.get("selected_tasks", sorted(inventory))
    if len(selected) != len(set(selected)) or not set(selected) <= set(inventory):
        raise ValueError("Unknown or duplicate task selection")
    if not set(results) <= set(selected) or not set(record["failures"]) <= set(selected):
        raise ValueError("Unselected result or failure")
    grouped = defaultdict(list)
    task_scores = {}
    for name, item in results.items():
        values = validate_result(name, item, inventory[name])
        task_scores[name] = sum(values) / len(values)
        grouped[item["category"]].append(task_scores[name])
    categories = {name: sum(values) / len(values) for name, values in grouped.items()}
    return {"aggregation_policy": POLICY, "task_scores": task_scores,
            "category_scores": categories,
            "mean_category_score": sum(categories.values()) / len(categories) if categories else None,
            "complete": bool(inventory) and set(selected) == set(inventory)
                        and set(results) == set(inventory) and not record["failures"],
            "selected_complete": bool(selected) and set(results) == set(selected) and not record["failures"]}


def provenance(source_paths: list[Path], *, devices: list[dict], batch_size: int) -> dict:
    packages = {}
    # Include transitive packages: changes invalidate exact resume.
    for distribution in importlib.metadata.distributions():
        packages[distribution.metadata["Name"]] = distribution.version
    sources = {}
    for path in source_paths:
        module_name = ".".join(path.with_suffix("").parts) if not path.is_absolute() else None
        module = sys.modules.get(module_name) if module_name else None
        actual_path = Path(module.__file__) if module and getattr(module, "__file__", None) else path
        sources[str(path)] = digest(actual_path)
    numeric_environment = {key: value for key, value in sorted(os.environ.items())
                           if key.startswith(("JAX_", "XLA_", "LIBTPU_", "PJRT_", "TORCH_", "PT_XLA_"))
                           or key in {"PJRT_DEVICE", "TPU_MEGACORE", "TPU_USE_MEGACORE"}}
    jax_module = sys.modules.get("jax")
    effective_jax_config = getattr(getattr(jax_module, "config", None), "values", {})
    effective_jax_config = {key: value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
                            for key, value in sorted(effective_jax_config.items())}
    torch_module = sys.modules.get("torch")
    effective_torch_config = {} if torch_module is None else {
        "default_dtype": str(torch_module.get_default_dtype()),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
    }
    deployment = {}
    run_root = os.environ.get("FLAXCHAT_RUN_ROOT")
    if run_root:
        during_setup = os.environ.get("FLAXCHAT_SETUP_QUALIFICATION") == "1"
        runtime_name = "runtime-setup-receipt.json" if during_setup else "runtime-receipt.json"
        for name in ("manifest-identity.txt", runtime_name, "hardware-receipt.json"):
            path = Path(run_root) / name
            if not path.is_file():
                raise ValueError(f"Missing deployment runtime evidence: {path}")
            deployment[name] = digest(path)
        if during_setup:
            runtime = json.loads((Path(run_root) / runtime_name).read_text())
            if (runtime.get("schema_version") != 2 or runtime.get("runtime_verified") is not True
                    or runtime.get("setup_checks_executed") is not False
                    or runtime.get("physical_acceptance") is not False
                    or runtime.get("qualification") is not None
                    or runtime.get("manifest_sha256") != (Path(run_root) / "manifest-identity.txt").read_text().strip()
                    or runtime.get("manifest_sha256") != os.environ.get("FLAXCHAT_RUN_MANIFEST_SHA256")
                    or runtime.get("runtime_lock_sha256") != os.environ.get("FLAXCHAT_RUNTIME_LOCK_SHA256")):
                raise ValueError("Invalid provisional deployment runtime evidence")
    return {"source_sha256": sources,
            "interpreter": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                            "build": list(platform.python_build()), "executable_sha256": digest(Path(sys.executable))},
            "numeric_environment": numeric_environment, "effective_jax_config": effective_jax_config,
            "effective_torch_config": effective_torch_config,
            "deployment_receipts_sha256": deployment,
            "runtime_packages": dict(sorted(packages.items())),
            "devices": devices, "inference_batch_size": batch_size,
            "aggregation_policy": POLICY}
