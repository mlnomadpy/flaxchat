from pathlib import Path
import re

import yaml


ROOT = Path(__file__).parents[1]


def test_pinned_baseline_plans_match_canonical_protocol():
    protocol = yaml.safe_load((ROOT / "benchmarks/protocol.yaml").read_text())
    expected = {
        "target_model_parameters": protocol["model"]["target_trainable_parameters"],
        "parameter_tolerance_fraction": protocol["model"]["parameter_tolerance_fraction"],
        "sequence_length": protocol["model"]["sequence_length"],
        "global_batch_size": protocol["training"]["global_batch_size"],
        "precision": protocol["training"]["precision"],
        "optimizer": protocol["training"]["optimizer"],
        "seed": protocol["training"]["seed"],
        "dataset": protocol["data"]["dataset"],
        "dataset_revision": protocol["data"]["revision"],
        "hardware": protocol["measurement"]["hardware"],
        "device_count": protocol["measurement"]["device_count"],
        "warmup_steps": protocol["measurement"]["warmup_steps"],
        "measured_steps": protocol["measurement"]["measured_steps"],
    }

    paths = sorted((ROOT / "benchmarks/baselines").glob("*.yaml"))
    assert len(paths) == 3
    for path in paths:
        plan = yaml.safe_load(path.read_text())
        assert {key: plan[key] for key in expected} == expected
        assert plan["parameter_counting"] == "trainable_parameters"
        assert plan["status"] == "pending_matched_run"
        assert re.fullmatch(r"[0-9a-f]{40}", plan["source_revision"])
