"""Adversarial comparison evidence validation; no backend/model imports."""

import copy
import hashlib
import json

import pytest
from scripts import compare_embedding_receipts as comparison
from scripts.evaluation_contract import POLICY, aggregate


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture(tmp_path):
    inventory = {
        "Retrieval": {
            "category": "Retrieval",
            "dataset_revision": "a" * 40,
            "eval_splits": ["test"],
            "hf_subsets": ["en", "ar"],
            "expected_rows": [["test", "en"], ["test", "ar"]],
            "metric": "main_score",
            "aggregation_policy": POLICY,
        }
    }
    common = {
        "benchmark": "Frozen complete fixture suite",
        "mteb_version": "2.21.8",
        "sequence_length": 512,
        "padding_policy": "power-of-two-buckets",
        "truncation_policy": "right-truncate-token-ids-to-sequence-length-before-padding",
        "score_scale": "native-mteb-main-score/no-rescaling",
    }
    plan = {
        "format": comparison.FORMAT,
        "inventory": inventory,
        "common": common,
        "aggregation_policy": POLICY,
        "models": {},
        "scoring_source_sha256": {"scripts/evaluation_contract.py": "b" * 64},
    }
    records = {}
    for index, role in enumerate(comparison.ROLES):
        protocol = {
            "source_sha256": {
                "scripts/evaluation_contract.py": "b" * 64,
                "model_loader.py": str(index) * 64,
            },
            "runtime_packages": {"mteb": "2.21.8"},
            "interpreter": {
                "version": "3.12",
                "implementation": "CPython",
                "build": ["pinned"],
                "executable_sha256": "c" * 64,
            },
            "numeric_environment": {},
            "effective_jax_config": {},
            "effective_torch_config": {},
            "deployment_receipts_sha256": {},
            "devices": [{"platform": "tpu"}],
        }
        model = {
            "model": role,
            "files_sha256": {
                name: str(index) * 64
                for name in ("model.safetensors", "config.json", "tokenizer.json")
            },
            "prompts": "query: " if index else "none",
            "pooling": "mean" if index else "last-token",
            "normalization": "L2 FP32",
        }
        record = {
            "identity": {
                **common,
                **model,
                "protocol": protocol,
                "backend": "physical TPU",
            },
            "inventory": copy.deepcopy(inventory),
            "selected_tasks": ["Retrieval"],
            "failures": {},
            "results": {
                "Retrieval": {
                    "category": "Retrieval",
                    "result": {
                        "task_name": "Retrieval",
                        "dataset_revision": "a" * 40,
                        "scores": {
                            "test": [
                                {"hf_subset": "en", "main_score": 0.2 + index * 0.1},
                                {"hf_subset": "ar", "main_score": 0.4 + index * 0.1},
                            ]
                        },
                    },
                }
            },
        }
        record["aggregate"] = aggregate(record)
        records[role] = record
        plan["models"][role] = model
    return plan, records


def materialize(tmp_path, plan, records):
    paths = {}
    for role, record in records.items():
        paths[role] = tmp_path / (role + ".json")
        payload = json.dumps(record).encode()
        paths[role].write_bytes(payload)
        plan["models"][role]["receipt_sha256"] = sha(payload)
    path = tmp_path / "plan.json"
    payload = json.dumps(plan).encode()
    path.write_bytes(payload)
    return path, paths, sha(payload)


def compare_fixture(tmp_path, plan, records):
    path, inputs, pin = materialize(tmp_path, plan, records)
    return comparison.compare(
        path, inputs["baseline"], inputs["candidate"], plan_sha256=pin
    )


def test_complete_comparison_recomputes_rows_and_retains_model_specific_policies(
    tmp_path,
):
    result = compare_fixture(tmp_path, *fixture(tmp_path))
    assert result["matched"] and len(result["per_row"]) == 2
    assert result["mean_category_delta"] == pytest.approx(0.1)
    assert result["model_specific_differences"]["prompts"] == {
        "baseline": "none",
        "candidate": "query: ",
    }
    assert result["model_specific_differences"]["pooling"] == {
        "baseline": "last-token",
        "candidate": "mean",
    }
    assert result["architecture_effect_isolated"] is False
    assert result["paper_scores_included"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence_length", 8192),
        ("padding_policy", "fixed"),
        ("truncation_policy", None),
        ("score_scale", "percentage"),
        ("mteb_version", "2.22"),
    ],
)
def test_rehashed_receipt_with_different_shared_protocol_still_rejected(
    tmp_path, field, value
):
    plan, records = fixture(tmp_path)
    records["candidate"]["identity"][field] = value
    with pytest.raises(ValueError, match="shared protocol"):
        compare_fixture(tmp_path, plan, records)


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "subset",
        "split",
        "duplicate",
        "selected",
        "failure",
        "forged_aggregate",
        "score_nan",
        "category",
        "paper_only",
    ],
)
def test_incomplete_or_rehashed_score_evidence_cannot_become_matched(
    tmp_path, mutation
):
    plan, records = fixture(tmp_path)
    record = records["candidate"]
    raw = record["results"]["Retrieval"]["result"]
    if mutation == "revision":
        raw["dataset_revision"] = "d" * 40
    elif mutation == "subset":
        raw["scores"]["test"].pop()
    elif mutation == "split":
        raw["scores"]["train"] = raw["scores"].pop("test")
    elif mutation == "duplicate":
        raw["scores"]["test"].append(raw["scores"]["test"][0])
    elif mutation == "selected":
        record["selected_tasks"] = []
    elif mutation == "failure":
        record["failures"]["Retrieval"] = {"error": "failed"}
    elif mutation == "forged_aggregate":
        record["aggregate"]["mean_category_score"] = 99
    elif mutation == "score_nan":
        raw["scores"]["test"][0]["main_score"] = float("nan")
    elif mutation == "category":
        record["results"]["Retrieval"]["category"] = "STS"
    elif mutation == "paper_only":
        records["candidate"] = {"paper_score": 53.9}
    with pytest.raises((ValueError, KeyError)):
        compare_fixture(tmp_path, plan, records)


@pytest.mark.parametrize(
    "mutation",
    ["model_prompt", "scoring_source", "runtime_version", "backend", "devices"],
)
def test_declared_model_and_runtime_identity_are_enforced(tmp_path, mutation):
    plan, records = fixture(tmp_path)
    identity = records["candidate"]["identity"]
    if mutation == "model_prompt":
        identity["prompts"] = "undocumented prompt"
    elif mutation == "scoring_source":
        identity["protocol"]["source_sha256"]["scripts/evaluation_contract.py"] = (
            "a" * 64
        )
    elif mutation == "runtime_version":
        identity["protocol"]["runtime_packages"]["mteb"] = "other"
    elif mutation == "backend":
        identity["backend"] = "cpu"
    elif mutation == "devices":
        identity["protocol"]["devices"] = [{"platform": "cpu"}]
    with pytest.raises(ValueError):
        compare_fixture(tmp_path, plan, records)


def test_pinned_bytes_reject_tampering_even_if_schema_remains_valid(tmp_path):
    plan, records = fixture(tmp_path)
    path, inputs, pin = materialize(tmp_path, plan, records)
    records["candidate"]["identity"]["model"] = "changed"
    inputs["candidate"].write_text(json.dumps(records["candidate"]))
    with pytest.raises(ValueError, match="content hash"):
        comparison.compare(
            path, inputs["baseline"], inputs["candidate"], plan_sha256=pin
        )


def test_independently_frozen_inventory_prevents_both_receipts_narrowing(tmp_path):
    plan, records = fixture(tmp_path)
    for record in records.values():
        spec = record["inventory"]["Retrieval"]
        spec["hf_subsets"] = ["en"]
        spec["expected_rows"] = [["test", "en"]]
        record["results"]["Retrieval"]["result"]["scores"]["test"].pop()
        record["aggregate"] = aggregate(record)
    with pytest.raises(ValueError, match="independently frozen"):
        compare_fixture(tmp_path, plan, records)


@pytest.mark.parametrize(
    "inventory",
    [
        {
            "model-card.md": "a" * 64,
            "config.json": "a" * 64,
            "tokenizer.json": "a" * 64,
        },
        {"model.safetensors": "a" * 64},
    ],
)
def test_model_card_or_weights_without_config_tokenizer_is_not_model_identity(
    tmp_path, inventory
):
    plan, records = fixture(tmp_path)
    plan["models"]["candidate"]["files_sha256"] = inventory
    records["candidate"]["identity"]["files_sha256"] = inventory
    with pytest.raises(ValueError, match="artifact hash"):
        compare_fixture(tmp_path, plan, records)
