import copy
import json
import subprocess
import sys
from typing import Any

import pytest

from scripts.validate_encoder_classifier_pair import compare, validate_training


def fixture() -> tuple[Any, Any, Any]:
    frozen = dict(
        steps=12272,
        batch_size=32,
        warmup_steps=737,
        languages=["en", "fr"],
        commands={
            "candidate-seed-17": {
                "train": [
                    "--encoder-checkpoint",
                    "gs://candidate",
                    "--encoder-step",
                    "1972",
                ]
            }
        },
        dataset_audit=[
            dict(split="validation", language=language, path=language, rows=10)
            for language in ("en", "fr")
        ],
        input_sha256={
            language + "/manifest.json": language + "hash" for language in ("en", "fr")
        },
    )
    recipe = dict(
        steps=12272,
        batch_size=32,
        warmup_steps=737,
        seed=17,
        origin=dict(released_weights={"weights": "sha"}),
    )
    metadata = dict(
        step=12272,
        resolved_config=recipe,
        source_python_sha256="source",
        data_manifest_identity="trainhash",
        tokenizer_identity="tokenizer",
        dataset="xnli",
        dataset_revision="revision",
    )
    left: dict[str, Any] = dict(
        scope="end_to_end_encoder_classification",
        checkpoint_step=12272,
        training_metadata=metadata,
        evaluator_source_sha256="evaluator",
        runtime={"jax": "pinned"},
        datasets=[
            dict(
                data_manifest=dict(language=language, split="validation"),
                data_manifest_sha256=language + "hash",
                examples=10,
                correct=7,
                accuracy=0.7,
                loss=0.5,
            )
            for language in ("en", "fr")
        ],
    )
    right = copy.deepcopy(left)
    right["training_metadata"]["resolved_config"]["origin"]["checkpoint"] = dict(
        path="gs://candidate", step=1972
    )
    right["datasets"][0].update(correct=8, accuracy=0.8)
    return frozen, left, right


def test_matched_full_comparison_is_not_production_acceptance():
    frozen, left, right = fixture()
    r = compare(left, right, frozen, 17)
    assert r["protocol_matched"] and not r["production_quality_qualified"]
    assert r["macro_delta_percentage_points"] == pytest.approx(5)


@pytest.mark.parametrize(
    "defect",
    [
        "step",
        "seed",
        "source",
        "runtime",
        "manifest",
        "missing-language",
        "duplicate-language",
        "test-split",
        "count",
        "metric",
        "origin",
        "baseline-origin",
        "recipe",
    ],
)
def test_incomplete_or_unmatched_comparison_rejected(defect):
    frozen, left, right = fixture()
    if defect == "step":
        right["checkpoint_step"] = 100
    if defect == "seed":
        right["training_metadata"]["resolved_config"]["seed"] = 29
    if defect == "source":
        right["evaluator_source_sha256"] = "other"
    if defect == "runtime":
        right["runtime"] = {}
    if defect == "manifest":
        right["datasets"][0]["data_manifest_sha256"] = "changed"
    if defect == "missing-language":
        right["datasets"].pop()
    if defect == "duplicate-language":
        right["datasets"].append(right["datasets"][0])
    if defect == "test-split":
        right["datasets"][0]["data_manifest"]["split"] = "test"
    if defect == "count":
        right["datasets"][0]["examples"] = 9
    if defect == "metric":
        right["datasets"][0]["correct"] = 0
    if defect == "origin":
        right["training_metadata"]["resolved_config"]["origin"]["checkpoint"][
            "step"
        ] = 100
    if defect == "baseline-origin":
        left["training_metadata"]["resolved_config"]["origin"]["checkpoint"] = {}
    if defect == "recipe":
        right["training_metadata"]["resolved_config"]["learning_rate"] = 9
    with pytest.raises(ValueError):
        compare(left, right, frozen, 17)


@pytest.mark.parametrize(
    "defect", [None, "cpu", "missing", "rejected", "nan", "compile"]
)
def test_training_requires_complete_accepted_physical_updates(tmp_path, defect):
    rows: list[dict[str, Any]] = [
        dict(event="classifier_run_config", devices=4, processes=1, backend="tpu")
    ]
    rows += [
        dict(
            event="classifier_train_step",
            step=i,
            examples=32,
            updated=True,
            loss=1.0,
            seconds=0.1,
            includes_compilation=i == 101,
        )
        for i in range(101, 104)
    ]
    if defect == "cpu":
        rows[0]["backend"] = "cpu"
    if defect == "missing":
        rows.pop()
    if defect == "rejected":
        rows[-1]["updated"] = False
    if defect == "nan":
        rows[-1]["loss"] = float("nan")
    if defect == "compile":
        rows[-1]["includes_compilation"] = True
    p = tmp_path / "log"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    if defect:
        with pytest.raises(ValueError):
            validate_training(p, first=101, last=103)
    else:
        validate_training(p, first=101, last=103)


def test_parent_import_does_not_initialize_jax():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.validate_encoder_classifier_pair; assert 'jax' not in sys.modules; assert 'flaxchat' not in sys.modules",
        ],
        check=True,
        timeout=30,
    )
