import copy
import os
import subprocess
import sys
from typing import Any

import pytest

from scripts.calibrate_encoder_classifier import calibration_command, review


def fixture():
    recipe: dict[str, Any] = dict(
        steps=12272, batch_size=32, warmup_steps=737, evaluation_examples=37350
    )
    recipe["commands"] = {
        "baseline-seed-17": {
            "train": [
                "python",
                "-m",
                "scripts.finetune_encoder_classifier",
                "--steps",
                "12272",
                "--batch-size",
                "32",
                "--warmup-steps",
                "737",
                "--output",
                "gs://bucket/run",
            ]
        }
    }
    records: list[dict[str, Any]] = [
        dict(
            event="classifier_run_config",
            backend="tpu",
            devices=4,
            processes=1,
            recipe=copy.deepcopy(recipe),
        )
    ]
    records += [
        dict(
            event="classifier_train_step",
            step=i,
            examples=32,
            updated=True,
            includes_compilation=i == 1,
            loss=1.0,
            seconds=0.1,
        )
        for i in range(1, 101)
    ]
    return recipe, records


def test_calibration_preserves_full_schedule_and_output():
    frozen, records = fixture()
    argv = calibration_command(frozen, "baseline-seed-17")
    assert argv[-2:] == ["--stop-after", "100"]
    assert argv[argv.index("--steps") + 1] == "12272"
    assert argv[argv.index("--output") + 1] == "gs://bucket/run"
    result = review(records, frozen, devices=4, lease_seconds=6500)
    assert result["steps"] == 12272
    assert result["full_recipe_execution_still_required"]
    assert result["quality_qualified"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "cpu",
        "topology",
        "missing-update",
        "rejected",
        "schedule",
        "duplicate-config",
        "short-lease",
    ],
)
def test_bad_physical_evidence_rejected(defect):
    frozen, records = fixture()
    if defect == "cpu":
        records[0]["backend"] = "cpu"
    if defect == "topology":
        records[0]["processes"] = 2
    if defect == "missing-update":
        records.pop()
    if defect == "rejected":
        records[3]["updated"] = False
    if defect == "schedule":
        records[0]["recipe"]["steps"] = 100
    if defect == "duplicate-config":
        records.append(records[0])
    with pytest.raises(ValueError):
        review(
            records,
            frozen,
            devices=4,
            lease_seconds=10 if defect == "short-lease" else 6500,
        )


def test_changed_command_schedule_rejected():
    frozen, _ = fixture()
    frozen["steps"] = 100
    with pytest.raises(ValueError):
        calibration_command(frozen, "baseline-seed-17")


def test_coordinator_import_does_not_claim_tpu():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.calibrate_encoder_classifier; assert 'jax' not in sys.modules; assert 'flaxchat' not in sys.modules",
        ],
        env=os.environ,
        check=True,
        timeout=30,
    )
