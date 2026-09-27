import pytest
from scripts.plan_encoder_classification import plan


def records():
    return [
        dict(
            event="classifier_train_step",
            step=i + 1,
            updated=True,
            examples=32,
            includes_compilation=i == 0,
            loss=1.0,
            seconds=0.1,
        )
        for i in range(100)
    ]


def test_full_recipe_is_not_shortened_to_fit_lease():
    r = plan(
        records(),
        steps=12272,
        batch_size=32,
        evaluation_examples=37350,
        remaining_seconds=3000,
    )
    assert r["steps"] == 12272 and r["estimated_total_seconds"] < 3000
    assert not r["quality_qualified"]
    with pytest.raises(ValueError, match="do not fit"):
        plan(
            records(),
            steps=12272,
            batch_size=32,
            evaluation_examples=37350,
            remaining_seconds=1000,
        )


@pytest.mark.parametrize(
    "defect", ["short", "duplicate", "nan", "rejected", "batch", "compile", "duration"]
)
def test_invalid_calibration_cannot_authorize_full_training(defect):
    rows = records()
    if defect == "short":
        rows = rows[:20]
    elif defect == "duplicate":
        rows[-1]["step"] = 1
    elif defect == "nan":
        rows[-1]["loss"] = float("nan")
    elif defect == "rejected":
        rows[-1]["updated"] = False
    elif defect == "batch":
        rows[-1]["examples"] = 16
    elif defect == "compile":
        rows[-1]["includes_compilation"] = True
    elif defect == "duration":
        rows[-1]["seconds"] = 0
    with pytest.raises(ValueError):
        plan(
            rows,
            steps=12272,
            batch_size=32,
            evaluation_examples=37350,
            remaining_seconds=3000,
        )
