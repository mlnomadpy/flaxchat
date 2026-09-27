import copy

import pytest

from tests.test_classifier_pair import fixture
from scripts.summarize_encoder_classifier_seeds import summarize


def inputs():
    plan, left, right = fixture()
    plan["seeds"] = [17, 29, 43]
    pairs = {}
    for seed in plan["seeds"]:
        plan["commands"][f"candidate-seed-{seed}"] = copy.deepcopy(
            plan["commands"]["candidate-seed-17"]
        )
        a, b = copy.deepcopy(left), copy.deepcopy(right)
        for r in (a, b):
            r["training_metadata"]["resolved_config"]["seed"] = seed
        pairs[seed] = (a, b)
    return plan, pairs


def test_all_seeds_report_actual_dispersion():
    plan, pairs = inputs()
    pairs[29][1]["datasets"][0].update(correct=6, accuracy=0.6)
    report = summarize(plan, pairs)
    delta = report["paired_macro_delta_percentage_points"]
    assert delta["mean"] == pytest.approx(5 / 3)
    assert delta["sample_stddev"] == pytest.approx(10 / 3**0.5)
    assert delta["minimum"] == pytest.approx(-5)
    assert not report["production_quality_qualified"]


@pytest.mark.parametrize(
    "defect", ["missing", "extra", "duplicate-plan", "paired-source", "partial"]
)
def test_reject_incomplete_or_incompatible_evidence(defect):
    plan, pairs = inputs()
    if defect == "missing":
        del pairs[43]
    elif defect == "extra":
        pairs[99] = pairs[17]
    elif defect == "duplicate-plan":
        plan["seeds"] = [17, 17, 29, 43]
    elif defect == "paired-source":
        for r in pairs[29]:
            r["training_metadata"]["source_python_sha256"] = "different"
    else:
        pairs[29][1]["checkpoint_step"] = 100
    with pytest.raises(ValueError):
        summarize(plan, pairs)
