from copy import deepcopy

import pytest

from scripts.yat_benchmark_admission import require_candidates


def passing_report():
    checks = {field: {"passed": True} for field in ("output", "q", "k", "v", "alpha")}
    return dict(
        complete=True, backend="cpu", correctness_passed=True, failures={},
        rows=[dict(variant=name, radius=radius, layer=layer, passed=True, checks=deepcopy(checks))
              for name in ("candidate", "baseline") for radius in (None, 64) for layer in (0, 7, 14, 21)],
        timings=[dict(radius=radius, checks={name: deepcopy(checks) for name in ("candidate", "baseline")},
                      samples_seconds={name: [.01]*20 for name in ("candidate", "baseline")})
                 for radius in (None, 128)],
    )


def test_baseline_success_cannot_hide_candidate_failure():
    report = passing_report()
    report["failures"]["candidate"] = {"stage": "saved_oracle"}
    assert report["correctness_passed"]
    with pytest.raises(ValueError, match="Required candidate failed"):
        require_candidates(report, ["candidate"], backend="cpu")


@pytest.mark.parametrize("mutation", ["missing_saved", "duplicate_mask", "missing_timing", "nonfinite", "failed_gradient"])
def test_incomplete_or_failed_candidate_is_rejected(mutation):
    report = passing_report()
    if mutation == "missing_saved":
        report["rows"].pop(0)
    elif mutation == "duplicate_mask":
        report["timings"][1]["radius"] = None
    elif mutation == "missing_timing":
        del report["timings"][0]["samples_seconds"]["candidate"]
    elif mutation == "nonfinite":
        report["timings"][0]["samples_seconds"]["candidate"][0] = float("nan")
    else:
        report["rows"][0]["checks"]["q"]["passed"] = False
    with pytest.raises(ValueError):
        require_candidates(report, ["candidate"], backend="cpu")


def test_complete_candidate_report_is_accepted():
    assert require_candidates(passing_report(), ["candidate"], backend="cpu")["passed"]


def test_backend_must_match():
    with pytest.raises(ValueError, match="unexpected backend"):
        require_candidates(passing_report(), ["candidate"], backend="tpu")
