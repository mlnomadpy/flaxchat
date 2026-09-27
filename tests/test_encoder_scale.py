import json
import numpy as np
import pytest

from scripts.validate_encoder_scale import validate_updates
from scripts.evaluate_encoder_xnli import pair_features, fit_probe, score_probe


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "missing-step",
        "nan",
        "rejected",
        "fallback",
        "missing-checkpoint",
        "short",
        "wrong-tokens",
        "unmarked-compilation",
    ],
)
def test_endurance_requires_complete_measured_updates(tmp_path, defect):
    rows = [
        dict(
            event="train_step",
            step=i,
            updated=True,
            projection_dense_fallback=False,
            tokens=64 * 512,
            masked_tokens=100,
            loss=2.0,
            seconds=10.0,
            includes_compilation=i == 1,
        )
        for i in range(1, 5)
    ]
    rows.append(dict(event="checkpoint", step=4, seconds=1.0))
    if defect == "missing-step":
        rows.pop(1)
    if defect == "nan":
        rows[1]["loss"] = float("nan")
    if defect == "rejected":
        rows[1]["updated"] = False
    if defect == "fallback":
        rows[1]["projection_dense_fallback"] = True
    if defect == "missing-checkpoint":
        rows.pop()
    if defect == "unmarked-compilation":
        rows[0]["includes_compilation"] = False
    if defect == "wrong-tokens":
        rows[1]["tokens"] = 3
    path = tmp_path / "run.log"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    kwargs = dict(
        first=1,
        last=4,
        batch=64,
        length=512,
        minimum_seconds=31 if defect == "short" else 30,
    )
    if defect:
        with pytest.raises(ValueError):
            validate_updates(path, **kwargs)
    else:
        report = validate_updates(path, **kwargs)
        assert report["measured_seconds"] == 30
        assert report["tokens_per_second"] == 64 * 512 / 10
        assert report["masked_targets_per_second"] == 10


def test_probe_separable_reference_and_confidence_interval():
    x = np.repeat(np.eye(3), 10, axis=0)
    labels = np.repeat(np.arange(3), 10)
    features = pair_features(x, x)
    score = score_probe(features, labels, fit_probe(features, labels))
    assert score["accuracy"] == 1
    assert 0 < score["wilson_95"][0] < score["wilson_95"][1] <= 1


@pytest.mark.parametrize('bad_earlier_checkpoint', [False, True])
def test_timing_report_separates_warm_steps_and_checkpoint_cost(tmp_path, bad_earlier_checkpoint):
    rows = [dict(event='train_step', step=i, updated=True,
                 projection_dense_fallback=False, tokens=128, masked_tokens=20,
                 loss=2., seconds=duration, includes_compilation=i == 1,
                 includes_profiling=i == 2)
            for i, duration in enumerate((60., 10., 2., 4.), 1)]
    rows.extend([dict(event='checkpoint', step=2, seconds=float('nan') if bad_earlier_checkpoint else 7.),
                 dict(event='checkpoint', step=4, seconds=9.)])
    path = tmp_path/'timings.log'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    if bad_earlier_checkpoint:
        with pytest.raises(ValueError, match='checkpoint'):
            validate_updates(path, first=1, last=4, batch=1, length=128)
        return
    result = validate_updates(path, first=1, last=4, batch=1, length=128)
    assert result['measured_steps'] == 2
    assert result['measured_seconds'] == 6
    assert result['median_measured_step_seconds'] == 3
    assert result['tokens_per_second'] == 256/6
    assert result['checkpoint_seconds'] == 16 and result['checkpoint_events'] == 2
    assert result['compilation_inclusive_step_seconds'] == 60
    assert result['profiling_inclusive_step_seconds'] == 10


@pytest.mark.parametrize("defect", ["nan", "missing-class", "bad-ridge"])
def test_probe_rejects_invalid_evidence(defect):
    x = np.eye(3)
    labels = np.arange(3)
    if defect == "nan":
        x[0, 0] = np.nan
    if defect == "missing-class":
        labels[:] = 0
    with pytest.raises(ValueError):
        fit_probe(x, labels, regularization=0 if defect == "bad-ridge" else 1)


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "missing-fault",
        "raw-mismatch",
        "controller-failed",
        "duplicate-controller-rank",
        "missing-controller-rank",
        "boolean-controller-returncode",
        "nonprimary-updates",
        "missing-worker",
        "wrong-backend",
        "stale-manifest",
    ],
)
def test_scale_aggregation_requires_raw_recovery_and_controller(tmp_path, defect):
    from scripts.summarize_encoder_scale import summarize

    controller = tmp_path / "controller.json"
    controller_defects = {
        "controller-failed",
        "duplicate-controller-rank",
        "missing-controller-rank",
        "boolean-controller-returncode",
    }
    workers = [{"rank": rank, "returncode": 0} for rank in range(2)]
    if defect == "duplicate-controller-rank":
        workers[1]["rank"] = 0
    elif defect == "missing-controller-rank":
        del workers[1]["rank"]
    elif defect == "boolean-controller-returncode":
        workers[1]["returncode"] = False
    controller.write_text(
        json.dumps(
            dict(
                passed=defect != "controller-failed",
                workers=workers,
            )
        )
    )
    for rank in range(2):
        if defect == "missing-worker" and rank == 1:
            continue
        d = tmp_path / f"worker-{rank}"
        d.mkdir()
        hardware = dict(
            backend="tpu", process_count=2, device_count=8, process_index=rank
        )
        (d / "hardware.log").write_text(json.dumps(hardware))
        (d / "summary.json").write_text(
            json.dumps(
                dict(
                    hardware=hardware,
                    passed=True,
                    steps=4,
                    batch_size=64,
                    minimum_seconds=30,
                    mlm_loss_backend="pallas" if defect == "wrong-backend" else "xla",
                    stages=[
                        dict(name=n, returncode=c, passed=True)
                        for n, c in [
                            ("hardware", 0),
                            ("baseline", 0),
                            ("interrupt", -9),
                            ("resume", 0),
                        ]
                    ],
                )
            )
        )
        (d / "interrupt.log").write_text(
            "killed"
            if defect == "missing-fault"
            else "FAULT_INJECTION: encoder SIGKILL after committed step 2"
        )
        manifest = dict(
            step=3 if defect == "stale-manifest" else 4,
            model_state={"x": "abc"},
            optimizer_state={"x": "xyz"},
            training_state={"step": 4},
        )
        (d / "baseline-manifest.json").write_text(json.dumps(manifest))
        if defect == "raw-mismatch":
            manifest["model_state"]["x"] = "bad"
        (d / "recovery-manifest.json").write_text(json.dumps(manifest))
        for stage, first in [("baseline", 1), ("resume", 3)]:
            rows = (
                [
                    dict(
                        event="train_step",
                        step=i,
                        updated=True,
                        projection_dense_fallback=False,
                        tokens=64 * 512,
                        masked_tokens=12,
                        loss=1.0,
                        seconds=10.0,
                        includes_compilation=i == first,
                    )
                    for i in range(first, 5)
                ]
                if rank == 0 or defect == "nonprimary-updates"
                else []
            )
            rows.append(dict(event="checkpoint", step=4, seconds=1.0))
            (d / f"{stage}.log").write_text("\n".join(json.dumps(r) for r in rows))
    report = summarize(
        tmp_path,
        controller,
        processes=2,
        devices=8,
        steps=4,
        batch=64,
        minimum_seconds=30,
    )
    assert report["passed"] == (defect is None)
    assert report["controller_passed"] == (defect not in controller_defects)
    assert report["raw_worker_evidence_passed"] == (
        defect is None or defect in controller_defects
    )
    assert report["production_quality_qualified"] is False


@pytest.mark.parametrize(
    "defect", [None, "dataset", "language", "count", "macro", "nan"]
)
def test_quality_comparison_requires_matching_raw_evidence(defect):
    from copy import deepcopy
    from scripts.compare_encoder_xnli import compare

    reference = dict(
        scope="xnli_frozen_encoder_probe",
        dataset=dict(languages=["en", "ar"], npz_sha256="abc"),
        scores=dict(
            en=dict(accuracy=0.5, examples=100), ar=dict(accuracy=0.4, examples=100)
        ),
        macro_accuracy=0.45,
    )
    candidate = deepcopy(reference)
    candidate["scores"]["en"]["accuracy"] = 0.6
    candidate["macro_accuracy"] = 0.5
    if defect == "dataset":
        candidate["dataset"]["npz_sha256"] = "different"
    elif defect == "language":
        del candidate["scores"]["ar"]
    elif defect == "count":
        candidate["scores"]["en"]["examples"] = 99
    elif defect == "macro":
        candidate["macro_accuracy"] = 0.9
    elif defect == "nan":
        candidate["scores"]["ar"]["accuracy"] = float("nan")
    if defect:
        with pytest.raises(ValueError):
            compare(reference, candidate)
    else:
        result = compare(reference, candidate)
        assert result["macro_delta_percentage_points"] == pytest.approx(5)
        assert result["language_delta_percentage_points"]["ar"] == 0
        assert result["production_quality_qualified"] is False


@pytest.mark.parametrize(
    "macro_limit,language_limit,expected",
    [(3, 4, True), (1, 4, False), (3, 1, False), (None, 2, "error"), (-1, 2, "error")],
)
def test_predeclared_quality_gate(macro_limit, language_limit, expected):
    from copy import deepcopy
    from scripts.compare_encoder_xnli import compare

    reference = dict(
        scope="xnli_frozen_encoder_probe",
        dataset=dict(languages=["en", "ar"]),
        scores=dict(
            en=dict(accuracy=0.5, examples=100), ar=dict(accuracy=0.5, examples=100)
        ),
        macro_accuracy=0.5,
    )
    candidate = deepcopy(reference)
    candidate["scores"]["en"]["accuracy"] = 0.46
    candidate["macro_accuracy"] = 0.48
    kwargs = dict(max_macro_drop_pp=macro_limit, max_language_drop_pp=language_limit)
    if expected == "error":
        with pytest.raises(ValueError):
            compare(reference, candidate, **kwargs)
    else:
        result = compare(reference, candidate, **kwargs)
        assert result["regression_gate_passed"] is expected
        assert result["production_quality_qualified"] is False


@pytest.mark.parametrize(
    "defect", [None, "missing", "boolean", "overfull", "below-targets"]
)
def test_endurance_separates_padding_from_useful_tokens(tmp_path, defect):
    rows = [
        dict(
            event="train_step",
            step=i,
            updated=True,
            projection_dense_fallback=False,
            tokens=64 * 512,
            nonpadding_tokens=1000 if i == 1 else 2000,
            masked_tokens=100,
            loss=1.0,
            seconds=10.0,
            includes_compilation=i == 1,
        )
        for i in range(1, 5)
    ]
    if defect == "missing":
        del rows[2]["nonpadding_tokens"]
    elif defect == "boolean":
        rows[2]["nonpadding_tokens"] = True
    elif defect == "overfull":
        rows[2]["nonpadding_tokens"] = 64 * 512 + 1
    elif defect == "below-targets":
        rows[2]["nonpadding_tokens"] = 99
    rows.append(dict(event="checkpoint", step=4, seconds=1.0))
    log = tmp_path / "train.log"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    if defect:
        with pytest.raises(ValueError, match="nonpadding"):
            validate_updates(log, first=1, last=4, batch=64, length=512)
    else:
        report = validate_updates(log, first=1, last=4, batch=64, length=512)
        assert report["tokens_per_second_basis"] == "padded_input_positions"
        assert report["tokens_per_second"] == 3276.8
        assert report["nonpadding_tokens_per_second"] == 200
        assert report["nonpadding_fraction"] == 2000 / 32768
        for row in rows:
            row.pop("nonpadding_tokens", None)
        log.write_text("\n".join(json.dumps(r) for r in rows))
        legacy = validate_updates(log, first=1, last=4, batch=64, length=512)
        assert legacy["nonpadding_tokens_per_second"] is None
        assert legacy["nonpadding_fraction"] is None
