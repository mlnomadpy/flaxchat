"""Driver admission tests only; no model, backend, or numerical acceptance."""

import json
import pytest
from scripts import qualify_goat_contrastive_tpu as qualification
from scripts.run_goat_contrastive_pilot import next_stop


def traces(tmp_path, monkeypatch):
    run = dict(event="embedding_run", backend="tpu", devices=8, processes=1)
    rows = [
        dict(
            event="embedding_step",
            step=i,
            loss=1.0 / i,
            gradient_norm=0.2,
            learning_rate=i * 1e-7,
            batch_sha256=str(i) * 64,
            candidate_masking={},
            global_pairs=128,
            pair_only=i % 2 == 0,
        )
        for i in range(1, 5)
    ]
    logs = [tmp_path / x for x in ["reference", "first", "resumed"]]
    for p, part in zip(logs, [rows, rows[:2], rows[2:]], strict=True):
        p.write_text("\n".join(json.dumps(x) for x in [run, *part]))
    state = {
        x: {"leaf": {"sha256": "a" * 64}}
        for x in ["model_state", "optimizer_state", "training_state"]
    }
    roots = [tmp_path / x for x in ["left", "right"]]
    for root in roots:
        (root / "4/manifest").mkdir(parents=True)
        (root / "4/manifest/metadata").write_text(json.dumps(state))
    monkeypatch.setattr(qualification, "read_committed_metadata", lambda *a, **k: {})
    return logs, roots


def test_matching_actual_trace_and_state_metadata(tmp_path, monkeypatch):
    logs, roots = traces(tmp_path, monkeypatch)
    assert qualification.compare(*logs, *roots)["exact_model_Adam_training_state"]


@pytest.mark.parametrize("kind", ["rows", "LR", "backend", "Adam", "cursor"])
def test_recovery_evidence_drift_rejected(tmp_path, monkeypatch, kind):
    logs, roots = traces(tmp_path, monkeypatch)
    if kind in ["rows", "LR", "backend"]:
        rows = [json.loads(x) for x in logs[2].read_text().splitlines()]
        if kind == "backend":
            rows[0]["backend"] = "cpu"
        elif kind == "rows":
            rows[1]["batch_sha256"] = "f" * 64
        else:
            rows[1]["learning_rate"] = 0.01
        logs[2].write_text("\n".join(json.dumps(x) for x in rows))
    else:
        p = roots[1] / "4/manifest/metadata"
        v = json.loads(p.read_text())
        v["optimizer_state" if kind == "Adam" else "training_state"]["leaf"][
            "sha256"
        ] = "b" * 64
        p.write_text(json.dumps(v))
    with pytest.raises(ValueError):
        qualification.compare(*logs, *roots)


def test_forecast_preserves_evidence_and_horizon():
    assert next_stop(0, None, 27000) == 250
    assert next_stop(250, 600, 2500) is None
    assert next_stop(250, 600, 3000) == 500
    assert next_stop(10000, 600, 27000) is None
    assert next_stop(250, float("inf"), 27000) is None


def test_qualification_development_budget_covers_multilingual_mixture(tmp_path):
    import numpy as np
    from flaxchat.goat_contrastive_campaign import stage_arguments
    from flaxchat.embedding_quality import stratified_dev_indices

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = [
        "marco_replay",
        "code_replay",
        "bitext_replay",
        *(
            "native_" + language
            for language in ["ar", "bn", "fr", "hi", "ja", "ru", "sw", "zh"]
        ),
    ]
    (inputs / "source-weights.json").write_text(json.dumps({n: 1 / 11 for n in names}))
    args = stage_arguments(
        tmp_path, tmp_path / "checkpoints", scope="qualification", stop=4
    )
    maximum = int(args[args.index("--dev-max-rows") + 1])
    languages = np.repeat(np.arange(28), 10)
    with pytest.raises(ValueError, match="every language"):
        stratified_dev_indices(languages, 32, 1028)
    selected = stratified_dev_indices(languages, maximum, 1028)
    assert len(set(languages[selected])) == 28
    assert min(np.unique(languages[selected], return_counts=True)[1]) >= 2
