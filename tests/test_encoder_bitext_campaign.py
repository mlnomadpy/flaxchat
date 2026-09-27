"""Complete campaign execution and fail-closed frozen-input boundaries."""

import json
from pathlib import Path
import shutil
from dataclasses import asdict

import pytest
import numpy as np
import optax
from flax import nnx
from safetensors.numpy import save_file

from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
from flaxchat.encoder_data import file_hash
from scripts import export_encoder_bitext_campaign as campaign
from scripts.train_encoder import pretrained_shapes
from tests.test_encoder_retrieval_export import fixture


def prepared(tmp_path):
    model, _, args = fixture(tmp_path)
    checkpoint, queries, corpus, judgments, _ = args
    snapshot = tmp_path / "released"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(asdict(model.config) | {"model_type": "modernbert"})
    )
    (snapshot / "tokenizer.json").write_text(
        json.dumps(
            {"added_tokens": [{"id": i, "special": True} for i in [0, 1, 2, 3, 4]]}
        )
    )
    rng = np.random.default_rng(5)
    save_file(
        {
            k: rng.normal(size=shape).astype(np.float32)
            for k, shape in pretrained_shapes(model.config).items()
        },
        str(snapshot / "model.safetensors"),
    )
    checkpoint = str(tmp_path / "pinned-checkpoint")
    metadata = dict(
        model_family="modernbert",
        tokenizer_identity=file_hash(snapshot / "tokenizer.json"),
        resolved_config=dict(
            encoder=asdict(model.config), special_token_ids=[0, 1, 2, 3, 4]
        ),
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    with create_checkpoint_manager(checkpoint, async_checkpointing=False) as manager:
        save_checkpoint(manager, 2, model, optimizer, metadata)
        manager.wait_until_finished()
    entries = []
    hashes = {}
    for subset in ("a-eng", "b-eng"):
        root = tmp_path / "prepared" / subset
        root.mkdir(parents=True)
        for role, source in (("queries", queries), ("corpus", corpus)):
            shutil.copytree(source, root / role)
            path = root / role / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["split"] = "test"
            manifest["tokenizer_sha256"] = metadata["tokenizer_identity"]
            path.write_text(json.dumps(manifest))
        task = json.loads(judgments.read_text()) | dict(subset=subset, split="test")
        (root / "judgments.json").write_text(json.dumps(task))
        for path in root.rglob("*"):
            if path.is_file():
                hashes[str(path)] = file_hash(path)
        entries.append(dict(subset=subset, pairs=3))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(dict(dataset="fixture", revision="v1", shards=entries))
    )
    plan = dict(
        campaign._RECIPE,
        dataset="fixture",
        revision="v1",
        inventory_path=str(inventory),
        inventory_sha256=file_hash(inventory),
        subsets=2,
        pairs=6,
        sequence_length=4,
        input_sha256=hashes,
        baseline=dict(
            snapshot=str(snapshot),
            weights_sha256=file_hash(snapshot / "model.safetensors"),
        ),
        candidate=dict(checkpoint=checkpoint, step=2),
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def test_complete_real_checkpoint_campaign_restores_once(tmp_path, monkeypatch):
    path, _ = prepared(tmp_path)
    calls = []
    original = campaign.EmbeddingSession

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign, "EmbeddingSession", counted)
    report = campaign.run(path, "candidate", tmp_path / "output", batch_size=2)
    assert len(calls) == 1 and calls[0][1] == {"step": 2}
    assert report["complete"]
    assert report["scores"]["subset_count"] == 2
    assert report["scores"]["pair_count"] == 6
    assert report["scores"]["subset_macro"]["accuracy"] == 1
    assert not report["production_quality_qualified"]
    assert len(report["manifests"]) == len(report["subset_timings"]) == 2
    assert json.loads((tmp_path / "output/report.json").read_text()) == report
    with pytest.raises(ValueError, match="existing"):
        campaign.run(path, "candidate", tmp_path / "output")


@pytest.mark.parametrize(
    "mutation", ["file", "inventory", "coverage", "extra", "weights", "recipe", "step"]
)
def test_preflight_rejects_mutation_before_model_allocation(
    tmp_path, monkeypatch, mutation
):
    path, plan = prepared(tmp_path)
    if mutation == "file":
        Path(next(iter(plan["input_sha256"]))).write_text("corrupt")
    elif mutation == "inventory":
        Path(plan["inventory_path"]).write_text("{}")
    elif mutation == "coverage":
        plan["pairs"] += 1
    elif mutation == "extra":
        plan["input_sha256"]["unexpected"] = "unused"
    elif mutation == "weights":
        plan["baseline"]["weights_sha256"] = "wrong"
    elif mutation == "recipe":
        plan["all_pairs_required"] = False
    else:
        plan["candidate"]["step"] = None
    path.write_text(json.dumps(plan))

    def forbidden(*args, **kwargs):
        pytest.fail("Model allocated for invalid frozen plan")

    monkeypatch.setattr(campaign, "EmbeddingSession", forbidden)
    with pytest.raises(ValueError):
        campaign.run(path, "candidate", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_interrupted_campaign_never_publishes_complete_report(tmp_path, monkeypatch):
    path, _ = prepared(tmp_path)
    original = campaign.EmbeddingSession.export
    calls = 0

    def fail_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("worker interrupted")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(campaign.EmbeddingSession, "export", fail_second)
    with pytest.raises(TimeoutError):
        campaign.run(path, "candidate", tmp_path / "output", batch_size=2)
    assert not (tmp_path / "output/report.json").exists()
    progress = json.loads((tmp_path / "output/progress.json").read_text())
    assert not progress["complete"] and len(progress["completed_subsets"]) == 1


def test_matched_released_and_checkpoint_comparison_recomputes_scores(tmp_path):
    from scripts.compare_encoder_bitext_campaign import compare

    path, _ = prepared(tmp_path)
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    campaign.run(path, "baseline", baseline, batch_size=2)
    campaign.run(path, "candidate", candidate, batch_size=2)
    result = compare(path, baseline, candidate)
    assert result["complete"] and result["pairs"] == 6 and result["subsets"] == 2
    assert result["delta"]["accuracy"] == 0
    assert not result["production_quality_qualified"]
    report_path = candidate / "report.json"
    original = json.loads(report_path.read_text())
    changed = json.loads(report_path.read_text())
    changed["scores"]["subset_macro"]["accuracy"] = 0.25
    report_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="recomputation"):
        compare(path, baseline, candidate)
    report_path.write_text(json.dumps(original))
    (candidate / "embeddings/a-eng/queries.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        compare(path, baseline, candidate)
