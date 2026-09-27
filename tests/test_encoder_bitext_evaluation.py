import json
import numpy as np
import pytest
from flaxchat.encoder_data import file_hash
from scripts.evaluate_encoder_bitext import evaluate_campaign


@pytest.mark.parametrize(
    "defect", [None, "missing", "model", "pairs", "alignment", "hash"]
)
def test_campaign_requires_complete_same_model_aligned_artifacts(tmp_path, defect):
    subsets = ["a-eng", "b-eng"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            dict(
                dataset="fixture",
                revision="pinned",
                shards=[dict(subset=s, pairs=3) for s in subsets],
            )
        )
    )
    for subset in subsets:
        if defect == "missing" and subset == subsets[1]:
            continue
        directory = tmp_path / subset
        directory.mkdir()
        np.save(directory / "queries.npy", np.array([[1.0, 0.0]] * 3))
        np.save(
            directory / "corpus.npy", np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        )
        manifest = dict(
            format="flaxchat-retrieval-embeddings-v1",
            dataset="fixture",
            revision="pinned",
            split="test",
            subset=subset,
            model_identity="weights-a",
            pooling="mean",
            query_ids=["q0", "q1", "q2"],
            document_ids=["d0", "d1", "d2"],
            qrels={f"q{i}": {f"d{i}": 1} for i in range(3)},
            languages={f"q{i}": subset for i in range(3)},
            queries_sha256=file_hash(directory / "queries.npy"),
            corpus_sha256=file_hash(directory / "corpus.npy"),
        )
        if subset == subsets[1]:
            if defect == "model":
                manifest["model_identity"] = "weights-b"
            if defect == "alignment":
                manifest["qrels"]["q1"] = {"d0": 1}
            if defect == "hash":
                manifest["queries_sha256"] = "wrong"
        (directory / "manifest.json").write_text(json.dumps(manifest))
    if defect == "pairs":
        data = json.loads(inventory.read_text())
        data["shards"][0]["pairs"] = 4
        inventory.write_text(json.dumps(data))
    if defect:
        with pytest.raises((ValueError, FileNotFoundError)):
            evaluate_campaign(tmp_path, inventory)
    else:
        result = evaluate_campaign(tmp_path, inventory)
        assert result["subset_count"] == 2 and result["pair_count"] == 6
        assert result["subset_macro"]["f1"] == pytest.approx(1 / 6)
        assert result["subset_macro"]["accuracy"] == pytest.approx(1 / 3)
        assert result["per_subset"][0]["predictions"] == [0, 0, 0]
        assert not result["quality_qualified"] and not result["official_mteb_parity"]
