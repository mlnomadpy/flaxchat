"""Metadata-only first-stage admission; tiny tokenizer fixtures never qualify production."""

from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import hashlib
import copy
import pytest
from flaxchat.mlm_parent_exposure import validate_mlm_only_parent
from flaxchat.embedding_dev_exposure import parent_exposure

RELEASE = Path(__file__).resolve().parent / "fixtures/goat_mlm_lineage"


@pytest.fixture(autouse=True)
def isolated_test_pins(monkeypatch):
    from flaxchat import mlm_parent_exposure

    monkeypatch.setattr(
        mlm_parent_exposure,
        "MLM_ONLY_PARENTS",
        copy.deepcopy(mlm_parent_exposure.MLM_ONLY_PARENTS),
    )


def make_parent(tmp_path):
    root = tmp_path / "parent"
    root.mkdir()
    for name in [
        "config.json",
        "checkpoint-metadata.json",
        "checkpoint-manifest.json",
        "export.json",
    ]:
        shutil.copyfile(RELEASE / name, root / name)
    # Exercise lineage authentication with a tiny tokenizer fixture. This is
    # metadata-only testing, not release-byte or numerical acceptance. Original
    # public lineage documents stay separately retained in fixtures.
    from flaxchat.mlm_parent_exposure import MLM_ONLY_PARENTS
    from scripts.release_contract import canonical_hash, checkpoint_identity

    (root / "tokenizer.json").write_text('{"fixture": "metadata-only-tokenizer"}')
    token_hash = hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
    m = json.loads((root / "checkpoint-metadata.json").read_text())
    m["tokenizer_identity"] = token_hash
    (root / "checkpoint-metadata.json").write_text(json.dumps(m))
    c = json.loads((root / "checkpoint-manifest.json").read_text())
    c.update(metadata_sha256=canonical_hash(m), identity=checkpoint_identity(m))
    c["identity_sha256"] = canonical_hash(c["identity"])
    (root / "checkpoint-manifest.json").write_text(json.dumps(c))
    e = json.loads((root / "export.json").read_text())
    e.update(
        tokenizer_identity=token_hash,
        source_checkpoint_metadata_sha256=canonical_hash(m),
        source_checkpoint_manifest_identity_sha256=c["identity_sha256"],
    )
    for n in [
        "config.json",
        "tokenizer.json",
        "checkpoint-metadata.json",
        "checkpoint-manifest.json",
    ]:
        e["artifacts_sha256"][n] = hashlib.sha256((root / n).read_bytes()).hexdigest()
    (root / "export.json").write_text(json.dumps(e))
    MLM_ONLY_PARENTS[e["sha256"]].update(
        metadata=canonical_hash(m),
        manifest_identity=c["identity_sha256"],
        export_sha256=hashlib.sha256((root / "export.json").read_bytes()).hexdigest(),
    )
    return root, {
        n: e["artifacts_sha256"][n]
        for n in ["model.safetensors", "config.json", "tokenizer.json"]
    }


def test_known_mlm_parent_admits_empty_contrastive_history_without_claiming_unseen_mlm(
    tmp_path,
):
    root, hashes = make_parent(tmp_path)
    spec = dict(
        format="flaxchat-known-contrastive-exposure-input-v1",
        parent_files_sha256=hashes,
        known_contrastive_inventory_complete=True,
        expected_stage_identities=[],
        stages=[],
        mlm_only_parent={"directory": "parent"},
    )
    path = tmp_path / "exposure.json"
    path.write_text(json.dumps(spec))
    p = parent_exposure(
        path, SimpleNamespace(identity={"fixture": "candidate"}), hashes
    )
    assert p["complete"] and not p["checked_stages"] and p["overlap_rows"] == 0
    assert p["mlm_only_parent"]["step"] == 379079
    assert "inherited MLM/pretraining exposure" in p["unresolved_exposure"]
    spec.pop("mlm_only_parent")
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Complete immutable"):
        parent_exposure(path, SimpleNamespace(identity={}), hashes)


@pytest.mark.parametrize(
    "file",
    [
        "checkpoint-metadata.json",
        "checkpoint-manifest.json",
        "config.json",
        "export.json",
        "tokenizer.json",
    ],
)
def test_mlm_lineage_mutation_is_rejected(tmp_path, file):
    root, hashes = make_parent(tmp_path)
    p = root / file
    d = json.loads(p.read_text())
    d["unauthorized_edit"] = True
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="lineage identity mismatch"):
        validate_mlm_only_parent(root, hashes)


def test_same_family_cannot_certify_unknown_weights_as_zero_exposure(tmp_path):
    root, hashes = make_parent(tmp_path)
    hashes["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="No independently authenticated"):
        validate_mlm_only_parent(root, hashes)


def test_mlm_parent_cannot_mix_in_unchecked_contrastive_stages(tmp_path):
    root, hashes = make_parent(tmp_path)
    p = tmp_path / "exposure.json"
    p.write_text(
        json.dumps(
            dict(
                format="flaxchat-known-contrastive-exposure-input-v1",
                parent_files_sha256=hashes,
                known_contrastive_inventory_complete=True,
                expected_stage_identities=[],
                stages=[{"stage_identity_sha256": "0" * 64}],
                mlm_only_parent={"directory": "parent"},
            )
        )
    )
    with pytest.raises(ValueError, match="stage coverage mismatch"):
        parent_exposure(p, SimpleNamespace(identity={}), hashes)
