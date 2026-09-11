import json
from pathlib import Path

from scripts.validate_result_provenance import DEFAULT_INDEX, ROOT, validate


def _copy_publication_fixture(tmp_path: Path) -> Path:
    index = json.loads(DEFAULT_INDEX.read_text(encoding="utf-8"))
    result_dir = tmp_path / "benchmarks" / "results"
    result_dir.mkdir(parents=True)
    for identity in index["records"].values():
        source = DEFAULT_INDEX.parent / identity["path"]
        (result_dir / source.name).write_bytes(source.read_bytes())
    for publication in index["publications"]:
        target = tmp_path / publication
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / publication).read_text(encoding="utf-8"), encoding="utf-8")
    target_index = result_dir / DEFAULT_INDEX.name
    target_index.write_text(json.dumps(index), encoding="utf-8")
    return target_index


def test_published_result_provenance_is_valid():
    assert validate() == []


def test_validator_rejects_stale_revision(tmp_path):
    index_path = _copy_publication_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["current_source_revision"] = "0" * 40
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert any("stale source revision" in error for error in validate(index_path, tmp_path))


def test_validator_rejects_missing_evidence_link(tmp_path):
    index_path = _copy_publication_fixture(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("benchmarks/results/provenance-index.json", ""),
        encoding="utf-8",
    )
    assert "README.md: missing provenance index link" in validate(index_path, tmp_path)


def test_validator_rejects_divergent_claim_value(tmp_path):
    index_path = _copy_publication_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["claims"]["strong_8_device_tokens_per_second"]["value"] = 1
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert any("claim value diverged" in error for error in validate(index_path, tmp_path))


def test_validator_rejects_missing_identity(tmp_path):
    index_path = _copy_publication_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["identity"]["hardware"]["pointer"] = "/hardware/not-there"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert "hardware: identity evidence is missing" in validate(index_path, tmp_path)
