"""Corpus identity/quarantine/provenance contracts; no model or accelerator calls."""

import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from scripts import prepare_yat_mlm_continuation as prep
from scripts.prepare_representation_development import export


def candidate(tmp_path):
    spec = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/data/representation-development-v1.json"
        ).read_text()
    )
    spec.update(max_scanned_rows_per_config=100, max_selected_rows_per_config=4)
    spec["sources"] = [spec["sources"][0]]
    spec["sources"][0]["configs"] = ["en"]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    output = tmp_path / "candidate"
    export(
        path,
        output,
        loader=lambda *args, **kwargs: (
            {
                "id": i,
                "query": f"query {i}",
                "positive": f"positive {i}",
                "negative": f"negative {i}",
            }
            for i in range(100)
        ),
    )
    return output


def test_quarantine_authenticates_and_blocks_both_identity_policies(tmp_path):
    directory = candidate(tmp_path)
    hashes, identity = prep.candidate_hashes(directory)
    raw = next(directory.glob("*.jsonl"))
    row = json.loads(raw.read_text().splitlines()[0])
    assert identity["raw_hashes"][raw.name] == prep.file_hash(raw)
    assert prep.excluded_text(row["positive"], "hq", hashes)
    assert prep.excluded_text(
        "  " + row["positive"].replace(" ", "\t") + " ", "hq", hashes
    )
    assert not prep.excluded_text("unrelated clean document", "hq", hashes)
    raw.write_text(raw.read_text() + "{}\n")
    with pytest.raises(ValueError, match="checksum"):
        prep.candidate_hashes(directory)


def test_historical_documents_authenticated_before_streaming_dedup(tmp_path):
    path = tmp_path / "documents.jsonl"
    digest = prep.normalized_digest("existing document")
    path.write_text(json.dumps({"text_sha256": digest}) + "\n")
    seen = sqlite3.connect(":memory:")
    seen.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
    with pytest.raises(ValueError, match="checksum"):
        prep.seed_prior_documents(seen, [path], ["0" * 64])
    assert seen.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 0
    receipts = prep.seed_prior_documents(seen, [path, path], [prep.file_hash(path)] * 2)
    assert len(receipts) == 2
    assert seen.execute("SELECT digest FROM seen").fetchall() == [(digest,)]
    with pytest.raises(ValueError, match="paired"):
        prep.seed_prior_documents(seen, [path], [])


def test_missing_or_false_historical_normalized_identity_rejected(tmp_path):
    path = tmp_path / "documents.jsonl"
    seen = sqlite3.connect(":memory:")
    seen.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
    for row, expected in [
        ({"id": "opaque"}, "missing"),
        ({"text_sha256": "0" * 64, "text": "changed text"}, "mismatch"),
    ]:
        path.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match=expected):
            prep.seed_prior_documents(seen, [path], [prep.file_hash(path)])


def test_raw_export_preserves_license_text_urls_and_original_content(tmp_path):
    row = {
        "text": '\tprint("hello")\n',
        "licenses": [{"spdx": "MIT"}],
        "license_text": "Copyright holder; permission granted...",
        "url": "https://example.org/file",
    }
    document = {"text": row["text"], "upstream_provenance": prep.provenance(row)}
    raw = tmp_path / "input.jsonl"
    raw.write_text(json.dumps(document) + "\n")
    receipt = prep.retain_raw(raw, tmp_path, 3, "train")
    output = tmp_path / receipt["path"]
    with gzip.open(output, "rt") as stream:
        assert json.loads(stream.readline()) == document
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    first = output.read_bytes()
    prep.retain_raw(raw, tmp_path, 3, "train")
    assert output.read_bytes() == first
    assert prep.provenance({})["license_provenance"] == "not-specified-in-upstream-row"


def test_atomic_status_has_timestamp_and_explicit_failure(tmp_path):
    prep.write_status(tmp_path, state="failed", error="deadline")
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "failed" and status["updated_unix"] > 0
    assert not (tmp_path / "status.tmp").exists()


def test_gzip_historical_legacy_hash_and_conflicts(tmp_path):
    path = tmp_path / "documents.jsonl.gz"
    digest = prep.normalized_digest("historical multilingual document")
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"normalized_text_sha256": digest}) + "\n")
    seen = sqlite3.connect(":memory:")
    seen.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
    receipt = prep.seed_prior_documents(seen, [path], [prep.file_hash(path)])[0]
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert seen.execute("SELECT digest FROM seen").fetchone()[0] == digest
    assert (
        prep.historical_digest(
            {"normalized_text_sha256": digest, "text_sha256": digest}
        )
        == digest
    )
    with pytest.raises(ValueError, match="conflicting"):
        prep.historical_digest(
            {"normalized_text_sha256": digest, "text_sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="checksum"):
        prep.seed_prior_documents(seen, [path], ["0" * 64])


def test_validation_minimum_blocks_qualification_not_prepared_data():
    train = {
        "language_counts": {
            "eng_Latn": {"documents": 100000},
            "code:Julia": {"documents": 20},
        }
    }
    validation = {"language_counts": {"eng_Latn": {"documents": 200}}}
    result = prep.validation_qualification(train, validation, 100)
    assert result["qualified"] is False
    assert result["deficits"] == {"code:Julia": 100}
    validation["language_counts"]["code:Julia"] = {"documents": 100}
    assert prep.validation_qualification(train, validation, 100)["qualified"] is True


def test_new_stage_split_is_deterministic_one_percent():
    assert prep.split_for_digest(f"{100:016x}" + "0" * 48) == "validation"
    assert prep.split_for_digest(f"{101:016x}" + "0" * 48) == "train"
    assert prep.split_for_digest(f"{100:016x}" + "0" * 48, 1000) == "train"
    with pytest.raises(ValueError, match="modulus"):
        prep.split_for_digest("0" * 64, 1)


def test_initial_reservation_is_per_label_disjoint_and_does_not_oversample():
    counts = {}
    # These identities hash to train; initial reservation intentionally overrides that.
    identities = [f"{i:016x}" + "0" * 48 for i in range(1, 5)]
    results = [
        prep.reserve_validation_split(value, "code:Julia", counts, 2)
        for value in identities
    ]
    assert results == ["validation", "validation", "train", "train"]
    assert counts == {"code:Julia": 2}
    assert (
        prep.reserve_validation_split(identities[0], "eng_Latn", counts, 2)
        == "validation"
    )
    assert counts == {"code:Julia": 2, "eng_Latn": 1}
    # Once reserved, ordinary hash-heldout rows stay held out without increasing the reservation.
    assert (
        prep.reserve_validation_split("0" * 64, "code:Julia", counts, 2) == "validation"
    )
    assert counts["code:Julia"] == 2
    assert len(results) == len(
        identities
    )  # Each admitted document emitted exactly once.


def test_reserved_only_language_cannot_qualify_without_training_examples():
    train = {"language_counts": {"eng_Latn": {"documents": 1}}}
    validation = {
        "language_counts": {
            "eng_Latn": {"documents": 100},
            "code:Rare": {"documents": 100},
        }
    }
    result = prep.validation_qualification(train, validation, 100)
    assert result["qualified"] is False and result["deficits"] == {}
    assert result["missing_training_labels"] == ["code:Rare"]


def test_all_development_candidates_are_authenticated_and_unioned(tmp_path):
    first = candidate(tmp_path)
    second_root = tmp_path / "second"
    second_root.mkdir()
    second = candidate(second_root)
    raw = next(second.glob("*.jsonl"))
    # Same candidates may repeat; union retains one identity namespace and both receipts.
    hashes, identities = prep.candidate_hashes_union([first, second])
    expected, identity = prep.candidate_hashes(first)
    assert hashes == expected and identities == [identity, identity]
    assert prep.candidate_hashes_union(first) == (expected, [identity])
    raw.write_text(raw.read_text() + "{}\n")
    with pytest.raises(ValueError, match="checksum"):
        prep.candidate_hashes_union([first, second])
    with pytest.raises(ValueError, match="at least one"):
        prep.candidate_hashes_union([])
