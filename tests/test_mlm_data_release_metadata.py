"""Cloud data-worker contracts without model execution or cloud credentials."""

import hashlib
import json
import subprocess

import pytest

from scripts.run_mlm_data_release_gce import (
    authenticated_heldout_index,
    best_effort_upload,
)
from scripts.prepare_representation_v2_training import PAIR_SOURCE
from scripts.prepare_yat_mlm_continuation import normalized_digest


def heldouts(tmp_path):
    development, supplement = tmp_path / "development", tmp_path / "supplement"
    history = development / "historical"
    for name in (
        "msmarco",
        "code",
        "miracl",
        PAIR_SOURCE,
        "contrastive_second",
        "contrastive_third",
    ):
        directory = history / name
        directory.mkdir(parents=True)
        target = supplement / name / "dev.jsonl"
        target.parent.mkdir(parents=True)
        row = (
            {"sentence1": name + " left", "sentence2": name + " right"}
            if name.startswith("contrastive_")
            else {"query": name + " query", "positive": name + " document"}
        )
        target.write_text(json.dumps(row) + "\n")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = (
            {"files": {"dev.jsonl": {"sha256": digest}}}
            if name.startswith("contrastive_")
            else {"raw_files": {"dev.jsonl": digest}}
        )
        (directory / "manifest.json").write_text(json.dumps(manifest))
    return development, supplement


def test_index_covers_authenticated_pairs_and_retrieval(tmp_path):
    development, supplement = heldouts(tmp_path)
    output = tmp_path / "index.jsonl"
    inventory = authenticated_heldout_index(development, supplement, output)
    assert len(inventory) == 6
    assert sum(item["texts"] for item in inventory) == 12
    hashes = {
        json.loads(line)["text_sha256"] for line in output.read_text().splitlines()
    }
    assert normalized_digest(PAIR_SOURCE + " left") in hashes
    assert normalized_digest("msmarco query") in hashes


def test_incomplete_or_mutated_heldout_refused(tmp_path):
    development, supplement = heldouts(tmp_path)
    target = supplement / "code" / "dev.jsonl"
    target.write_text("{}\n")
    with pytest.raises(ValueError, match="bytes differ"):
        authenticated_heldout_index(development, supplement, tmp_path / "index.jsonl")
    target.unlink()
    with pytest.raises(ValueError, match="Incomplete historical"):
        authenticated_heldout_index(development, supplement, tmp_path / "index.jsonl")


def test_upload_failure_does_not_abort_preparation():
    def fail(argv, seconds):
        raise subprocess.CalledProcessError(1, argv)

    assert not best_effort_upload(fail, ["gcloud"], 1)
    assert best_effort_upload(lambda argv, seconds: None, ["gcloud"], 1)


def test_publication_retry_excludes_own_inventory_and_reuses_verified_tag(tmp_path):
    from types import SimpleNamespace
    from scripts.run_mlm_data_release_gce import publish_dataset, sha

    folder = tmp_path / "release"
    folder.mkdir()
    (folder / "data.jsonl").write_text('{"text":"fixture"}\n')
    files = {
        "data.jsonl": {
            "sha256": sha(folder / "data.jsonl"),
            "bytes": (folder / "data.jsonl").stat().st_size,
        }
    }
    (folder / "files.json").write_text(json.dumps(files, indent=2) + "\n")

    class Api:
        def repo_exists(self, *args, **kwargs):
            return True

        def list_repo_refs(self, *args, **kwargs):
            return SimpleNamespace(
                tags=[SimpleNamespace(name="v2", target_commit="immutable")]
            )

        def dataset_info(self, *args, **kwargs):
            assert kwargs["revision"] == "immutable"
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename=p.name, size=p.stat().st_size, lfs=None)
                    for p in folder.iterdir()
                ]
            )

        def upload_folder(self, **kwargs):
            raise AssertionError("Verified tagged publication must not be rewritten")

    def download(repo, name, **kwargs):
        return folder / name

    for _ in range(2):
        result = publish_dataset(
            folder, "fixture/repo", "v2", Api(), download, "fixture-token"
        )
        assert result["revision"] == "immutable"
        assert result["files_verified"] == 1
        assert json.loads((folder / "files.json").read_text()) == files


def test_publication_rejects_partial_corpus_despite_complete_flag(tmp_path):
    from scripts.run_mlm_data_release_gce import authenticate_completed_corpus, sha

    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    artifacts = {
        "recipe_sha256": (
            "recipe.json",
            {
                "target_nonpadding_tokens": 10_000_000_000,
                "actual_train_nonpadding_tokens": 3_000_000_000,
            },
        ),
        "train_manifest_sha256": (
            "train/manifest.json",
            {"nonpadding_tokens": 3_000_000_000},
        ),
        "validation_manifest_sha256": ("validation/manifest.json", {}),
        "qualification_sha256": ("QUALIFICATION.json", {}),
    }
    for split in ("train", "validation"):
        for name in ("tokens.npy", "documents.jsonl"):
            (tmp_path / split / name).write_bytes(b"fixture")
        manifest = artifacts[split + "_manifest_sha256"][1]
        manifest.update(
            tokens_sha256=sha(tmp_path / split / "tokens.npy"),
            documents_sha256=sha(tmp_path / split / "documents.jsonl"),
        )
    marker = {"training_qualified": True}
    for key, (name, value) in artifacts.items():
        (tmp_path / name).write_text(json.dumps(value))
        marker[key] = sha(tmp_path / name)
    (tmp_path / "COMPLETE.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="partial"):
        authenticate_completed_corpus(tmp_path)
