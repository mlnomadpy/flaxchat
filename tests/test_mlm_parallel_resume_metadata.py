"""Real tiny tokenization/resume corruption checks; no model or device execution."""

import gzip
import json
import sqlite3

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts import prepare_yat_mlm_parallel as parallel


def test_fetch_skips_committed_payload_without_collapsing_traversal(monkeypatch):
    """Fake pinned remote file only; no network, models or hardware."""
    import sys
    import types

    opened, decoded = [], []

    class Remote:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Filesystem:
        def open(self, path, *args, **kwargs):
            opened.append(path)
            return Remote()

    metadata = types.SimpleNamespace(
        row_group=lambda group: types.SimpleNamespace(
            num_columns=1,
            column=lambda column: types.SimpleNamespace(
                total_uncompressed_size=16, path_in_schema="text"
            ),
        )
    )

    class Parquet:
        num_row_groups = 3
        schema_arrow = types.SimpleNamespace(names=["text"])

        def __init__(self, remote):
            self.metadata = metadata

        def read_row_group(self, group, columns):
            decoded.append(group)
            return types.SimpleNamespace(
                to_pylist=lambda: [{"text": "group " + str(group)}]
            )

    hub = types.ModuleType("huggingface_hub")
    hub.HfFileSystem = Filesystem
    arrow = types.ModuleType("pyarrow")
    parquet = types.ModuleType("pyarrow.parquet")
    parquet.ParquetFile = Parquet
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    original = ("english", "eng_Latn", "data/file.parquet")
    # Old API still decodes descending first group.
    assert parallel.fetch_group((*original, None)) == (
        original,
        2,
        [0, 1],
        [{"text": "group 2"}],
    )
    assert decoded == [2]
    decoded.clear()
    # First skipped coordinate still reads footer once, never its payload.
    assert parallel.fetch_group((*original, None, {1, 2})) == (original, 2, [0, 1], [])
    assert not decoded
    opens_before = len(opened)
    # Subsequent committed coordinate avoids both footer reopen and decode.
    assert parallel.fetch_group((*original, [0, 1], {1, 2})) == (original, 1, [0], [])
    assert len(opened) == opens_before and not decoded
    assert parallel.fetch_group((*original, [0], {1, 2})) == (
        original,
        0,
        [],
        [{"text": "group 0"}],
    )
    assert decoded == [0]
    dataset, pin = parallel.base.SOURCES["english"]
    assert all(path == f"datasets/{dataset}@{pin}/data/file.parquet" for path in opened)
    with pytest.raises(ValueError, match="nonnegative integers"):
        parallel.fetch_group((*original, None, {True}))


def corpus(tmp_path):
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "hello": 5,
                "world": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    token_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(token_path))
    root = tmp_path / "corpus"
    root.mkdir()
    for split, text in [
        ("train", "hello world hello"),
        ("validation", "world hello world"),
    ]:
        identity = parallel.base.normalized_digest(text)
        row = dict(
            dataset=parallel.base.DATASET,
            revision="original",
            split=split,
            id=identity,
            text=text,
            language_script="eng_Latn",
            text_sha256=identity,
        )
        (root / f"raw-0-{split}.jsonl").write_text(json.dumps(row) + "\n")
    manifests, raw = parallel.tokenize_chunk(
        str(root), 0, {"train": 1, "validation": 1}, str(token_path), "original", 8
    )
    recipe = dict(
        sources=parallel.base.SOURCES,
        tokenizer_sha256=parallel.base.file_hash(token_path),
        sequence_length=8,
        minimum_validation_documents_per_label=100,
        raw_shards=raw,
        chunks=[
            dict(
                source_group="english",
                language="eng_Latn",
                selected_documents={"train": 1, "validation": 1},
                train_nonpadding_tokens=manifests["train"]["nonpadding_tokens"],
                validation_nonpadding_tokens=manifests["validation"][
                    "nonpadding_tokens"
                ],
            )
        ],
    )
    parallel.atomic_json(root / "recipe.partial.json", recipe)
    return root, token_path


def seen():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
    return connection


def test_resume_authenticates_actual_tokens_documents_and_retained_raw(tmp_path):
    root, tokenizer = corpus(tmp_path)
    connection = seen()
    recipe, shards, counters, reserved, revision = parallel.authenticate_resume(
        root, tokenizer, connection
    )
    assert revision == "original"
    assert counters == {("english", "eng_Latn"): 3}
    assert reserved == {"eng_Latn": 1}
    assert len(shards["train"]) == len(shards["validation"]) == 1
    assert connection.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 2
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "authenticating"
    assert status["verified_committed_chunks"] == status["total_committed_chunks"] == 1
    assert status["verified_train_tokens"] == 3
    with gzip.open(root / recipe["raw_shards"][0]["path"], "rt") as stream:
        assert json.loads(stream.readline())["text"] == "hello world hello"


@pytest.mark.parametrize(
    "filename",
    [
        "shard-0-train/tokens.npy",
        "shard-0-train/documents.jsonl",
        "raw/train/part-0000000.jsonl.gz",
    ],
)
def test_resume_rejects_changed_committed_bytes(tmp_path, filename):
    root, tokenizer = corpus(tmp_path)
    path = root / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        parallel.authenticate_resume(root, tokenizer, seen())


def test_resume_rejects_committed_overlap_with_history(tmp_path):
    root, tokenizer = corpus(tmp_path)
    connection = seen()
    connection.execute(
        "INSERT INTO seen VALUES (?)",
        (parallel.base.normalized_digest("hello world hello"),),
    )
    with pytest.raises(ValueError, match="duplicates history"):
        parallel.authenticate_resume(root, tokenizer, connection)


def test_resume_preserves_committed_shards_and_quarantines_interrupted_output(tmp_path):
    root, tokenizer = corpus(tmp_path)
    orphan = root / "shard-1-train"
    orphan.mkdir()
    (orphan / "unfinished").write_text("partial")
    (root / "raw-1-train.jsonl").write_text("in flight")
    parallel.authenticate_resume(root, tokenizer, seen())
    assert not orphan.exists()
    assert (root / "shard-0-train/tokens.npy").exists()
    assert len(list((tmp_path / "corpus-interrupted").iterdir())) == 2
