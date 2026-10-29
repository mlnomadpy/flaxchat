"""Data-only parity checks for native batched tokenization and provenance."""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from scripts import prepare_encoder_corpus as preparation


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
                "世界": 6,
                "مرحبا": 7,
                "def": 8,
                "return": 9,
                "x": 10,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    texts = [
        ("en", "hello " * 37),
        ("zh", "世界 世界"),
        ("ar", "مرحبا hello"),
        ("Python", "def x():\n return x " * 19),
        ("en", "hello"),
    ]
    rows = [
        dict(
            dataset="fixture",
            revision="pinned",
            split="train",
            id=str(i),
            text=text,
            language_script=language,
            provenance={"row": i},
        )
        for i, (language, text) in enumerate(texts)
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    return source, path, rows


class SerialTokenizer:
    """Use the previous per-document native encode operation as the oracle."""

    calls = []

    @staticmethod
    def from_file(path):
        delegate = Tokenizer.from_file(path)

        class Adapter:
            def __getattr__(self, key):
                return getattr(delegate, key)

            def encode_batch(self, texts, **kwargs):
                SerialTokenizer.calls.append(len(texts))
                return [delegate.encode(text, **kwargs) for text in texts]

        return Adapter()


@pytest.mark.parametrize(
    "batch_size,batch_max_chars", [(1, 100000), (2, 100000), (64, 100000), (64, 1)]
)
def test_batch_matches_serial_tokens_windows_and_provenance(
    tmp_path, monkeypatch, batch_size, batch_max_chars
):
    source, tokenizer, _ = corpus(tmp_path)
    kwargs = dict(
        dataset="fixture",
        revision="pinned",
        split="train",
        sequence_length=8,
        max_rows=1000,
        batch_size=batch_size,
        batch_max_chars=batch_max_chars,
    )
    actual = preparation.prepare(source, tokenizer, tmp_path / "batched", **kwargs)
    monkeypatch.setattr(preparation, "Tokenizer", SerialTokenizer)
    expected = preparation.prepare(source, tokenizer, tmp_path / "serial", **kwargs)
    assert actual == expected
    for name in ("tokens.npy", "documents.jsonl", "manifest.json"):
        assert (tmp_path / "batched" / name).read_bytes() == (
            tmp_path / "serial" / name
        ).read_bytes()
    tokens = np.load(tmp_path / "batched/tokens.npy")
    assert (
        len(tokens) > 5
    )  # Long multilingual/code documents keep every overflow window.
    assert actual["documents"] == 5
    assert tokens[0].tolist() == [2, 5, 5, 5, 5, 5, 5, 3]


def test_duplicate_across_batches_is_rejected_atomically(tmp_path):
    source, tokenizer, rows = corpus(tmp_path)
    with source.open("a") as stream:
        stream.write(json.dumps(rows[0]) + "\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Duplicate document ID"):
        preparation.prepare(
            source,
            tokenizer,
            output,
            dataset="fixture",
            revision="pinned",
            split="train",
            batch_size=2,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".corpus-*"))


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"batch_size": True}, {"batch_max_chars": 0}]
)
def test_invalid_batch_limits(tmp_path, kwargs):
    source, tokenizer, _ = corpus(tmp_path)
    with pytest.raises(ValueError):
        preparation.prepare(
            source,
            tokenizer,
            tmp_path / "output",
            dataset="fixture",
            revision="pinned",
            split="train",
            **kwargs,
        )


def test_character_limit_bounds_native_batches(tmp_path, monkeypatch):
    source, tokenizer, _ = corpus(tmp_path)
    monkeypatch.setattr(preparation, "Tokenizer", SerialTokenizer)
    SerialTokenizer.calls = []
    preparation.prepare(
        source,
        tokenizer,
        tmp_path / "output",
        dataset="fixture",
        revision="pinned",
        split="train",
        batch_size=64,
        batch_max_chars=1,
    )
    assert SerialTokenizer.calls == [1] * 5


def test_large_documents_postprocess_only_flat_windows_and_match_original(
    tmp_path, monkeypatch
):
    """Exercise the 200k-character admission boundary without splitting the text."""
    source, tokenizer, rows = corpus(tmp_path)
    rows[0]["text"] = ("hello " * 33334)[:200000]
    rows[1]["text"] = "世界 " * 20000
    rows[3]["text"] = ("def x():\n return x " * 12000)[:200000]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    checked = []

    class FlatOnlyTokenizer:
        @staticmethod
        def from_file(path):
            delegate = Tokenizer.from_file(path)

            class Adapter:
                def __getattr__(self, key):
                    return getattr(delegate, key)

                def post_process(self, encoding):
                    assert not encoding.overflowing
                    assert len(encoding.ids) <= 510
                    checked.append(len(encoding.ids))
                    return delegate.post_process(encoding)

            return Adapter()

    kwargs = dict(
        dataset="fixture",
        revision="pinned",
        split="train",
        sequence_length=512,
        batch_size=16,
        batch_max_chars=262144,
    )
    monkeypatch.setattr(preparation, "Tokenizer", FlatOnlyTokenizer)
    actual = preparation.prepare(source, tokenizer, tmp_path / "flat", **kwargs)
    assert len(checked) > 100
    monkeypatch.setattr(preparation, "Tokenizer", Tokenizer)

    def original_windows(encoding, content_length):
        encoding.truncate(content_length)
        yield from [encoding, *encoding.overflowing]

    monkeypatch.setattr(preparation, "_document_windows", original_windows)
    expected = preparation.prepare(source, tokenizer, tmp_path / "original", **kwargs)
    assert actual == expected
    for name in ("tokens.npy", "documents.jsonl", "manifest.json"):
        assert (tmp_path / "flat" / name).read_bytes() == (
            tmp_path / "original" / name
        ).read_bytes()


def test_detached_window_preserves_native_metadata_and_postprocessing(tmp_path):
    from tokenizers import trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    text = "hello 世界 مرحبا\n def x(): return x; " * 200
    tokenizer.train_from_iterator(
        [text],
        trainers.BpeTrainer(
            vocab_size=256,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        ),
    )
    tokenizer.post_processor = processors.RobertaProcessing(
        ("[SEP]", 3), ("[CLS]", 2), trim_offsets=True, add_prefix_space=True
    )
    original = tokenizer.encode(text, add_special_tokens=False)
    original.truncate(30)
    expected_windows = [original, *original.overflowing]
    actual_windows = list(
        preparation._document_windows(
            tokenizer.encode(text, add_special_tokens=False), 30
        )
    )
    assert len(actual_windows) == len(expected_windows) > 10
    for actual, expected in zip(actual_windows, expected_windows, strict=True):
        expected_state = json.loads(expected.__getstate__())
        expected_state["overflowing"] = []
        assert json.loads(actual.__getstate__()) == expected_state
        actual_processed = tokenizer.post_process(actual)
        expected_processed = tokenizer.post_process(expected)
        assert actual_processed.ids == expected_processed.ids
        assert actual_processed.offsets == expected_processed.offsets
        assert actual_processed.word_ids == expected_processed.word_ids
        assert not actual_processed.overflowing
