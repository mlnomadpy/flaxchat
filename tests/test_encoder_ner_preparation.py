import json
from dataclasses import replace
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from flaxchat.encoder import EncoderConfig
from flaxchat.encoder_data import file_hash
from scripts.prepare_encoder_ner import prepare, load_rows


def fixture(tmp_path, rows=None, length=8):
    tok = Tokenizer(
        models.WordPiece(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "play": 5,
                "##ing": 6,
                "Tokyo": 7,
            },
            unk_token="[UNK]",
        )
    )
    tok.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = tmp_path / "tokenizer.json"
    tok.save(str(tokenizer))
    source = tmp_path / "source.jsonl"
    rows = (
        rows
        if rows is not None
        else [{"tokens": ["playing", "Tokyo"], "ner_tags": [0, 1]}]
    )
    source.write_text("\n".join(json.dumps(row) for row in rows))
    kwargs = dict(
        dataset="test/ner",
        revision="pinned",
        split="validation",
        language="en",
        label_names=["O", "B-LOC", "I-LOC"],
        sequence_length=length,
        max_rows=4,
    )
    return source, tokenizer, tmp_path / "prepared", kwargs


def test_roundtrip_word_and_label_provenance(tmp_path):
    source, tok, out, kwargs = fixture(tmp_path)
    report = prepare(source, tok, out, **kwargs)
    config = EncoderConfig(vocab_size=8, max_position_embeddings=16)
    tokens, labels, word_ids, manifest = load_rows(
        out, config, expected_split="validation"
    )
    assert report == manifest and report["words"] == 2
    assert manifest["source_sha256"] == file_hash(source)
    np.testing.assert_array_equal(tokens, [[2, 5, 6, 7, 3, 0, 0, 0]])
    np.testing.assert_array_equal(labels, [[-100, 0, -100, 1, -100, -100, -100, -100]])
    np.testing.assert_array_equal(word_ids, [[-1, 0, 0, 1, -1, -1, -1, -1]])
    with pytest.raises(ValueError, match="Wrong official split"):
        load_rows(out, config, expected_split="train")
    with pytest.raises(ValueError, match="vocabulary"):
        load_rows(out, replace(config, vocab_size=9))
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, tok, out, **kwargs)


@pytest.mark.parametrize(
    "rows,length",
    [
        ([{"tokens": ["Tokyo", "playing"], "ner_tags": [0, 1]}], 4),
        ([{"tokens": ["playing"], "ner_tags": []}], 8),
        ([{"tokens": ["Tokyo"], "ner_tags": [9]}], 8),
        ([{"tokens": [""], "ner_tags": [0]}], 8),
        ([{"tokens": ["unknown"], "ner_tags": [0]}], 8),
        ([], 8),
    ],
)
def test_invalid_or_truncated_input_is_never_published(tmp_path, rows, length):
    source, tok, out, kwargs = fixture(tmp_path, rows, length)
    with pytest.raises(ValueError):
        prepare(source, tok, out, **kwargs)
    assert not out.exists()
    assert not list(tmp_path.glob(".ner-*"))


def test_semantic_alignment_tampering_rejected_even_with_rehashed_array(tmp_path):
    source, tok, out, kwargs = fixture(tmp_path)
    prepare(source, tok, out, **kwargs)
    labels = np.load(out / "labels.npy")
    labels[0, 2] = 2
    np.save(out / "labels.npy", labels)
    config = EncoderConfig(vocab_size=8, max_position_embeddings=16)
    with pytest.raises(ValueError, match="checksum"):
        load_rows(out, config)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["labels_sha256"] = file_hash(out / "labels.npy")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_rows(out, config)


def test_max_rows_rejects_partial_dataset(tmp_path):
    source, tok, out, kwargs = fixture(
        tmp_path, [{"tokens": ["Tokyo"], "ner_tags": [1]}] * 5
    )
    with pytest.raises(ValueError, match="max-rows"):
        prepare(source, tok, out, **kwargs)
    assert not out.exists()


def test_source_change_during_preparation_rejects_publication(tmp_path, monkeypatch):
    from scripts import prepare_encoder_ner as preparation

    source, tok, out, kwargs = fixture(tmp_path)
    original = preparation.file_hash
    reads = 0

    def changing_hash(path):
        nonlocal reads
        if path == source:
            reads += 1
            if reads > 1:
                return "changed"
        return original(path)

    monkeypatch.setattr(preparation, "file_hash", changing_hash)
    with pytest.raises(ValueError, match="changed during"):
        preparation.prepare(source, tok, out, **kwargs)
    assert not out.exists()
