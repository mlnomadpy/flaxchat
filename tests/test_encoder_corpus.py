import json
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors, AddedToken
from flaxchat.encoder import EncoderConfig
from flaxchat.encoder_data import load_prepared_rows, file_hash
from scripts.prepare_encoder_corpus import prepare


def inputs(tmp_path):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4} | {
        c: i + 5 for i, c in enumerate("abcdefghi")
    }
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken(t, special=True) for t in list(vocab)[:5]])
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    rows = [
        dict(text="a b c d e f g h i", id="one", language_script="eng_Latn"),
        dict(text="a b c", id="two", language_script="fra_Latn"),
    ]
    rows = [r | dict(dataset="toy", revision="pinned", split="train") for r in rows]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(map(json.dumps, rows)))
    return source, path, rows


def run(source, tokenizer, out, **kwargs):
    return prepare(
        source,
        tokenizer,
        out,
        dataset="toy",
        revision="pinned",
        split="train",
        sequence_length=6,
        **kwargs,
    )


def test_complete_windows_preserve_tails_and_per_language_provenance(tmp_path):
    source, tok, _ = inputs(tmp_path)
    out = tmp_path / "out"
    report = run(source, tok, out)
    config = EncoderConfig(vocab_size=14, max_position_embeddings=16)
    rows, manifest = load_prepared_rows(out, config)
    assert report == manifest
    assert rows.shape == (4, 6)
    np.testing.assert_array_equal(rows[2], [2, 13, 3, 0, 0, 0])
    assert manifest["nonpadding_tokens"] == 20
    assert manifest["language_counts"]["eng_Latn"]["rows"] == 3
    assert manifest["language_counts"]["fra_Latn"]["rows"] == 1
    docs = [
        json.loads(line) for line in (out / "documents.jsonl").read_text().splitlines()
    ]
    assert [(r["first_row"], r["end_row"]) for r in docs] == [(0, 3), (3, 4)]
    assert all("text" not in r for r in docs)
    assert manifest["documents_sha256"] == file_hash(out / "documents.jsonl")


def test_row_limit_rejects_instead_of_dropping_documents(tmp_path):
    source, tok, _ = inputs(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="silent truncation"):
        run(source, tok, out, max_rows=3)
    assert not out.exists() and not list(tmp_path.glob(".corpus-*"))


@pytest.mark.parametrize(
    "defect", ["revision", "split", "language", "duplicate", "empty"]
)
def test_bad_document_identity_leaves_no_partial_corpus(tmp_path, defect):
    source, tok, rows = inputs(tmp_path)
    if defect == "revision":
        rows[1]["revision"] = "changed"
    if defect == "split":
        rows[1]["split"] = "validation"
    if defect == "language":
        rows[1]["language_script"] = ""
    if defect == "duplicate":
        rows[1]["id"] = "one"
    if defect == "empty":
        rows[1]["text"] = ""
    source.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        run(source, tok, tmp_path / "out")
    assert not (tmp_path / "out").exists()
