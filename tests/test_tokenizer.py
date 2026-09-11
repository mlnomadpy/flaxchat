"""
Tests for flaxchat/tokenizer.py — HuggingFaceTokenizer training and encoding.

Uses shakespeare_text fixture for realistic BPE training on small data.
RustBPETokenizer is not tested here (requires rustbpe which may not be
available in CI).
"""

import os
import pytest

from scripts import tok_train
from flaxchat.tokenizer import (
    BYTE_TOKENIZER_FILENAME,
    ByteTokenizer,
    HuggingFaceTokenizer,
    SPECIAL_TOKENS,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_tokenizer(shakespeare_text):
    """Train a small BPE tokenizer on Shakespeare text."""
    # Use a very small vocab for fast tests
    lines = shakespeare_text.split("\n")
    tok = HuggingFaceTokenizer.train_from_iterator(iter(lines), vocab_size=512)
    return tok


@pytest.fixture
def small_tokenizer(shakespeare_text):
    """Train a tiny tokenizer from shakespeare_text fixture."""
    lines = shakespeare_text.split("\n")[:500]
    return HuggingFaceTokenizer.train_from_iterator(iter(lines), vocab_size=300)


# ---------------------------------------------------------------------------
# Tests for train_from_iterator
# ---------------------------------------------------------------------------

class TestTrainFromIterator:
    def test_vocab_size(self, trained_tokenizer):
        """Requested vocab size is an upper bound for a finite corpus."""
        size = trained_tokenizer.get_vocab_size()
        assert len(SPECIAL_TOKENS) < size <= 512

    def test_special_tokens_present(self, trained_tokenizer):
        """All special tokens should be in the trained vocabulary."""
        special = trained_tokenizer.get_special_tokens()
        for token in SPECIAL_TOKENS:
            assert token in special, f"Missing special token: {token}"

    def test_special_tokens_have_ids(self, trained_tokenizer):
        """Each special token should map to a valid integer ID."""
        for token in SPECIAL_TOKENS:
            token_id = trained_tokenizer.encode_special(token)
            assert isinstance(token_id, int)
            assert token_id >= 0


# ---------------------------------------------------------------------------
# Tests for encode / decode
# ---------------------------------------------------------------------------

class TestEncodeDecode:
    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        """Encoding a string should return a list of integers."""
        ids = trained_tokenizer.encode("Hello, world!")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_roundtrip(self, trained_tokenizer):
        """Encoding then decoding should recover the original text."""
        text = "To be, or not to be, that is the question."
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_encode_batch(self, trained_tokenizer):
        """Encoding a list of strings should return a list of lists."""
        texts = ["Hello", "World", "Foo bar"]
        results = trained_tokenizer.encode(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)
            assert all(isinstance(i, int) for i in r)

    def test_encode_with_prepend(self, trained_tokenizer):
        """Encoding with prepend should insert the BOS token at position 0."""
        bos_id = trained_tokenizer.get_bos_token_id()
        ids = trained_tokenizer.encode("test", prepend=bos_id)
        assert ids[0] == bos_id

    def test_encode_with_append(self, trained_tokenizer):
        """Encoding with append should add the token at the end."""
        bos_id = trained_tokenizer.get_bos_token_id()
        ids = trained_tokenizer.encode("test", append=bos_id)
        assert ids[-1] == bos_id

    def test_encode_empty_string(self, trained_tokenizer):
        """Encoding an empty string should return an empty list."""
        ids = trained_tokenizer.encode("")
        assert isinstance(ids, list)
        assert len(ids) == 0

    def test_encode_invalid_type_raises(self, trained_tokenizer):
        """Encoding a non-string/non-list should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid input type"):
            trained_tokenizer.encode(12345)

    def test_callable(self, trained_tokenizer):
        """Tokenizer should be callable (delegates to encode)."""
        ids = trained_tokenizer("Hello")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_unicode_roundtrip(self, trained_tokenizer):
        """BPE with byte_fallback should handle Unicode via byte fallback."""
        text = "Caf\u00e9 \u2603 \u00fc\u00f6\u00e4"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text


# ---------------------------------------------------------------------------
# Tests for special tokens
# ---------------------------------------------------------------------------

class TestSpecialTokens:
    def test_bos_token_id(self, trained_tokenizer):
        """get_bos_token_id should return the ID of <|bos|>."""
        bos_id = trained_tokenizer.get_bos_token_id()
        assert isinstance(bos_id, int)
        assert bos_id == trained_tokenizer.encode_special("<|bos|>")

    def test_encode_special_returns_int(self, trained_tokenizer):
        """encode_special should return an integer for known special tokens."""
        for token in SPECIAL_TOKENS:
            tid = trained_tokenizer.encode_special(token)
            assert isinstance(tid, int), f"encode_special({token!r}) returned {type(tid)}"

    def test_id_to_token(self, trained_tokenizer):
        """id_to_token should return a string for valid IDs."""
        for i in range(min(10, trained_tokenizer.get_vocab_size())):
            token = trained_tokenizer.id_to_token(i)
            assert isinstance(token, str)


class TestConversationRendering:
    def test_huggingface_backend_marks_only_assistant_targets(
        self, trained_tokenizer
    ):
        conversation = {
            "messages": [
                {"role": "user", "content": "Tell me a story."},
                {"role": "assistant", "content": "Once upon a time."},
            ]
        }
        ids, mask = trained_tokenizer.render_conversation(conversation)
        assert len(ids) == len(mask)
        assert any(mask)
        first_target = mask.index(1)
        assert all(value == 0 for value in mask[:first_target])
        assert ids[-1] == trained_tokenizer.encode_special("<|assistant_end|>")

    def test_completion_removes_reference_answer(self, trained_tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "Continue."},
                {"role": "assistant", "content": "Secret answer."},
            ]
        }
        ids = trained_tokenizer.render_for_completion(conversation)
        assert ids[-1] == trained_tokenizer.encode_special("<|assistant_start|>")
        assert "Secret answer" not in trained_tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Tests for save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_reload(self, trained_tokenizer, tmp_path):
        """Saving and reloading should produce identical encode results."""
        tok_dir = str(tmp_path / "tokenizer")
        trained_tokenizer.save(tok_dir)

        loaded = HuggingFaceTokenizer.from_directory(tok_dir)

        text = "Shall I compare thee to a summer's day?"
        original_ids = trained_tokenizer.encode(text)
        loaded_ids = loaded.encode(text)
        assert original_ids == loaded_ids

    def test_save_creates_file(self, trained_tokenizer, tmp_path):
        """save() should create tokenizer.json in the target directory."""
        tok_dir = str(tmp_path / "tok_out")
        trained_tokenizer.save(tok_dir)
        assert os.path.exists(os.path.join(tok_dir, "tokenizer.json"))

    def test_loaded_vocab_size_matches(self, trained_tokenizer, tmp_path):
        """Loaded tokenizer should have the same vocab size."""
        tok_dir = str(tmp_path / "tok_out")
        trained_tokenizer.save(tok_dir)
        loaded = HuggingFaceTokenizer.from_directory(tok_dir)
        assert loaded.get_vocab_size() == trained_tokenizer.get_vocab_size()


class TestByteTokenizer:
    def test_uses_byt5_byte_id_mapping(self):
        tokenizer = ByteTokenizer()
        assert tokenizer.encode("A") == [ord("A") + 3]
        assert tokenizer.encode("\x00\xff") == [3, 198, 194]
        assert tokenizer.id_to_token(0) == "<pad>"
        assert tokenizer.id_to_token(1) == "</s>"
        assert tokenizer.id_to_token(2) == "<unk>"

    def test_unicode_roundtrip(self):
        tokenizer = ByteTokenizer()
        text = "Café ☃ ภาษาไทย 中文 🚀"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_batch_prepend_and_append(self):
        tokenizer = ByteTokenizer()
        bos = tokenizer.get_bos_token_id()
        eos = tokenizer.encode_special("</s>")
        rows = tokenizer.encode(["a", "é"], prepend=bos, append=eos)
        assert all(row[0] == bos and row[-1] == eos for row in rows)
        assert [tokenizer.decode(row[1:-1]) for row in rows] == ["a", "é"]

    def test_vocab_contains_bytes_reserved_and_chat_tokens(self):
        tokenizer = ByteTokenizer()
        assert tokenizer.get_vocab_size() == 3 + 256 + len(SPECIAL_TOKENS)
        assert tokenizer.encode_special("<|bos|>") == 259
        assert tokenizer.encode_special("<|assistant_end|>") == 263
        assert tokenizer.get_special_tokens() == [
            "<pad>", "</s>", "<unk>", *SPECIAL_TOKENS
        ]

    def test_special_tokens_are_explicit_not_parsed_from_text(self):
        tokenizer = ByteTokenizer()
        literal = "<|bos|>"
        assert tokenizer.encode(literal) != [tokenizer.get_bos_token_id()]
        assert tokenizer.decode(tokenizer.encode(literal)) == literal

    def test_conversation_rendering(self):
        tokenizer = ByteTokenizer()
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello 👋"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        assert len(ids) == len(mask)
        assert ids[0] == tokenizer.get_bos_token_id()
        assert ids[-1] == tokenizer.encode_special("<|assistant_end|>")
        assert any(mask)

    def test_save_load_and_factory_roundtrip(self, tmp_path):
        tokenizer = ByteTokenizer()
        tokenizer.save(tmp_path)
        assert (tmp_path / BYTE_TOKENIZER_FILENAME).is_file()
        loaded = load_tokenizer(tmp_path)
        assert isinstance(loaded, ByteTokenizer)
        text = "raw bytes survive 💾"
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == text

    def test_stream_decode_preserves_multibyte_characters(self):
        tokenizer = ByteTokenizer()
        assert "".join(tokenizer.decode_stream(tokenizer.encode("Hi 👋"))) == "Hi 👋"

    def test_rejects_unknown_special_and_invalid_id(self):
        tokenizer = ByteTokenizer()
        with pytest.raises(ValueError, match="Unknown special token"):
            tokenizer.encode_special("<|missing|>")
        with pytest.raises(ValueError, match="out of range"):
            tokenizer.id_to_token(tokenizer.get_vocab_size())

    def test_training_cli_needs_no_dataset(self, tmp_path, monkeypatch):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("byte backend attempted to download training data")

        monkeypatch.setattr(tok_train, "download_shards", forbidden)
        monkeypatch.setattr(tok_train, "get_base_dir", lambda: str(tmp_path))
        assert tok_train.main(["--backend", "byte"]) == 0
        loaded = load_tokenizer(tmp_path / "tokenizer")
        assert isinstance(loaded, ByteTokenizer)
