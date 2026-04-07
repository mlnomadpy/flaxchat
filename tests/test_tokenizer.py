"""
Tests for flaxchat/tokenizer.py — HuggingFaceTokenizer training and encoding.

Uses shakespeare_text fixture for realistic BPE training on small data.
RustBPETokenizer is not tested here (requires rustbpe which may not be
available in CI).
"""

import os
import pytest

from flaxchat.tokenizer import HuggingFaceTokenizer, SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory, shakespeare_text=None):
    """Train a small BPE tokenizer on Shakespeare text."""
    # We can't use the session-scoped shakespeare_text directly in a
    # module-scoped fixture, so we do inline download/cache.
    import urllib.request
    cache_dir = tmp_path_factory.mktemp("tok_cache")
    cache_path = str(cache_dir / "shakespeare.txt")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, cache_path)
    with open(cache_path, "r") as f:
        text = f.read()

    # Use a very small vocab for fast tests
    lines = text.split("\n")
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
        """Trained tokenizer should have the requested vocab size."""
        assert trained_tokenizer.get_vocab_size() == 512

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
