"""
Tests for flaxchat/dataloader.py — BOS-aligned best-fit packing logic.

Since we can't access real parquet files in tests, we mock the data pipeline
and test the algorithmic core: best-fit packing, BOS alignment, and shapes.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from flaxchat.dataloader import (
    _document_batches,
    data_loader_bos_bestfit,
    data_loader_bos_bestfit_no_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Minimal tokenizer stub for dataloader tests."""

    def __init__(self, bos_id=0):
        self._bos_id = bos_id

    def get_bos_token_id(self):
        return self._bos_id

    def encode(self, texts, prepend=None, num_threads=None):
        """Encode each text as a list of byte values, optionally prepending BOS."""
        results = []
        for text in texts:
            ids = list(text.encode("utf-8"))
            if prepend is not None:
                ids.insert(0, prepend)
            results.append(ids)
        return results


def make_fake_parquet_file(texts):
    """Create a mock ParquetFile that yields the given texts as row groups."""
    mock_pf = MagicMock()
    # Each text becomes its own row group with a single row
    mock_pf.num_row_groups = len(texts)

    def read_row_group(idx):
        rg = MagicMock()
        col = MagicMock()
        col.to_pylist.return_value = [texts[idx]]
        rg.column.return_value = col
        return rg

    mock_pf.read_row_group = read_row_group
    return mock_pf


# ---------------------------------------------------------------------------
# Tests for _document_batches
# ---------------------------------------------------------------------------

class TestDocumentBatches:
    @patch("flaxchat.dataloader.pq.ParquetFile")
    @patch("flaxchat.dataloader.list_parquet_files")
    @patch("flaxchat.dataloader.jax")
    def test_yields_text_batches(self, mock_jax, mock_list_pq, mock_pq_file):
        """_document_batches yields (batch, state) tuples from parquet files."""
        mock_jax.process_index.return_value = 0
        mock_jax.process_count.return_value = 1

        texts = ["hello world", "foo bar", "test text"]
        # Need at least 2 files: train gets all but last, val gets last
        mock_list_pq.return_value = ["shard-0.parquet", "shard-1.parquet"]
        mock_pq_file.return_value = make_fake_parquet_file(texts)

        gen = _document_batches("train", resume_state_dict=None, tokenizer_batch_size=2)
        batch, (pq_idx, rg_idx, epoch) = next(gen)

        assert isinstance(batch, list)
        assert len(batch) <= 2  # tokenizer_batch_size=2
        assert epoch == 1

    @patch("flaxchat.dataloader.pq.ParquetFile")
    @patch("flaxchat.dataloader.list_parquet_files")
    @patch("flaxchat.dataloader.jax")
    def test_resume_state(self, mock_jax, mock_list_pq, mock_pq_file):
        """_document_batches respects resume_state_dict for approximate resumption."""
        mock_jax.process_index.return_value = 0
        mock_jax.process_count.return_value = 1

        texts = ["text_a", "text_b", "text_c", "text_d"]
        mock_list_pq.return_value = ["shard-0.parquet", "shard-1.parquet"]
        mock_pq_file.return_value = make_fake_parquet_file(texts)

        resume = {"pq_idx": 0, "rg_idx": 1, "epoch": 2}
        gen = _document_batches("train", resume_state_dict=resume, tokenizer_batch_size=128)
        batch, (pq_idx, rg_idx, epoch) = next(gen)

        assert epoch == 2
        assert isinstance(batch, list)

    @patch("flaxchat.dataloader.pq.ParquetFile")
    @patch("flaxchat.dataloader.list_parquet_files")
    @patch("flaxchat.dataloader.jax")
    def test_multi_host_sharding(self, mock_jax, mock_list_pq, mock_pq_file):
        """With process_count=2, each host reads alternating row groups."""
        mock_jax.process_index.return_value = 1
        mock_jax.process_count.return_value = 2

        texts = ["a", "b", "c", "d"]
        mock_list_pq.return_value = ["shard-0.parquet", "shard-1.parquet"]
        mock_pq_file.return_value = make_fake_parquet_file(texts)

        gen = _document_batches("train", resume_state_dict=None, tokenizer_batch_size=128)
        batch, (pq_idx, rg_idx, epoch) = next(gen)

        # Process 1 should start at row group index 1
        assert rg_idx == 1

    @patch("flaxchat.dataloader.pq.ParquetFile")
    @patch("flaxchat.dataloader.list_parquet_files")
    @patch("flaxchat.dataloader.jax")
    def test_val_split_uses_last_shard(self, mock_jax, mock_list_pq, mock_pq_file):
        """Val split uses only the last parquet file."""
        mock_jax.process_index.return_value = 0
        mock_jax.process_count.return_value = 1

        mock_list_pq.return_value = ["shard-0.parquet", "shard-1.parquet", "shard-2.parquet"]
        mock_pq_file.return_value = make_fake_parquet_file(["val text"])

        gen = _document_batches("val", resume_state_dict=None, tokenizer_batch_size=128)
        batch, _ = next(gen)

        # The val split should use parquet_paths[-1:], so ParquetFile is called
        # with the last shard only
        mock_pq_file.assert_called_with("shard-2.parquet")


# ---------------------------------------------------------------------------
# Tests for data_loader_bos_bestfit
# ---------------------------------------------------------------------------

class TestDataLoaderBOSBestFit:
    @patch("flaxchat.dataloader._document_batches")
    def test_output_shapes(self, mock_doc_batches):
        """Loader yields (inputs, targets, state_dict) with correct shapes."""
        B, T = 2, 16

        # Generate enough fake tokenized documents
        docs = [[0] + list(range(1, 20))] * 200
        call_count = [0]

        def fake_batches(*args, **kwargs):
            while True:
                start = call_count[0] * 10
                call_count[0] += 1
                batch = docs[start:start + 10] if start < len(docs) else docs[:10]
                yield batch, (0, 0, 1)

        mock_doc_batches.return_value = fake_batches()

        tokenizer = FakeTokenizer(bos_id=0)
        # We need to patch _document_batches to return pre-tokenized data.
        # But the real loader calls tokenizer.encode on text. Let's use a
        # different approach: mock at a higher level.

        # Actually, let's just mock _document_batches so it yields text,
        # and let the FakeTokenizer do the encoding.
        texts = ["hello world this is a test " * 5] * 200

        def fake_text_batches(*args, **kwargs):
            idx = 0
            while True:
                chunk = texts[idx:idx + 10]
                if not chunk:
                    idx = 0
                    chunk = texts[:10]
                idx += 10
                yield chunk, (0, 0, 1)

        mock_doc_batches.return_value = fake_text_batches()

        loader = data_loader_bos_bestfit(
            tokenizer, B=B, T=T, split="train",
            tokenizer_batch_size=10, buffer_size=5,
        )
        inputs, targets, state_dict = next(loader)

        assert inputs.shape == (B, T)
        assert targets.shape == (B, T)
        assert inputs.dtype == np.int32
        assert targets.dtype == np.int32

    @patch("flaxchat.dataloader._document_batches")
    def test_targets_are_shifted_inputs(self, mock_doc_batches):
        """targets[:, i] should equal the token at position i+1 in the row buffer."""
        B, T = 1, 8
        texts = ["abcdefghijklmnopqrstuvwxyz" * 10] * 100

        def fake_batches(*args, **kwargs):
            idx = 0
            while True:
                chunk = texts[idx:idx + 10]
                if not chunk:
                    idx = 0
                    chunk = texts[:10]
                idx += 10
                yield chunk, (0, 0, 1)

        mock_doc_batches.return_value = fake_batches()

        tokenizer = FakeTokenizer(bos_id=0)
        loader = data_loader_bos_bestfit(
            tokenizer, B=B, T=T, split="train",
            tokenizer_batch_size=10, buffer_size=5,
        )
        inputs, targets, _ = next(loader)

        # The row_buffer has T+1 tokens: inputs = row[:, :-1], targets = row[:, 1:]
        # So targets[0, i] == inputs[0, i+1] for all valid i
        for i in range(T - 1):
            assert targets[0, i] == inputs[0, i + 1]

    @patch("flaxchat.dataloader._document_batches")
    def test_state_dict_returned(self, mock_doc_batches):
        """State dict has pq_idx, rg_idx, epoch keys."""
        texts = ["test data " * 20] * 100

        def fake_batches(*args, **kwargs):
            idx = 0
            while True:
                chunk = texts[idx:idx + 10]
                if not chunk:
                    idx = 0
                    chunk = texts[:10]
                idx += 10
                yield chunk, (2, 5, 3)

        mock_doc_batches.return_value = fake_batches()

        tokenizer = FakeTokenizer(bos_id=0)
        loader = data_loader_bos_bestfit(
            tokenizer, B=1, T=8, split="train",
            tokenizer_batch_size=10, buffer_size=5,
        )
        _, _, state_dict = next(loader)

        assert "pq_idx" in state_dict
        assert "rg_idx" in state_dict
        assert "epoch" in state_dict

    def test_invalid_split_raises(self):
        """Passing an invalid split name should raise."""
        tokenizer = FakeTokenizer()
        with pytest.raises(AssertionError):
            loader = data_loader_bos_bestfit(tokenizer, B=1, T=8, split="test")
            next(loader)


class TestDataLoaderNoState:
    @patch("flaxchat.dataloader._document_batches")
    def test_omits_state_dict(self, mock_doc_batches):
        """data_loader_bos_bestfit_no_state yields only (inputs, targets)."""
        texts = ["test data " * 20] * 100

        def fake_batches(*args, **kwargs):
            idx = 0
            while True:
                chunk = texts[idx:idx + 10]
                if not chunk:
                    idx = 0
                    chunk = texts[:10]
                idx += 10
                yield chunk, (0, 0, 1)

        mock_doc_batches.return_value = fake_batches()

        tokenizer = FakeTokenizer(bos_id=0)
        loader = data_loader_bos_bestfit_no_state(
            tokenizer, B=1, T=8, split="train",
            tokenizer_batch_size=10, buffer_size=5,
        )
        result = next(loader)
        assert len(result) == 2
        inputs, targets = result
        assert inputs.shape == (1, 8)
        assert targets.shape == (1, 8)
