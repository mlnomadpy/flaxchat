"""Distributed BOS-aligned packing with exact, validated resume state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os

import jax
import numpy as np
import pyarrow.parquet as pq

from flaxchat.dataset import DATASET_REVISION, list_parquet_files


DATALOADER_STATE_VERSION = 2


class DataloaderStateError(ValueError):
    """Raised when exact resume cannot be guaranteed."""


@dataclass(frozen=True)
class DocumentCursor:
    file_index: int
    row_group_index: int
    epoch: int
    row_offset: int = 0

    # Preserve the original three-value unpacking API used by callers/tests.
    def __iter__(self):
        yield self.file_index
        yield self.row_group_index
        yield self.epoch

    def to_dict(self):
        return {
            "file_index": self.file_index,
            "row_group_index": self.row_group_index,
            "row_offset": self.row_offset,
            "epoch": self.epoch,
        }


def _selected_paths(split: str) -> list[str]:
    paths = list_parquet_files(warn_on_legacy=(jax.process_index() == 0 and split == "train"))
    if not paths:
        raise FileNotFoundError("No dataset parquet files found")
    return paths[:-1] if split == "train" else paths[-1:]


def _dataset_manifest(split: str, paths: list[str]) -> dict:
    files = []
    for path in paths:
        try:
            stat = os.stat(path)
            files.append({
                "path": os.path.abspath(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        except OSError:
            # Remote and mocked paths still receive a stable ordered identity.
            files.append({"path": path})
    payload = {
        "dataset": "karpathy/climbmix-400b-shuffle",
        "revision": DATASET_REVISION,
        "split": split,
        "files": files,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _tokenizer_identity(tokenizer) -> dict:
    identity = {"class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"}
    for method, key in (("get_vocab_size", "vocab_size"), ("get_bos_token_id", "bos_token_id")):
        if hasattr(tokenizer, method):
            identity[key] = int(getattr(tokenizer, method)())
    return identity


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """Yield text batches and an exact cursor identifying the next text row."""
    process_index = jax.process_index()
    process_count = jax.process_count()
    parquet_paths = _selected_paths(split)

    state = resume_state_dict or {}
    if state.get("version") == DATALOADER_STATE_VERSION:
        cursor = state["document_cursor"]
        resume_pq_idx = int(cursor["file_index"])
        resume_rg_idx = int(cursor["row_group_index"])
        resume_row_offset = int(cursor["row_offset"])
        epoch = int(cursor["epoch"])
    else:
        # Legacy state is accepted only as its historical, row-group resume.
        resume_pq_idx = int(state.get("pq_idx", 0))
        legacy_rg = state.get("rg_idx")
        resume_rg_idx = (
            ((int(legacy_rg) // process_count) + 1) * process_count + process_index
            if legacy_rg is not None else process_index
        )
        resume_row_offset = 0
        epoch = int(state.get("epoch", 1))

    first_pass = True
    while True:
        pq_start = resume_pq_idx if first_pass else 0
        for pq_idx in range(pq_start, len(parquet_paths)):
            pf = pq.ParquetFile(parquet_paths[pq_idx])
            if first_pass and pq_idx == resume_pq_idx:
                rg_start = resume_rg_idx
            else:
                rg_start = process_index
            for rg_idx in range(rg_start, pf.num_row_groups, process_count):
                rows = pf.read_row_group(rg_idx).column("text").to_pylist()
                row_start = (
                    resume_row_offset
                    if first_pass and pq_idx == resume_pq_idx and rg_idx == resume_rg_idx
                    else 0
                )
                for start in range(row_start, len(rows), tokenizer_batch_size):
                    end = min(start + tokenizer_batch_size, len(rows))
                    next_cursor = DocumentCursor(pq_idx, rg_idx, epoch, end)
                    yield rows[start:end], next_cursor
        first_pass = False
        resume_pq_idx = 0
        resume_rg_idx = process_index
        resume_row_offset = 0
        epoch += 1


def _validate_resume(state, *, manifest, tokenizer_identity, packing, topology):
    if state.get("version") != DATALOADER_STATE_VERSION:
        if state:
            raise DataloaderStateError(
                "Legacy dataloader state is approximate and cannot be resumed exactly"
            )
        return
    expected = {
        "dataset_manifest": manifest,
        "tokenizer_identity": tokenizer_identity,
        "packing": packing,
        "topology": topology,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise DataloaderStateError(f"Dataloader resume contract changed: {mismatches}")


def data_loader_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    resume_state_dict=None, buffer_size=1000,
):
    """Yield packed batches plus bounded state that reproduces the exact next batch."""
    assert split in {"train", "val"}, f"Invalid split {split!r}; expected 'train' or 'val'"

    try:
        paths = _selected_paths(split)
    except FileNotFoundError:
        # Unit tests may replace the document source directly. The real source
        # will still fail clearly when the iterator is first advanced.
        paths = []
    manifest = _dataset_manifest(split, paths)
    tokenizer_id = _tokenizer_identity(tokenizer)
    packing = {
        "batch_size": B,
        "sequence_length": T,
        "buffer_size": buffer_size,
        "tokenizer_batch_size": tokenizer_batch_size,
        "policy": "bos_bestfit_v2",
    }
    topology = {
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
    }
    state = resume_state_dict or {}
    _validate_resume(
        state, manifest=manifest, tokenizer_identity=tokenizer_id,
        packing=packing, topology=topology,
    )

    row_capacity = T + 1
    batches = _document_batches(split, state or None, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = [list(tokens) for tokens in state.get("doc_buffer", [])]
    cursor = DocumentCursor(0, jax.process_index(), 1, 0)

    def refill_buffer():
        nonlocal cursor
        doc_batch, cursor = next(batches)
        token_lists = tokenizer.encode(
            doc_batch, prepend=bos_token, num_threads=tokenizer_threads
        )
        doc_buffer.extend([list(tokens) for tokens in token_lists])

    row_buffer = np.empty((B, row_capacity), dtype=np.int32)
    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                fitting = [
                    (len(doc), index)
                    for index, doc in enumerate(doc_buffer)
                    if len(doc) <= remaining
                ]
                if fitting:
                    _, index = max(fitting)
                    doc = doc_buffer.pop(index)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    index = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(index)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = row_buffer[:, :-1].copy()
        targets = row_buffer[:, 1:].copy()
        cursor_dict = cursor.to_dict() if hasattr(cursor, "to_dict") else {
            "file_index": int(cursor[0]),
            "row_group_index": int(cursor[1]),
            "row_offset": 0,
            "epoch": int(cursor[2]),
        }
        resume = {
            "version": DATALOADER_STATE_VERSION,
            "dataset_manifest": manifest,
            "tokenizer_identity": tokenizer_id,
            "packing": packing,
            "topology": topology,
            "document_cursor": cursor_dict,
            "doc_buffer": [list(tokens) for tokens in doc_buffer],
            # Legacy summary fields for logging only.
            "pq_idx": cursor_dict["file_index"],
            "rg_idx": cursor_dict["row_group_index"],
            "epoch": cursor_dict["epoch"],
        }
        yield inputs, targets, resume


def data_loader_bos_bestfit_no_state(*args, **kwargs):
    for inputs, targets, _ in data_loader_bos_bestfit(*args, **kwargs):
        yield inputs, targets
