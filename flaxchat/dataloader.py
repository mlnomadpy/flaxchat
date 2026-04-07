"""
Distributed dataloader for pretraining on TPU pods.

BOS-aligned best-fit packing — port of nanochat's dataloader:
- Every row starts with BOS token
- Documents packed using best-fit algorithm to minimize cropping
- 100% utilization (no padding), ~35% tokens cropped at T=2048

Adapted for JAX multi-host TPU training:
- Uses jax.process_index() / jax.process_count() for sharding
- Returns JAX arrays instead of PyTorch tensors
- Supports jax.make_array_from_process_local_data for global sharding
"""

import numpy as np
import jax
import jax.numpy as jnp
import pyarrow.parquet as pq

from flaxchat.dataset import list_parquet_files


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches from parquet files.
    Handles multi-host sharding and approximate resume.
    """
    process_index = jax.process_index()
    process_count = jax.process_count()

    parquet_paths = list_parquet_files(warn_on_legacy=(process_index == 0 and split == "train"))
    assert len(parquet_paths) != 0, "No dataset parquet files found"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)

            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // process_count
                base_idx += 1
                rg_idx = base_idx * process_count + process_index
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = process_index

            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += process_count
            pq_idx += 1
        first_pass = False
        epoch += 1


def data_loader_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    resume_state_dict=None, buffer_size=1000,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping for JAX/TPU.

    Yields (inputs, targets, state_dict) where:
    - inputs: np.array (B, T) int32
    - targets: np.array (B, T) int32
    - state_dict: dict for resume

    Returns numpy arrays — caller is responsible for sharding to TPU.
    """
    assert split in ["train", "val"]

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate numpy buffer
    row_buffer = np.empty((B, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = doc
                    pos += doc_len
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = row_buffer[:, :-1].copy()
        targets = row_buffer[:, 1:].copy()
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        yield inputs, targets, state_dict


def data_loader_bos_bestfit_no_state(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, _ in data_loader_bos_bestfit(*args, **kwargs):
        yield inputs, targets
