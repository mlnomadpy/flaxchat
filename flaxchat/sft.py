"""Reusable supervised fine-tuning data and update functions."""

from __future__ import annotations

import json
import os

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


def load_conversations(dataset_name: str, limit: int = 100_000) -> list[dict]:
    if os.path.exists(dataset_name):
        with open(dataset_name, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()][:limit]
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    conversations = []
    for item in dataset:
        if "messages" in item:
            conversations.append(item)
        if len(conversations) >= limit:
            break
    return conversations


def make_sft_batch(conversations, tokenizer, batch_size, max_seq_len, rng_key):
    if not conversations:
        raise ValueError("SFT requires at least one conversation")
    indices = jax.random.randint(rng_key, (batch_size,), 0, len(conversations))
    all_ids = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    all_targets = np.full((batch_size, max_seq_len), -1, dtype=np.int32)
    for row in range(batch_size):
        ids, mask = tokenizer.render_conversation(
            conversations[int(indices[row])], max_tokens=max_seq_len + 1
        )
        sequence_length = min(len(ids) - 1, max_seq_len)
        all_ids[row, :sequence_length] = ids[:sequence_length]
        supervised = np.asarray(mask[1:sequence_length + 1], dtype=bool)
        shifted = np.asarray(ids[1:sequence_length + 1], dtype=np.int32)
        all_targets[row, :sequence_length][supervised] = shifted[supervised]
    if not np.any(all_targets >= 0):
        raise ValueError("SFT batch contains no supervised assistant tokens")
    return jnp.asarray(all_ids), jnp.asarray(all_targets)


@nnx.jit
def train_step(model, optimizer, inputs, targets):
    loss, gradients = nnx.value_and_grad(
        lambda current: current(inputs, targets)
    )(model)
    optimizer.update(model, gradients)
    return loss
