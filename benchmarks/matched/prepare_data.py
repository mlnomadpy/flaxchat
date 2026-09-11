"""Create the one immutable token-batch artifact consumed by every trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
import numpy as np

from benchmarks.matched.common import (
    DATASET,
    DATASET_REVISION,
    GLOBAL_BATCH_SIZE,
    MEASURED_STEPS,
    SEED,
    SEQUENCE_LENGTH,
    WARMUP_STEPS,
    file_sha256,
)


def encode_documents(documents, required_tokens: int) -> np.ndarray:
    """Use a framework-neutral byte encoding: pad=0, bos=1, eos=2, byte=3..258."""
    tokens: list[int] = []
    for document in documents:
        text = document.get("text", "")
        tokens.extend((1, *(byte + 3 for byte in text.encode("utf-8")), 2))
        if len(tokens) >= required_tokens:
            break
    if len(tokens) < required_tokens:
        raise RuntimeError(f"dataset stream yielded {len(tokens)} of {required_tokens} required tokens")
    return np.asarray(tokens[:required_tokens], dtype=np.int32)


def make_sequences(stream: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    width = SEQUENCE_LENGTH + 1
    sequences = stream[: count * width].reshape(count, width)
    return sequences[:, :-1], sequences[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train_batches = WARMUP_STEPS + MEASURED_STEPS + 1
    train_sequences = train_batches * GLOBAL_BATCH_SIZE
    validation_sequences = GLOBAL_BATCH_SIZE
    train = load_dataset(DATASET, revision=DATASET_REVISION, split="train", streaming=True)
    validation = load_dataset(DATASET, revision=DATASET_REVISION, split="validation", streaming=True)
    train_stream = encode_documents(train, train_sequences * (SEQUENCE_LENGTH + 1))
    validation_stream = encode_documents(validation, validation_sequences * (SEQUENCE_LENGTH + 1))
    train_inputs, train_targets = make_sequences(train_stream, train_sequences)
    validation_inputs, validation_targets = make_sequences(validation_stream, validation_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train_inputs=train_inputs.reshape(train_batches, GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH),
        train_targets=train_targets.reshape(train_batches, GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH),
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
    )
    metadata = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "encoding": "pad=0,bos=1,eos=2,utf8_byte=3..258",
        "sha256": file_sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
