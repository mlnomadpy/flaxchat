"""Run the matched trainer protocol with nanochat's native PyTorch model."""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

from benchmarks.compare import protocol_sha256
from benchmarks.matched.common import (
    MEASURED_STEPS,
    SEED,
    WARMUP_STEPS,
    file_sha256,
    load_batches,
    make_record,
    validate_hardware,
    write_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["NANOCHAT_DTYPE"] = "float32"
    sys.path.insert(0, str(args.source))
    from nanochat.gpt import GPT, GPTConfig  # pyright: ignore[reportMissingImports]

    if not torch.cuda.is_available():
        raise RuntimeError("nanochat matched run requires the Kaggle P100 GPU")
    detected = torch.cuda.get_device_name(0)
    validate_hardware(detected)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    batches = load_batches(args.data)
    config = GPTConfig(
        sequence_len=256,
        vocab_size=512,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
    )
    model = GPT(config).to(device)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def eager_step(inputs, targets):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        return loss

    step = torch.compile(eager_step)

    def tensors(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(np.asarray(batches["train_inputs"][index]), device=device),
            torch.as_tensor(np.asarray(batches["train_targets"][index]), device=device),
        )

    torch.cuda.reset_peak_memory_stats()
    inputs, targets = tensors(0)
    started = time.perf_counter()
    loss = step(inputs, targets)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started
    del loss
    for index in range(1, WARMUP_STEPS + 1):
        inputs, targets = tensors(index)
        step(inputs, targets)
        torch.cuda.synchronize()
    durations = []
    for index in range(WARMUP_STEPS + 1, WARMUP_STEPS + MEASURED_STEPS + 1):
        inputs, targets = tensors(index)
        torch.cuda.synchronize()
        started = time.perf_counter()
        step(inputs, targets)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    model.eval()
    with torch.no_grad():
        validation_value = float(model(
            torch.as_tensor(batches["validation_inputs"], device=device),
            torch.as_tensor(batches["validation_targets"], device=device),
        ).item())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "matched.pt"
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
    with checkpoint_path.open("rb") as saved:
        os.fsync(saved.fileno())
    checkpoint_seconds = time.perf_counter() - started
    record = make_record(
        framework="nanochat",
        source_revision=args.revision,
        model_parameters=parameters,
        steady_tokens_per_second=(8 * 256) / statistics.median(durations),
        compile_seconds=compile_seconds,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_seconds=checkpoint_seconds,
        validation_value=validation_value,
        protocol_sha256=protocol_sha256(args.protocol),
        data_sha256=file_sha256(args.data),
        software={"python": platform.python_version(), "torch": torch.__version__},
        limitations="PyTorch compile timing is the synchronized first compiled update; peak memory uses CUDA allocator telemetry.",
    )
    write_record(args.output, record)


if __name__ == "__main__":
    main()
