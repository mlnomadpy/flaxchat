"""Run and summarize a fail-fast physical multi-host JAX acceptance probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import socket
import subprocess
import sys
import time
from typing import Any


SHA40 = re.compile(r"[0-9a-f]{40}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_revision(root: Path) -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if not SHA40.fullmatch(revision):
        raise RuntimeError(f"expected a full immutable source revision, got {revision!r}")
    return revision


def validate_records(records: list[dict[str, Any]], *, cost_usd: float | None) -> dict[str, Any]:
    """Fail closed and build the committed summary from collected worker records."""
    if cost_usd is not None and (cost_usd < 0 or not math.isfinite(cost_usd)):
        raise ValueError("cost_usd must be a finite non-negative value")
    if len(records) < 2:
        raise ValueError("physical multi-host acceptance requires at least two records")
    for record in records:
        if not SHA40.fullmatch(record['source_revision']):
            raise ValueError('Source revision must be an immutable SHA')
        for key in ('source_python_sha256', 'environment_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', record[key]):
                raise ValueError(f'Invalid {key}')
    indexes = sorted(int(record["topology"]["process_index"]) for record in records)
    expected = list(range(len(records)))
    if indexes != expected:
        raise ValueError(f"process records must be contiguous {expected}, got {indexes}")
    fields = {
        "source_revision": {record["source_revision"] for record in records},
        "environment_sha256": {record["environment_sha256"] for record in records},
        "global_order_sha256": {record["data"]["global_order_sha256"] for record in records},
        "process_count": {int(record["topology"]["process_count"]) for record in records},
        "device_count": {int(record["topology"]["device_count"]) for record in records},
        "loss": {float(record["training"]["loss"]) for record in records},
        "gradient": {float(record["training"]["gradient"]) for record in records},
        "updated_parameter": {float(record["training"]["updated_parameter"]) for record in records},
        "source_python_sha256": {record["source_python_sha256"] for record in records},
    }
    mismatches = {name: values for name, values in fields.items() if len(values) != 1}
    if mismatches:
        raise ValueError(f"workers disagree on synchronized evidence: {mismatches}")
    process_count = fields["process_count"].pop()
    if process_count != len(records):
        raise ValueError(f"reported process_count {process_count} != records {len(records)}")
    if any(record['topology'].get('backend') != 'tpu' for record in records):
        raise ValueError('Physical acceptance requires TPU on every worker')
    if len({record['hostname'] for record in records}) != process_count:
        raise ValueError('Physical acceptance requires distinct worker hosts')
    if sum(record['topology']['local_device_count'] for record in records) != next(iter(fields['device_count'])):
        raise ValueError('Local device counts do not sum to global devices')
    local_batches = [set(record["data"]["local_indices"]) for record in records]
    if any(not batch for batch in local_batches):
        raise ValueError('Every worker must contribute a nonempty batch')
    if any(
        len(batch) != len(record["data"]["local_indices"])
        for batch, record in zip(local_batches, records, strict=True)
    ):
        raise ValueError("a worker batch contains duplicate indices")
    for left in range(len(local_batches)):
        for right in range(left + 1, len(local_batches)):
            if local_batches[left] & local_batches[right]:
                raise ValueError(f"worker batches overlap: {left} and {right}")
    combined = sorted(value for batch in local_batches for value in batch)
    if combined != list(range(len(combined))):
        raise ValueError("worker batches do not reproduce the canonical global order")
    canonical_order_sha256 = hashlib.sha256(
        struct.pack(f"<{len(combined)}i", *combined)
    ).hexdigest()
    if fields["global_order_sha256"] != {canonical_order_sha256}:
        raise ValueError("worker global-order digest does not match the canonical indices")
    if not all(record.get("passed") is True for record in records):
        raise ValueError("at least one worker probe failed")
    losses = fields["loss"]
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("worker loss must be finite")
    mean_square = sum(index ** 2 for index in combined) / len(combined)
    for name, expected_value in [('loss', 2.25 * mean_square),
                                 ('gradient', -3.0 * mean_square),
                                 ('updated_parameter', .5 + .003 * mean_square)]:
        if not math.isclose(next(iter(fields[name])), expected_value, rel_tol=1e-5):
            raise ValueError(f'Incorrect synchronized {name}')
    return {
        "format_version": 1,
        "status": "probe_passed",
        "source_revision": fields["source_revision"].pop(),
        "source_python_sha256": fields['source_python_sha256'].pop(),
        "environment_sha256": fields["environment_sha256"].pop(),
        "topology": {
            "process_count": process_count,
            "device_count": fields["device_count"].pop(),
            "single_host": False,
        },
        "data": {
            "global_order_sha256": canonical_order_sha256,
            "global_indices": combined,
            "disjoint": True,
        },
        "training": {
            "finite_matching_loss": fields["loss"].pop(),
            "all_workers_passed": True,
        },
        "cost_usd": cost_usd,
        "cost_status": 'unavailable' if cost_usd is None else 'operator_supplied',
        "limitations": [
            "This fail-fast probe covers initialization, topology, host-local data order, a synchronized gradient update, and sharding only.",
            "Cross-topology checkpoint restore and interrupted-run resume remain mandatory later phases of issue #12.",
        ],
        "workers": sorted(records, key=lambda record: record["topology"]["process_index"]),
    }


def run_probe(args: argparse.Namespace) -> int:
    # Importing flaxchat.common initializes distributed JAX before any backend query.
    from flaxchat.common import compute_init, replicate_on_mesh

    mesh = compute_init()
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    import numpy as np

    from scripts.validate_tpu import source_digest
    started = time.time()
    process_count = jax.process_count()
    process_index = jax.process_index()
    if process_count < args.min_processes:
        raise RuntimeError(
            f"physical multi-host probe requires >= {args.min_processes} processes, got {process_count}"
        )
    if args.require_tpu and jax.default_backend() != "tpu":
        raise RuntimeError(f"physical acceptance requires TPU, got {jax.default_backend()}")
    expected = np.arange(process_count * args.local_batch, dtype=np.int32)
    data_sharding = NamedSharding(mesh, P("data", None))
    global_shape = (len(expected), 1)
    ranges = set()
    for index in data_sharding.addressable_devices_indices_map(global_shape).values():
        if index is None:
            raise RuntimeError('An addressable device has no global batch index')
        ranges.add((index[0].start or 0, index[0].stop or len(expected)))
    ranges = sorted(ranges)
    local_indices = np.concatenate([np.arange(start, stop, dtype=np.int32) for start, stop in ranges])
    gathered = np.asarray(multihost_utils.process_allgather(local_indices, tiled=True)).reshape(-1)
    np.testing.assert_array_equal(np.sort(gathered), expected)
    def read_rows(index):
        if index is None:
            raise RuntimeError('Missing global batch index')
        return expected[index[0]].astype(np.float32).reshape(-1, 1)
    inputs = jax.make_array_from_callback(global_shape, data_sharding, read_rows)
    np.testing.assert_array_equal(np.asarray(multihost_utils.process_allgather(inputs, tiled=True)),
                                  expected.reshape(-1, 1))

    @jax.jit
    def update(weight, batch):
        def loss_fn(candidate):
            targets = 2.0 * batch
            return jnp.mean(jnp.square(candidate * batch - targets))

        loss, gradient = jax.value_and_grad(loss_fn)(weight)
        return loss, gradient, weight - 1e-3 * gradient

    weight = replicate_on_mesh(jnp.asarray(0.5, dtype=jnp.float32), mesh)
    loss, gradient, updated = update(weight, inputs)
    loss, gradient, updated = jax.block_until_ready((loss, gradient, updated))
    losses = np.asarray(multihost_utils.process_allgather(np.asarray(loss), tiled=False)).reshape(-1)
    if not np.all(np.isfinite(losses)) or not np.allclose(losses, losses[0]):
        raise RuntimeError(f"worker losses are not finite and matching: {losses.tolist()}")
    mean_square = np.mean(expected.astype(np.float64) ** 2)
    np.testing.assert_allclose(float(loss), 2.25 * mean_square, rtol=1e-5)
    np.testing.assert_allclose(float(gradient), -3.0 * mean_square, rtol=1e-5)
    np.testing.assert_allclose(float(updated), 0.5 + .003 * mean_square, rtol=1e-5)
    if inputs.shape != (process_count * args.local_batch, 1):
        raise RuntimeError('Unexpected global batch shape')
    revision = source_revision(args.repository)
    record = {
        "format_version": 1,
        "passed": True,
        "source_revision": revision,
        "source_python_sha256": source_digest(args.repository),
        "hostname": socket.gethostname(),
        "environment_sha256": file_sha256(args.environment_file),
        "command": sys.argv,
        "location": {"project": args.project, "zone": args.zone, "slice": args.slice},
        "topology": {
            "backend": jax.default_backend(),
            "process_index": process_index,
            "process_count": process_count,
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "device_kinds": sorted({device.device_kind for device in jax.devices()}),
        },
        "data": {
            "local_indices": local_indices.tolist(),
            "global_order_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        },
        "training": {
            "loss": float(loss),
            "gradient": float(gradient),
            "updated_parameter": float(updated),
            "input_shape": list(inputs.shape),
            "input_sharding": str(inputs.sharding),
            "addressable_shards": len(inputs.addressable_shards),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"process-{process_index}.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    multihost_utils.sync_global_devices("flaxchat-multihost-probe-complete")
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


def summarize(args: argparse.Namespace) -> int:
    paths = sorted(args.input_dir.rglob("process-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = validate_records(records, cost_usd=args.cost_usd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/multihost"))
    run.add_argument("--repository", type=Path, default=Path.cwd())
    run.add_argument("--environment-file", type=Path, default=Path("infra/tpu/environment.txt"))
    run.add_argument("--project", required=True)
    run.add_argument("--zone", required=True)
    run.add_argument("--slice", required=True)
    run.add_argument("--min-processes", type=int, default=2)
    run.add_argument("--local-batch", type=int, default=8)
    run.add_argument("--require-tpu", action=argparse.BooleanOptionalAction, default=True)
    report = subparsers.add_parser("summarize")
    report.add_argument("--input-dir", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--cost-usd", type=float, help='Actual billed cost; omit while billing is pending')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_probe(args) if args.mode == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
