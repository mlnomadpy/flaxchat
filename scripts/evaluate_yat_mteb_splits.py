"""Evaluate one named split of a large MTEB task on a physical TPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation_contract import task_inventory, freeze_task_revision, validate_result
from flax import nnx
import jax

from flaxchat.public_encoder import load_public_encoder
from scripts.evaluate_encoder_mteb_v2 import MTEB_VERSION
from scripts.evaluate_yat_public_mteb import (
    AdaptiveMtebEncoder,
    _identity,
    _install_arrow_query_fast_path,
    _write_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--model-id", default="mlnomad/yat-mmbert-base-contrastive-12000")
    args = parser.parse_args()

    import mteb
    from mteb.models.abs_encoder import AbsEncoder

    if mteb.__version__ != MTEB_VERSION or jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise RuntimeError("Requires pinned MTEB on a physical single-host TPU")
    if args.batch_size < 1 or not 1 <= args.sequence_length <= 8192:
        raise ValueError("Invalid batch size or sequence length")
    _install_arrow_query_fast_path()

    task = mteb.get_task(args.task)
    freeze_task_revision(task)
    full_inventory = task_inventory(task)
    full_splits = list(task.eval_splits)
    if args.split not in full_splits:
        raise ValueError(f"Unknown split {args.split}: {full_splits}")
    identity = _identity(args.model_dir, "multilingual", args.sequence_length, args.model_id, args.batch_size)
    record = {
        "identity": identity,
        "task_name": args.task,
        "dataset_revision": task.metadata.dataset["revision"],
        "full_splits": full_splits,
        "full_task_inventory": full_inventory,
        "task_inventory": {**full_inventory, "eval_splits": [args.split],
                           "expected_rows": [[args.split, subset] for subset in full_inventory["hf_subsets"]]},
        "split": args.split,
        "batch_size": args.batch_size,
        "result": None,
        "encoded_texts": 0,
        "truncated_texts": 0,
    }
    if args.output.exists():
        saved = json.loads(args.output.read_text())
        if any(saved.get(key) != record[key] for key in
               ("identity", "task_name", "dataset_revision", "full_splits", "split", "batch_size", "full_task_inventory", "task_inventory")):
            raise ValueError("Cannot resume a different split experiment")
        if saved.get("result"):
            validate_result(args.task, {"category": saved["task_inventory"]["category"], "result": saved["result"]}, saved["task_inventory"])
            print(json.dumps({"task": args.task, "split": args.split, "resumed": True}))
            return
    _write_atomic(args.output, record)

    task = mteb.get_task(args.task, eval_splits=[args.split])
    task.metadata.dataset["revision"] = full_inventory["dataset_revision"]
    model = load_public_encoder(args.model_dir)

    class Session:
        def __init__(self, instance):
            self._model = instance
            self._pool = nnx.jit(lambda encoder, ids: encoder.pool(ids))

    class Adapter(AdaptiveMtebEncoder, AbsEncoder):
        pass

    adapter = Adapter(Session(model), args.model_dir / "tokenizer.json", model.config,
                      args.sequence_length, identity["model"],
                      identity["files_sha256"]["model.safetensors"])
    adapter.tokenizer.no_truncation()
    adapter.tokenizer.no_padding()
    result = mteb.evaluate(adapter, tasks=[task],
                           encode_kwargs={"batch_size": args.batch_size},
                           cache=None, overwrite_strategy="always",
                           co2_tracker=False, show_progress_bar=False)
    dumped = result.model_dump(mode="json")
    rows = dumped.get("task_results", [])
    if dumped.get("exceptions") or len(rows) != 1 or rows[0].get("task_name") != args.task:
        raise RuntimeError(f"MTEB task failure: {dumped.get('exceptions')}")
    if rows[0].get("dataset_revision") != record["dataset_revision"]:
        raise RuntimeError("Dataset revision changed")
    if set(rows[0].get("scores", {})) != {args.split}:
        raise RuntimeError("MTEB returned an unexpected set of splits")
    validate_result(args.task, {"category": record["task_inventory"]["category"], "result": rows[0]}, record["task_inventory"])
    record["result"] = rows[0]
    record["encoded_texts"] = adapter.encoded
    record["truncated_texts"] = adapter.truncated
    _write_atomic(args.output, record)
    print(json.dumps({"task": args.task, "split": args.split,
                      "score": rows[0]["scores"][args.split][0]["main_score"]}), flush=True)


if __name__ == "__main__":
    main()
