"""Evaluate the published YAT checkpoint, unchanged, on complete MTEB v2 suites.

Model forward passes require a physical TPU. Results are written after every
task so a Spot preemption can be resumed without repeating completed tasks.
This is an as-is embedding evaluation, not a reproduction of mmBERT's separate
MS MARCO training stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from flaxchat.encoder_data import file_hash
from scripts.evaluation_contract import aggregate as _aggregate, write_atomic as _write_atomic, task_inventory, freeze_task_revision, provenance
from flaxchat.public_encoder import load_public_encoder
from scripts.evaluate_encoder_mteb_v2 import MTEB_VERSION, MtebEncoder


BENCHMARKS = {
    "english": "MTEB(eng, v2)",
    "multilingual": "MTEB(Multilingual, v2)",
    "coir": "CoIR",
}


def _install_arrow_query_fast_path() -> None:
    """Avoid materializing millions of query strings for plain-text retrieval.

    MTEB 2.21.8 copies ``dataset['text']`` into a Python sequence before
    adding its identical ``query`` column. Passing the underlying Arrow
    column preserves the exact values and buffers without that conversion.
    Instruction-bearing tasks retain MTEB's original transformation.
    """
    import mteb._create_dataloaders as dataloaders

    original = dataloaders._combine_queries_with_instruction_text

    def combine(dataset):
        if "instruction" in dataset.column_names or "text" not in dataset.column_names:
            return original(dataset)
        if "query" in dataset.column_names:
            dataset = dataset.remove_columns(["query"])
        # Arrow's physical table still contains rows removed by select/filter.
        # Use MTEB's logical-row path when an indices mapping is present.
        if dataset._indices is not None:
            return original(dataset)
        return dataset.add_column("query", dataset.data.column("text"))

    dataloaders._combine_queries_with_instruction_text = combine


def _identity(model_dir: Path, benchmark: str, length: int, model_id: str, batch_size: int = 16) -> dict:
    names = ("config.json", "tokenizer.json", "model.safetensors")
    return {
        "model": model_id,
        "files_sha256": {name: file_hash(model_dir / name) for name in names},
        "mteb_version": MTEB_VERSION,
        "benchmark": BENCHMARKS[benchmark],
        "sequence_length": length,
        "padding_policy": "per-batch power-of-two buckets from 32 up to sequence_length",
        "pooling": "mean nonpadding states including special tokens; FP32 L2 normalization",
        "prompts": "none",
        "normalization": "L2 FP32",
        "truncation_policy": "right-truncate-token-ids-to-sequence-length-before-padding",
        "score_scale": "native-mteb-main-score/no-rescaling",
        "backend": "physical TPU",
        "protocol": provenance([Path(__file__), Path("scripts/evaluation_contract.py"), Path("scripts/evaluate_yat_mteb_subsets.py"),
                                 Path("scripts/evaluate_yat_mteb_splits.py"), Path("scripts/merge_yat_mteb_shards.py"),
                                 Path("scripts/merge_yat_mteb_splits.py"), Path("scripts/merge_yat_mteb_subsets.py"),
                                 Path("scripts/evaluate_encoder_mteb_v2.py"),
                                 Path("flaxchat/public_encoder.py"), Path("flaxchat/encoder.py"),
                                 Path("flaxchat/yat.py"), Path("flaxchat/yat_attention.py"),
                                 Path("flaxchat/yat_attention_centered.py")],
                                devices=[{"kind": d.device_kind, "platform": d.platform,
                                          "process_index": d.process_index} for d in jax.devices()],
                                batch_size=batch_size),
    }


class AdaptiveMtebEncoder(MtebEncoder):
    """Use length buckets so short texts do not pay for 512 padded tokens."""

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs):
        if "prompt" in kwargs and kwargs["prompt"]:
            raise ValueError("This YAT evaluation does not use prompts")
        batch_size = kwargs.get("batch_size", 16)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive inference batch size required")
        dataset = getattr(inputs, "dataset", None)
        expected = len(dataset) if dataset is not None else None
        output = (np.empty((expected, self.config.hidden_size), np.float32)
                  if expected is not None else None)
        fallback = [] if output is None else None
        offset = 0
        for batch in inputs:
            texts = batch.get("text")
            if not isinstance(texts, (list, tuple)) or any(not isinstance(s, str) for s in texts):
                raise ValueError("Text-only MTEB tasks required")
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                encoded = self.tokenizer.encode_batch(chunk)
                longest = min(self.length, max((len(item.ids) for item in encoded), default=1))
                bucket = min(self.length, max(32, 1 << (longest - 1).bit_length()))
                ids = np.full((batch_size, bucket), self.config.pad_token_id, np.int32)
                for index, item in enumerate(encoded):
                    if len(item.ids) > self.length:
                        self.truncated += 1
                    ids[index, :min(len(item.ids), bucket)] = item.ids[:bucket]
                vectors = np.asarray(self.session._pool(self.session._model, jnp.asarray(ids)),
                                     dtype=np.float32)[:len(chunk)].copy()
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                if not np.isfinite(vectors).all():
                    raise ValueError("Nonfinite MTEB embedding")
                if output is None:
                    fallback.append(vectors)
                else:
                    end = offset + len(vectors)
                    if end > expected:
                        raise ValueError("MTEB yielded more rows than its dataset length")
                    output[offset:end] = vectors
                    offset = end
                previous_encoded = self.encoded
                self.encoded += len(chunk)
                if self.encoded // 50000 > previous_encoded // 50000:
                    print(json.dumps({"encoded_texts": self.encoded,
                                      "truncated_texts": self.truncated}), flush=True)
        if output is not None:
            if offset != expected:
                raise ValueError("MTEB yielded fewer rows than its dataset length")
            return output
        return (np.concatenate(fallback, axis=0) if fallback
                else np.empty((0, self.config.hidden_size), np.float32))


def evaluate(model_dir: Path, benchmark: str, output: Path, *, length: int,
             batch_size: int, only: set[str], stop_after: int | None,
             model_id: str = "mlnomad/yat-mmbert-base-contrastive-12000") -> dict:
    import mteb
    from mteb.models.abs_encoder import AbsEncoder

    if mteb.__version__ != MTEB_VERSION:
        raise ValueError(f"Requires MTEB {MTEB_VERSION}, got {mteb.__version__}")
    _install_arrow_query_fast_path()
    if jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise RuntimeError("Model evaluation requires a physical, single-host TPU")
    if not 1 <= length <= 8192 or batch_size < 1:
        raise ValueError("Invalid sequence length or batch size")

    benchmark_obj = mteb.get_benchmark(BENCHMARKS[benchmark])
    tasks = sorted(benchmark_obj.tasks, key=lambda task: task.metadata.name)
    for task in tasks:
        freeze_task_revision(task)
    inventory = {task.metadata.name: task_inventory(task) for task in tasks}
    if len(inventory) != len(tasks) or only - inventory.keys():
        raise ValueError("Duplicate or unknown MTEB task name")
    identity = _identity(model_dir, benchmark, length, model_id, batch_size)
    if output.exists():
        record = json.loads(output.read_text())
        if record["identity"] != identity or record["inventory"] != inventory:
            raise ValueError("Cannot resume a different model or benchmark inventory")
        if record.get("selected_tasks", sorted(inventory)) != (sorted(only) if only else sorted(inventory)):
            raise ValueError("Cannot resume with a different task selection")
    else:
        record = {"identity": identity, "inventory": inventory,
                  "selected_tasks": sorted(only) if only else sorted(inventory),
                  "results": {}, "failures": {}, "aggregate": {}}
        _write_atomic(output, record)

    record["aggregate"] = _aggregate(record)
    model = load_public_encoder(model_dir)

    class Session:
        def __init__(self, instance):
            self._model = instance
            # Store the jitted callable on the instance: a class attribute
            # would become a bound method and pass Session as an argument.
            self._pool = nnx.jit(lambda encoder, ids: encoder.pool(ids))

    class Adapter(AdaptiveMtebEncoder, AbsEncoder):
        pass

    adapter = Adapter(Session(model), model_dir / "tokenizer.json", model.config,
                      length, identity["model"], identity["files_sha256"]["model.safetensors"])
    adapter.tokenizer.no_truncation()
    adapter.tokenizer.no_padding()
    attempted = 0
    for task in tasks:
        name = task.metadata.name
        if name in record["results"] or (only and name not in only):
            continue
        if stop_after is not None and attempted >= stop_after:
            break
        attempted += 1
        try:
            result = mteb.evaluate(adapter, tasks=[task],
                                   encode_kwargs={"batch_size": batch_size},
                                   cache=None, overwrite_strategy="always",
                                   co2_tracker=False, show_progress_bar=False)
            dumped = result.model_dump(mode="json")
            rows = dumped.get("task_results", [])
            if dumped.get("exceptions") or len(rows) != 1 or rows[0].get("task_name") != name:
                raise RuntimeError(f"MTEB did not return one complete result: {dumped.get('exceptions')}")
            if rows[0].get("dataset_revision") != inventory[name]["dataset_revision"]:
                raise RuntimeError("MTEB dataset revision changed")
            record["results"][name] = {"category": inventory[name]["category"],
                                       "result": rows[0]}
            try:
                _aggregate(record)
            except ValueError:
                record["results"].pop(name, None)
                raise
            record["failures"].pop(name, None)
        except Exception as exc:  # keep other independent tasks running
            record["failures"][name] = {"type": type(exc).__name__,
                                        "message": str(exc),
                                        "traceback": traceback.format_exc()[-12000:]}
        record["encoded_texts"] = adapter.encoded
        record["truncated_texts"] = adapter.truncated
        record["aggregate"] = _aggregate(record)
        _write_atomic(output, record)
        print(json.dumps({"task": name, "completed": name in record["results"],
                          "results": len(record["results"]),
                          "failures": len(record["failures"]),
                          "total": len(inventory)}), flush=True)
    record["aggregate"] = _aggregate(record)
    _write_atomic(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-id", default="mlnomad/yat-mmbert-base-contrastive-12000")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks-file", type=Path,
                        help="JSON array of task names assigned to this worker")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    selected = set(args.task)
    if args.tasks_file:
        names = json.loads(args.tasks_file.read_text())
        if not isinstance(names, list) or len(names) != len(set(names)) or not all(isinstance(name, str) for name in names):
            raise ValueError("tasks-file must contain a unique JSON string array")
        selected.update(names)
    result = evaluate(args.model_dir, args.benchmark, args.output,
                      length=args.sequence_length, batch_size=args.batch_size,
                      only=selected, stop_after=args.stop_after,
                      model_id=args.model_id)
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
