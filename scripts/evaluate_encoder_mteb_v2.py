"""Run a small, paired MTEB v2 zero-shot suite on a physical TPU.

Install ``mteb==2.21.8`` on the TPU worker. Run this script once per role with
the same frozen plan, then compare reports with ``--compare``. MTEB itself
loads each task and computes its official per-task metrics. The selected tasks
are a diagnostic slice, never the full MTEB/mmBERT paper aggregate.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer

from flaxchat.checkpoint import load_checkpoint_metadata
from flaxchat.encoder import EncoderConfig
from flaxchat.encoder_data import file_hash
from scripts.export_encoder_bitext_campaign import _metadata_sha256
from scripts.export_encoder_retrieval import EmbeddingSession


MTEB_VERSION = "2.21.8"
TASKS = ("STSBenchmark.v2", "STS17", "SciFact", "NFCorpus")
TASK_REVISIONS = {
    "STSBenchmark.v2": "93b628c3969a75e76727db2b7ee252e53e96268d",
    "STS17": "faeb762787bd10488a50c8b5be4a3b82e411949c",
    "SciFact": "d56462d0e63a25450459c4f213e49ffdb866f7f9",
    "NFCorpus": "ec0fa4fe99da2ff19ca1214b7966684033a58814",
}
RUNTIME_FIELDS = (
    "compute_dtype", "residual_dtype", "use_remat", "attention_backend",
    "loss_chunk_size", "mlm_projection", "mlm_loss_backend", "mlm_vocab_tile",
)


def verify_plan(path):
    path = Path(path)
    identity = file_hash(path)
    plan = json.loads(path.read_text())
    if plan.get("mteb_version") != MTEB_VERSION or plan.get("tasks") != list(TASKS):
        raise ValueError("Unexpected MTEB version or diagnostic tasks")
    if plan.get("task_revisions") != TASK_REVISIONS or plan.get("eval_splits") != ["test"]:
        raise ValueError("Unexpected MTEB task revisions or split selection")
    if type(plan.get("sequence_length")) is not int or not 1 <= plan["sequence_length"] <= 8192:
        raise ValueError("Invalid frozen sequence length")
    if plan.get("pooling") != "mean of nonpadding token states, including special tokens" or plan.get("normalization") != "L2 FP32":
        raise ValueError("Unexpected embedding recipe")
    baseline = plan["baseline"]
    candidate = plan["candidate"]
    snapshot = Path(baseline["snapshot"])
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        if file_hash(snapshot / name) != baseline["sha256"][name]:
            raise ValueError(f"Released mmBERT snapshot changed: {name}")
    if type(candidate.get("step")) is not int or candidate["step"] < 1 or not candidate.get("checkpoint"):
        raise ValueError("Explicit candidate checkpoint required")
    metadata = load_checkpoint_metadata(candidate["checkpoint"], step=candidate["step"])
    if (metadata.get("model_family") != "modernbert_contrastive_encoder" or
        metadata.get("tokenizer_identity") != baseline["sha256"]["tokenizer.json"] or
        _metadata_sha256(metadata) != candidate.get("metadata_sha256")):
        raise ValueError("Candidate provenance differs from frozen plan")
    candidate_config = EncoderConfig(**metadata["resolved_config"]["encoder"])
    released_config = EncoderConfig.from_hf(
        json.loads((snapshot / "config.json").read_text()),
        **{name: getattr(candidate_config, name) for name in RUNTIME_FIELDS},
    )
    for name in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                 "max_position_embeddings", "pad_token_id", "mask_token_id"):
        if getattr(candidate_config, name) != getattr(released_config, name):
            raise ValueError(f"Unmatched model geometry: {name}")
    if plan["sequence_length"] > min(candidate_config.max_position_embeddings, released_config.max_position_embeddings):
        raise ValueError("Frozen length exceeds model position limit")
    if file_hash(path) != identity:
        raise ValueError("MTEB plan changed during verification")
    return plan, identity, candidate_config, released_config


def build_plan(checkpoint, step, snapshot, output, *, sequence_length=512):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an MTEB plan")
    snapshot = Path(snapshot)
    hashes = {name: file_hash(snapshot / name) for name in ("config.json", "tokenizer.json", "model.safetensors")}
    metadata = load_checkpoint_metadata(checkpoint, step=step)
    if metadata.get("model_family") != "modernbert_contrastive_encoder" or metadata.get("tokenizer_identity") != hashes["tokenizer.json"]:
        raise ValueError("Require a YAT contrastive checkpoint with the mmBERT tokenizer")
    plan = dict(
        scope="paired_zero_shot_mteb_v2_diagnostic_slice", mteb_version=MTEB_VERSION,
        tasks=list(TASKS), task_revisions=TASK_REVISIONS,
        eval_splits=["test"], sequence_length=sequence_length,
        pooling="mean of nonpadding token states, including special tokens",
        normalization="L2 FP32", prompts="none for either model",
        long_text_policy="truncate both models to frozen sequence length; count truncations",
        baseline=dict(snapshot=str(snapshot), sha256=hashes),
        candidate=dict(checkpoint=str(checkpoint), step=step,
                       metadata_sha256=_metadata_sha256(metadata)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n")
    try:
        verify_plan(output)
    except Exception:
        output.unlink()
        raise
    return dict(path=str(output), sha256=file_hash(output))


class MtebEncoder:
    """MTEB v2 text encoder backed by the same harness pooling for both roles."""

    def __init__(self, session, tokenizer_path, config, length, name, revision):
        from mteb.models.model_meta import ModelMeta, ScoringFunction

        self.mteb_model_meta = ModelMeta.create_empty(overwrites=dict(
            name=name, revision=revision, max_tokens=length,
            embed_dim=config.hidden_size, similarity_fn_name=ScoringFunction.COSINE,
            framework=["JAX"],
        ))
        self.session = session
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.config = config
        self.length = length
        self.truncated = 0
        self.encoded = 0

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        if "prompt" in kwargs and kwargs["prompt"]:
            raise ValueError("This paired protocol does not allow model-specific prompts")
        batch_size = kwargs.get("batch_size", 16)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive MTEB inference batch size required")
        output = []
        for batch in inputs:
            texts = batch.get("text")
            if not isinstance(texts, (list, tuple)) or any(not isinstance(s, str) for s in texts):
                raise ValueError("Text-only MTEB tasks required")
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                encoded = self.tokenizer.encode_batch(chunk)
                ids = np.full((batch_size, self.length), self.config.pad_token_id, np.int32)
                for index, item in enumerate(encoded):
                    sequence = item.ids
                    if len(sequence) > self.length:
                        self.truncated += 1
                    ids[index, :min(len(sequence), self.length)] = sequence[:self.length]
                vectors = np.asarray(self.session._pool(self.session._model, jnp.asarray(ids)), dtype=np.float32)[:len(chunk)].copy()
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                if not np.isfinite(vectors).all():
                    raise ValueError("Nonfinite MTEB embedding")
                output.append(vectors)
                self.encoded += len(chunk)
        return np.concatenate(output, axis=0) if output else np.empty((0, self.config.hidden_size), np.float32)


def run(plan_path, role, output, *, batch_size=16, progress=None):
    import mteb
    from mteb.models.abs_encoder import AbsEncoder

    if mteb.__version__ != MTEB_VERSION:
        raise ValueError(f"MTEB {MTEB_VERSION} required; found {mteb.__version__}")
    if jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise ValueError("MTEB model inference requires a physical single-host TPU")
    if role not in ("baseline", "candidate") or Path(output).exists():
        raise ValueError("Select one role and a fresh output")
    plan, identity, candidate_config, baseline_config = verify_plan(plan_path)
    snapshot = Path(plan["baseline"]["snapshot"])
    config = baseline_config if role == "baseline" else candidate_config
    session = (EmbeddingSession.from_pretrained(snapshot, config=config) if role == "baseline"
               else EmbeddingSession(plan["candidate"]["checkpoint"], step=plan["candidate"]["step"]))
    name = "jhu-clsp/mmBERT-base" if role == "baseline" else "mlnomad/yat-mmbert-base-contrastive"
    revision = plan["baseline"]["sha256"]["model.safetensors"] if role == "baseline" else plan["candidate"]["metadata_sha256"]
    class Adapter(MtebEncoder, AbsEncoder):
        pass

    model = Adapter(session, snapshot / "tokenizer.json", config,
                    plan["sequence_length"], name, revision)
    tasks = mteb.get_tasks(tasks=list(TASKS), eval_splits=["test"])
    if {task.metadata.name for task in tasks} != set(TASKS):
        raise ValueError("MTEB task selection differs from frozen plan")
    for task in tasks:
        if (task.metadata.dataset.get("revision") != TASK_REVISIONS[task.metadata.name]
                or list(task.eval_splits) != ["test"]):
            raise ValueError(f"MTEB task revision or test split changed: {task.metadata.name}")
    progress = Path(progress) if progress is not None else None
    if progress is not None and progress.exists():
        raise ValueError("Refusing existing MTEB progress receipt")
    raw_result = dict(model_name=name, model_revision=revision,
                      task_results=[], exceptions=None)
    for task in tasks:
        result = mteb.evaluate(model, tasks=[task],
                               encode_kwargs={"batch_size": batch_size},
                               cache=None, overwrite_strategy="always",
                               co2_tracker=False, show_progress_bar=False)
        partial = result.model_dump(mode="json")
        if (partial.get("exceptions") or len(partial.get("task_results", [])) != 1
                or partial["task_results"][0].get("task_name") != task.metadata.name):
            raise ValueError(f"MTEB task did not complete: {task.metadata.name}")
        task_result = partial["task_results"][0]
        if (task_result.get("dataset_revision") != TASK_REVISIONS[task.metadata.name]
                or set(task_result.get("scores", {})) != {"test"}
                or not task_result["scores"]["test"]):
            raise ValueError(f"MTEB task dataset or test scores changed: {task.metadata.name}")
        raw_result["task_results"].extend(partial["task_results"])
        if progress is not None:
            progress.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress.with_suffix(progress.suffix + ".tmp")
            temporary.write_text(json.dumps(dict(
                complete=False, role=role, plan_sha256=identity,
                completed_tasks=[item["task_name"] for item in raw_result["task_results"]],
                task_results=raw_result["task_results"],
                encoded_texts=model.encoded, truncated_texts=model.truncated,
            ), indent=2) + "\n")
            temporary.replace(progress)
    completed = {item["task_name"] for item in raw_result["task_results"]}
    if completed != set(TASKS) or raw_result.get("exceptions"):
        raise ValueError("MTEB did not complete every frozen task")
    if file_hash(plan_path) != identity:
        raise ValueError("MTEB plan changed during evaluation")
    output = Path(output)
    report = dict(role=role, plan_sha256=identity, model=name, revision=revision,
                  mteb_version=mteb.__version__, tasks=list(TASKS),
                  task_revisions=TASK_REVISIONS, eval_splits=["test"],
                  sequence_length=plan["sequence_length"], pooling=plan["pooling"],
                  normalization=plan["normalization"], prompts=plan["prompts"],
                  encoded_texts=model.encoded, truncated_texts=model.truncated,
                  backend=jax.default_backend(), devices=[str(device) for device in jax.devices()],
                  encoder_config=asdict(config),
                  result=raw_result,
                  full_paper_suite=False, source_sha256=file_hash(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(baseline_path, candidate_path, output):
    left = json.loads(Path(baseline_path).read_text())
    right = json.loads(Path(candidate_path).read_text())
    for key in ("plan_sha256", "mteb_version", "tasks", "task_revisions", "eval_splits",
                "sequence_length", "pooling",
                "normalization", "prompts", "source_sha256", "backend"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Unmatched MTEB protocol: {key}")
    if left.get("role") != "baseline" or right.get("role") != "candidate" or left.get("backend") != "tpu":
        raise ValueError("Require paired physical-TPU MTEB reports")
    per_task = {}
    left_tasks = {item["task_name"]: item for item in left["result"]["task_results"]}
    right_tasks = {item["task_name"]: item for item in right["result"]["task_results"]}
    if set(left_tasks) != set(TASKS) or set(right_tasks) != set(TASKS):
        raise ValueError("Incomplete paired MTEB task coverage")
    for name in TASKS:
        before, after = left_tasks[name], right_tasks[name]
        if before.get("dataset_revision") != after.get("dataset_revision") or before.get("mteb_version") != after.get("mteb_version"):
            raise ValueError(f"Mismatched MTEB task provenance: {name}")
        if before.get("scores", {}).keys() != after.get("scores", {}).keys():
            raise ValueError(f"Mismatched MTEB splits: {name}")
        for split in before["scores"]:
            left_subsets = [row.get("hf_subset") for row in before["scores"][split]]
            right_subsets = [row.get("hf_subset") for row in after["scores"][split]]
            if (not left_subsets or len(left_subsets) != len(set(left_subsets))
                    or left_subsets != right_subsets):
                raise ValueError(f"Mismatched MTEB subsets: {name}/{split}")
        per_task[name] = dict(dataset_revision=before.get("dataset_revision"),
                              baseline=before["scores"], candidate=after["scores"])
    report = dict(plan_sha256=left["plan_sha256"], tasks=left["tasks"],
                  per_task=per_task,
                  baseline_truncated_texts=left["truncated_texts"],
                  candidate_truncated_texts=right["truncated_texts"],
                  full_paper_suite=False,
                  limitations=["Diagnostic MTEB task slice, not the full mmBERT paper evaluation.",
                               "No training-data contamination clearance is inferred."])
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-plan", action="store_true")
    group.add_argument("--role", choices=("baseline", "candidate"))
    group.add_argument("--compare", action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int)
    parser.add_argument("--snapshot")
    parser.add_argument("--baseline-report")
    parser.add_argument("--candidate-report")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.build_plan:
        result = build_plan(args.checkpoint, args.step, args.snapshot, args.output,
                            sequence_length=args.sequence_length)
    elif args.compare:
        result = compare(args.baseline_report, args.candidate_report, args.output)
    else:
        result = run(args.plan, args.role, args.output, batch_size=args.batch_size,
                     progress=args.progress)
    print(json.dumps({key: value for key, value in result.items() if key != "result"}))


if __name__ == "__main__":
    main()
