"""Execute one model's full frozen bitext campaign; provision no resources.

Run baseline and candidate in separate processes. The TPU worker must enforce
an external hard deadline; the checks here are cooperative between subsets.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from flaxchat.checkpoint import load_checkpoint_metadata
from flaxchat.encoder import EncoderConfig
from flaxchat.encoder_data import file_hash
from scripts.evaluate_encoder_bitext import evaluate_campaign
from scripts.export_encoder_retrieval import EmbeddingSession
from scripts.workload_deadline import WorkloadDeadline


_RECIPE = {
    "evaluation_split": "test",
    "pooling": "mean of nonpadding token states, including special tokens",
    "similarity": "cosine",
    "direction": "sentence1 queries to sentence2 corpus per subset",
    "tie_policy": "ascending document ID",
    "primary_metric": "subset-macro support-weighted F1",
    "secondary_metric": "subset-macro pair accuracy",
    "all_pairs_required": True,
    "filtering_or_truncation_allowed": False,
}


def verify_plan(path):
    """Verify complete prepared inventory and every frozen file before model load."""
    path = Path(path)
    identity = file_hash(path)
    plan = json.loads(path.read_text())
    if any(plan.get(k) != v for k, v in _RECIPE.items()):
        raise ValueError("Unsupported frozen bitext recipe")
    if type(plan.get("sequence_length")) is not int or plan["sequence_length"] < 1:
        raise ValueError("Positive sequence length required")
    if type(plan["candidate"].get("step")) is not int or plan["candidate"]["step"] < 1:
        raise ValueError("Explicit positive checkpoint step required")
    if plan["candidate"].get("model_family", "modernbert") not in (
        "modernbert", "modernbert_contrastive_encoder"
    ):
        raise ValueError("Unsupported candidate model family")
    pinned_metadata = plan["candidate"].get("metadata_sha256")
    if pinned_metadata is not None and (
        not isinstance(pinned_metadata, str) or len(pinned_metadata) != 64
        or any(char not in "0123456789abcdef" for char in pinned_metadata)
    ):
        raise ValueError("Invalid candidate metadata SHA-256")
    baseline_config = plan["baseline"].get("encoder_config")
    if baseline_config is not None:
        if not isinstance(baseline_config, dict):
            raise ValueError("Invalid pinned baseline encoder configuration")
        EncoderConfig(**baseline_config)
    if "training_overlap_audit" in plan:
        audit_record = plan["training_overlap_audit"]
        audit_path = Path(audit_record["path"])
        if file_hash(audit_path) != audit_record["sha256"]:
            raise ValueError("Frozen Tatoeba overlap audit changed")
        audit = json.loads(audit_path.read_text())
        if (audit.get("passed") is not True or audit.get("overlapping_unique_texts") != 0
                or audit.get("prepared_manifest_sha256") != audit_record["prepared_manifest_sha256"]):
            raise ValueError("Invalid frozen Tatoeba overlap audit")
    inventory = Path(plan["inventory_path"])
    if file_hash(inventory) != plan["inventory_sha256"]:
        raise ValueError("Frozen inventory hash mismatch")
    declared = json.loads(inventory.read_text())
    entries = declared["shards"]
    subsets = [e["subset"] for e in entries]
    if (
        not subsets
        or len(subsets) != len(set(subsets))
        or any(
            not isinstance(s, str) or Path(s).name != s or s in (".", "..")
            for s in subsets
        )
        or any(type(e["pairs"]) is not int or e["pairs"] < 1 for e in entries)
        or len(entries) != plan["subsets"]
        or sum(e["pairs"] for e in entries) != plan["pairs"]
        or any(declared[k] != plan[k] for k in ("dataset", "revision"))
    ):
        raise ValueError("Frozen subset inventory mismatch")
    frozen = plan["input_sha256"]
    directories = {}
    expected = set()
    for entry in entries:
        subset = entry["subset"]
        matches = [
            Path(p).parent
            for p in frozen
            if Path(p).name == "judgments.json" and Path(p).parent.name == subset
        ]
        if len(matches) != 1:
            raise ValueError("Exactly one frozen judgments file per subset required")
        directory = directories[subset] = matches[0]
        paths = [directory / "judgments.json"] + [
            directory / role / name
            for role in ("queries", "corpus")
            for name in ("manifest.json", "tokens.npy")
        ]
        for item in paths:
            expected.add(str(item))
            if str(item) not in frozen or file_hash(item) != frozen[str(item)]:
                raise ValueError(f"Frozen input hash mismatch: {item}")
        task = json.loads((directory / "judgments.json").read_text())
        if (
            task.get("subset") != subset
            or task.get("split") != "test"
            or any(task.get(k) != plan[k] for k in ("dataset", "revision"))
        ):
            raise ValueError("Subset task identity mismatch")
        for role, ids in (("queries", "query_ids"), ("corpus", "document_ids")):
            rows = np.load(
                directory / role / "tokens.npy", mmap_mode="r", allow_pickle=False
            )
            if (
                rows.shape != (entry["pairs"], plan["sequence_length"])
                or len(task[ids]) != entry["pairs"]
            ):
                raise ValueError("Frozen pair count or sequence length mismatch")
    if set(frozen) != expected:
        raise ValueError("Frozen file inventory must cover exactly all prepared inputs")
    snapshot = Path(plan["baseline"]["snapshot"])
    if file_hash(snapshot / "model.safetensors") != plan["baseline"]["weights_sha256"]:
        raise ValueError("Released weights hash mismatch")
    snapshot_hashes = {
        name: file_hash(snapshot / name)
        for name in ("config.json", "tokenizer.json", "model.safetensors")
    }
    if file_hash(path) != identity or file_hash(inventory) != plan["inventory_sha256"]:
        raise ValueError("Plan or inventory changed during verification")
    return plan, identity, entries, directories, snapshot_hashes


def _metadata_sha256(metadata):
    persisted = {key: value for key, value in metadata.items() if key != "step"}
    return hashlib.sha256(json.dumps(persisted, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()


def run(plan_path, role, output, *, batch_size=8, max_seconds=3600):
    if role not in ("baseline", "candidate"):
        raise ValueError("Baseline or candidate role required")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive batch size required")
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing existing campaign output")
    deadline = WorkloadDeadline(max_seconds)
    started = time.monotonic()
    plan, identity, entries, directories, snapshot_hashes = verify_plan(plan_path)
    if plan.get("require_physical_tpu") and (
        jax.default_backend() != "tpu" or jax.process_count() != 1
    ):
        raise ValueError("Frozen paired benchmark requires a physical single-host TPU")
    verification_seconds = time.monotonic() - started
    candidate = plan["candidate"]
    metadata = load_checkpoint_metadata(candidate["checkpoint"], step=candidate["step"])
    if (
        metadata.get("model_family") != candidate.get("model_family", "modernbert")
        or metadata.get("step") != candidate["step"]
        or (candidate.get("metadata_sha256") is not None and
            _metadata_sha256(metadata) != candidate["metadata_sha256"])
    ):
        raise ValueError("Candidate checkpoint identity mismatch")
    config = EncoderConfig(**metadata["resolved_config"]["encoder"])
    baseline_config = EncoderConfig(**plan["baseline"].get("encoder_config", asdict(config)))
    if metadata["tokenizer_identity"] != snapshot_hashes["tokenizer.json"]:
        raise ValueError("Candidate and released-mmBERT tokenizers differ")
    deadline.remaining()
    load_started = time.monotonic()
    session = (
        EmbeddingSession.from_pretrained(plan["baseline"]["snapshot"], config=baseline_config)
        if role == "baseline"
        else EmbeddingSession(candidate["checkpoint"], step=candidate["step"])
    )
    load_seconds = time.monotonic() - load_started
    output.mkdir(parents=True)
    progress = []
    manifests = []
    for entry in entries:
        deadline.remaining()
        subset = entry["subset"]
        directory = directories[subset]
        before = time.monotonic()
        manifest = session.export(
            directory / "queries",
            directory / "corpus",
            directory / "judgments.json",
            output / "embeddings" / subset,
            batch_size=batch_size,
        )
        if (
            manifest["encoder_config"] != asdict(baseline_config if role == "baseline" else config)
            or manifest["pooling"] != plan["pooling"]
        ):
            raise ValueError("Matched inference configuration mismatch")
        if role == "candidate" and (
            manifest["checkpoint"] != candidate["checkpoint"]
            or manifest["checkpoint_step"] != candidate["step"]
        ):
            raise ValueError("Exported checkpoint identity mismatch")
        manifests.append(
            dict(
                subset=subset,
                sha256=file_hash(output / "embeddings" / subset / "manifest.json"),
            )
        )
        progress.append(
            dict(
                subset=subset,
                pairs=entry["pairs"],
                export_seconds=time.monotonic() - before,
            )
        )
        temporary = output / "progress.tmp"
        temporary.write_text(
            json.dumps(
                dict(
                    complete=False,
                    role=role,
                    plan_sha256=identity,
                    completed_subsets=progress,
                ),
                indent=2,
            )
            + "\n"
        )
        temporary.replace(output / "progress.json")
        print(json.dumps(progress[-1]), flush=True)
    deadline.remaining()
    scoring_started = time.monotonic()
    scores = evaluate_campaign(output / "embeddings", plan["inventory_path"])
    scoring_seconds = time.monotonic() - scoring_started
    if (
        scores["subset_count"] != plan["subsets"]
        or scores["pair_count"] != plan["pairs"]
    ):
        raise ValueError("Incomplete evaluation coverage")
    # Detect mutations across the campaign, including across different subsets.
    after, after_identity, _, _, after_snapshot = verify_plan(plan_path)
    if after != plan or after_identity != identity or after_snapshot != snapshot_hashes:
        raise ValueError("Frozen campaign inputs changed during execution")
    deadline.remaining()
    report = dict(
        complete=True,
        role=role,
        plan_sha256=identity,
        batch_size=batch_size,
        verification_seconds=verification_seconds,
        model_load_seconds=load_seconds,
        scoring_seconds=scoring_seconds,
        total_seconds=time.monotonic() - started,
        subset_timings=progress,
        manifests=manifests,
        snapshot_sha256=snapshot_hashes,
        candidate=candidate,
        encoder_config=asdict(baseline_config if role == "baseline" else config),
        scores=scores,
        backend=jax.default_backend(),
        jax_version=jax.__version__,
        available_devices=[str(d) for d in jax.devices()],
        execution="single-process unsharded model; available devices do not prove utilization",
        production_quality_qualified=False,
        official_mteb_parity=False,
        source_sha256={
            str(p): file_hash(p)
            for p in (
                Path(__file__),
                Path("scripts/export_encoder_retrieval.py"),
                Path("scripts/evaluate_encoder_bitext.py"),
                Path("flaxchat/bitext.py"),
            )
        },
    )
    temporary = output / "report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    report = run(
        args.plan,
        args.role,
        args.output,
        batch_size=args.batch_size,
        max_seconds=args.max_seconds,
    )
    print(
        json.dumps(
            {k: report[k] for k in ("role", "complete", "total_seconds", "plan_sha256")}
        )
    )


if __name__ == "__main__":
    main()
