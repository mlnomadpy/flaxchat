"""Evaluate a pinned encoder on isolated development pairs using a physical TPU.

The candidate pool is the complete prepared dev split in both directions. This
is an internal retrieval diagnostic, not a Tatoeba test or MTEB leaderboard run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


PARENT_STEP = 48236
PARENT_FAMILY = "modernbert"
CONTRASTIVE_FAMILY = "modernbert_contrastive_encoder"
PARENT_METADATA_SHA256 = "35cff96b97c0a9d5ba401c39545dfeff62e1c73e527d5f5661ba0960042bf4f4"


def ranking_metrics(
    scores: np.ndarray,
    query_text_ids: np.ndarray,
    document_text_ids: np.ndarray,
    positive_group_ids: np.ndarray,
    *,
    anchor_start: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rank known positives, excluding cross-side identical-text false negatives.

    ``scores`` contains consecutive query rows against the *entire* document
    pool. Equal scores break ties by original candidate index. Another row is a
    valid positive when it shares a query, document, or known-positive group ID
    with the anchor. Cross-side identical text not otherwise known positive is
    ignored, matching the training loss's false-negative policy.
    """
    scores = np.asarray(scores)
    query_text_ids = np.asarray(query_text_ids)
    document_text_ids = np.asarray(document_text_ids)
    positive_group_ids = np.asarray(positive_group_ids)
    count = len(query_text_ids)
    if (
        scores.ndim != 2
        or scores.shape[1] != count
        or scores.shape[0] < 1
        or anchor_start < 0
        or anchor_start + len(scores) > count
        or document_text_ids.shape != (count,)
        or positive_group_ids.shape != (count,)
        or any(not np.issubdtype(x.dtype, np.integer) for x in
               (query_text_ids, document_text_ids, positive_group_ids))
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Require finite score rows and complete integer pair IDs")
    indices = np.arange(count)
    ranks = np.empty(len(scores), np.int32)
    positive_counts = np.empty(len(scores), np.int32)
    ignored_counts = np.empty(len(scores), np.int32)
    for local, values in enumerate(scores):
        anchor = anchor_start + local
        positive = (
            (query_text_ids == query_text_ids[anchor])
            | (document_text_ids == document_text_ids[anchor])
            | (positive_group_ids == positive_group_ids[anchor])
        )
        # A literal query/document text match can be a translation benchmark
        # artifact or a false negative; do not promote it to a known positive.
        ignored = (document_text_ids == query_text_ids[anchor]) & ~positive
        eligible = ~ignored
        if not positive[anchor] or not positive.any() or not eligible[anchor]:
            raise ValueError("Aligned development pair lost its known positive")
        best_score = values[positive].max()
        best_index = int(indices[positive & (values == best_score)].min())
        better = eligible & ~positive & (values > best_score)
        earlier_tie = eligible & ~positive & (values == best_score) & (indices < best_index)
        ranks[local] = 1 + int(better.sum()) + int(earlier_tie.sum())
        positive_counts[local] = int(positive.sum())
        ignored_counts[local] = int(ignored.sum())
    return ranks, positive_counts, dict(
        ignored_cross_text_candidates=int(ignored_counts.sum()),
        anchors_with_ignored_candidates=int(np.count_nonzero(ignored_counts)),
        multi_positive_anchors=int(np.count_nonzero(positive_counts > 1)),
        maximum_positive_count=int(positive_counts.max()),
    )


def summarize_ranks(ranks: np.ndarray) -> dict:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or not len(ranks) or not np.issubdtype(ranks.dtype, np.integer) or np.any(ranks < 1):
        raise ValueError("Require nonempty positive integer ranks")
    return dict(
        recall_at_1=float(np.mean(ranks == 1)),
        mrr=float(np.mean(1.0 / ranks.astype(np.float64))),
        anchors=int(len(ranks)),
    )


def _metadata_sha256(metadata: dict) -> str:
    # Older checkpoint receipts omit the redundant directory-derived step.
    persisted = {key: value for key, value in metadata.items() if key != "step"}
    return hashlib.sha256(json.dumps(persisted, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()


def evaluate(args: argparse.Namespace) -> dict:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from flaxchat.checkpoint import load_checkpoint_metadata, restore_model_from_checkpoint
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.encoder_data import file_hash
    from flaxchat.runtime import runtime_identity
    from scripts.prepare_encoder_contrastive import load_tokenized, rows

    if jax.default_backend() != "tpu" or jax.process_count() != 1 or jax.device_count() < 1:
        raise ValueError("This development retrieval evaluation requires a physical TPU on one host")
    # Retrieval is unsharded. Keep all model state, embeddings and score blocks
    # on the same physical chip even when the training VM exposes a larger slice.
    evaluation_device = jax.devices()[0]
    jax.config.update("jax_default_device", evaluation_device)
    if args.batch_size < 1 or args.score_block_size < 1 or args.checkpoint_step < 1:
        raise ValueError("Positive batch, score block and explicit checkpoint step required")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite an evaluation report")
    metadata = load_checkpoint_metadata(args.checkpoint, step=args.checkpoint_step)
    family = metadata.get("model_family")
    if family not in (PARENT_FAMILY, CONTRASTIVE_FAMILY):
        raise ValueError("Expected an MLM parent or contrastive encoder checkpoint")
    if family == PARENT_FAMILY and args.checkpoint_step != PARENT_STEP:
        raise ValueError("MLM control must use committed parent step 48236")
    if family == PARENT_FAMILY and _metadata_sha256(metadata) != PARENT_METADATA_SHA256:
        raise ValueError("MLM control is not the pinned final parent checkpoint")
    if family == CONTRASTIVE_FAMILY and (
        metadata.get("resolved_config", {}).get("parent", {}).get("step") != PARENT_STEP
        or metadata.get("resolved_config", {}).get("parent", {}).get("metadata_sha256")
        != PARENT_METADATA_SHA256
    ):
        raise ValueError("Contrastive checkpoint was not initialized from the pinned MLM parent")
    tokenizer_hash = file_hash(args.tokenizer)
    if metadata.get("tokenizer_identity") != tokenizer_hash:
        raise ValueError("Checkpoint and development tokenizer differ")
    data_root = Path(args.data)
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tokenization = manifest.get("tokenization")
    if not isinstance(tokenization, dict):
        raise ValueError("Prepared development pairs must be tokenized")
    sequence_length = tokenization.get("sequence_length")
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("Invalid prepared development sequence length")
    arrays, checked_manifest = load_tokenized(
        data_root, args.tokenizer, split="dev", sequence_length=sequence_length,
    )
    if checked_manifest != manifest:
        raise ValueError("Development manifest changed during loading")
    data_hash = file_hash(manifest_path)
    if family == CONTRASTIVE_FAMILY and metadata.get("data_manifest_identity") != data_hash:
        raise ValueError("Contrastive checkpoint and development data differ")
    config = EncoderConfig(**metadata["resolved_config"]["encoder"])
    if sequence_length > config.max_position_embeddings or config.attention_backend != "xla":
        raise ValueError("Development length/backend is incompatible with checkpoint")
    if tokenization.get("pad_token_id") != config.pad_token_id or tokenization.get("vocab_size") != config.vocab_size:
        raise ValueError("Prepared token IDs do not match checkpoint")
    count = len(arrays["query_tokens"])
    if count < 2 or any(np.all(arrays[name] == config.pad_token_id, axis=1).any()
                        for name in ("query_tokens", "document_tokens")):
        raise ValueError("Need at least two nonempty development pairs")

    model = ModernBert(config, rngs=nnx.Rngs(0))
    restored = restore_model_from_checkpoint(
        model, args.checkpoint, step=args.checkpoint_step,
        expected_identity=dict(resolved_config=metadata["resolved_config"],
                               tokenizer=tokenizer_hash),
    )
    if restored != metadata:
        raise ValueError("Checkpoint metadata changed during restore")

    @nnx.jit
    def embed(inner, tokens):
        pooled = inner.pool(tokens).astype(jnp.float32)
        return pooled / jnp.maximum(jnp.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

    vectors = {}
    started = time.monotonic()
    for side in ("query", "document"):
        source = arrays[side + "_tokens"]
        result = np.empty((count, config.hidden_size), np.float32)
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            batch = np.full((args.batch_size, sequence_length), config.pad_token_id, np.int32)
            batch[: stop - start] = source[start:stop]
            result[start:stop] = np.asarray(embed(model, jnp.asarray(batch)))[: stop - start]
        if not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) < 0.99):
            raise ValueError("Invalid normalized development embedding")
        vectors[side] = result

    @jax.jit
    def score_block(left, right):
        return jnp.matmul(left, right.T, precision=jax.lax.Precision.HIGHEST)

    q_ids = np.asarray(arrays["query_text_ids"])
    d_ids = np.asarray(arrays["document_text_ids"])
    groups = np.asarray(arrays["positive_group_ids"])
    directions = {}
    all_ranks = {}
    for direction, left_name, right_name, left_ids, right_ids in (
        ("query_to_document", "query", "document", q_ids, d_ids),
        ("document_to_query", "document", "query", d_ids, q_ids),
    ):
        right = jnp.asarray(vectors[right_name])
        ranks = np.empty(count, np.int32)
        diagnostics = {"ignored_cross_text_candidates": 0,
                       "anchors_with_ignored_candidates": 0,
                       "multi_positive_anchors": 0,
                       "maximum_positive_count": 0}
        for start in range(0, count, args.score_block_size):
            stop = min(start + args.score_block_size, count)
            left = jnp.asarray(vectors[left_name][start:stop])
            scores = np.asarray(score_block(left, right))
            block_ranks, _, block = ranking_metrics(
                scores, left_ids, right_ids, groups, anchor_start=start,
            )
            ranks[start:stop] = block_ranks
            for key, value in block.items():
                if key == "maximum_positive_count":
                    diagnostics[key] = max(diagnostics[key], value)
                else:
                    diagnostics[key] += value
        directions[direction] = summarize_ranks(ranks) | diagnostics
        all_ranks[direction] = ranks
    per_language_pair = {}
    dev_path = data_root / "dev.jsonl"
    language_pairs = [row["language1"] + ":" + row["language2"]
                      for _, row in rows(dev_path)]
    if len(language_pairs) != count:
        raise ValueError("Development text/token row count changed")
    for pair in sorted(set(language_pairs)):
        selected = np.asarray([item == pair for item in language_pairs])
        per_language_pair[pair] = {
            direction: summarize_ranks(ranks[selected])
            for direction, ranks in all_ranks.items()
        }
    if file_hash(manifest_path) != data_hash:
        raise ValueError("Development data changed during evaluation")
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {name: file_hash(source_root / name) for name in (
        "scripts/evaluate_encoder_contrastive.py", "scripts/prepare_encoder_contrastive.py",
        "flaxchat/encoder.py", "flaxchat/checkpoint.py",
    )}
    report = dict(
        scope="isolated_contrastive_development_retrieval",
        production_quality_qualified=False,
        official_mteb_parity=False,
        split="dev",
        pairs=count,
        checkpoint=dict(path=args.checkpoint, step=args.checkpoint_step,
                        model_family=family, metadata_sha256=_metadata_sha256(metadata),
                        source_python_sha256=metadata.get("source_python_sha256")),
        data=dict(path=str(data_root), manifest_sha256=data_hash,
                  dev_jsonl_sha256=file_hash(dev_path),
                  tokenizer_sha256=tokenizer_hash,
                  source_manifest_sha256=manifest["source_manifest_sha256"],
                  sealed_manifest_sha256=manifest["sealed_manifest_sha256"],
                  sequence_length=sequence_length),
        method=dict(pooling="mean of nonpadding token states, including special tokens",
                    normalization="L2 FP32", similarity="TPU FP32 cosine dot product",
                    tie_policy="ascending prepared pair index",
                    positives="same query text, document text, or known-positive group",
                    false_negatives="exclude cross-side identical text unless known positive",
                    candidate_pool="all dev rows in both directions"),
        metrics=directions,
        per_language_pair=per_language_pair,
        backend=jax.default_backend(),
        evaluation_device=str(evaluation_device),
        visible_devices=[str(device) for device in jax.devices()],
        runtime=runtime_identity(),
        evaluator_source_sha256=source_hashes,
        elapsed_seconds=time.monotonic() - started,
        limitations=[
            "Development-only retrieval diagnostic; the Tatoeba test split is sealed.",
            "Exact normalized text overlap was excluded during preparation; semantic and pretrained exposure are not cleared.",
            "Mean pooling has not been established as an optimal embedding recipe.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", "--checkpoint-step", dest="checkpoint_step", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-block-size", type=int, default=64)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(dict(scope=report["scope"], pairs=report["pairs"],
                          checkpoint=report["checkpoint"], metrics=report["metrics"])), flush=True)


if __name__ == "__main__":
    main()
