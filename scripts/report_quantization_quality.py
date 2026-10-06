#!/usr/bin/env python3
"""Model-free reporting of frozen physical-TPU quantization output receipts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flaxchat.embedding_uncertainty import bootstrap_paired_delta


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def pinned_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return json.loads(Path(path).read_text())


def distribution(values):
    values = np.asarray(values, np.float64)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        **{f"p{p}": float(np.percentile(values, p)) for p in (1, 5, 50, 95, 99)},
    }


def normalized(values):
    values = np.asarray(values, np.float32)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("Nonfinite or zero embedding")
    return values / norms[:, None]


def ranks(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.r_[
        0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(values)
    ]
    result = np.empty(len(values), np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        result[order[start:stop]] = (start + stop - 1) / 2 + 1
    return result


def spearman(a, b):
    a, b = ranks(a), ranks(b)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        raise ValueError("Spearman undefined for constant ranks")
    return float(a @ b / denominator)


def retrieval_scores(top_ids, judgments):
    gains = np.array([max(0, judgments.get(doc, 0)) for doc in top_ids], np.float64)
    positive_count = sum(value > 0 for value in judgments.values())
    if not positive_count:
        raise ValueError("Retrieval query has no relevant documents")
    ideal = np.array(sorted((max(0, v) for v in judgments.values()), reverse=True)[:10])
    discounts = 1 / np.log2(np.arange(2, 12))
    ideal_dcg = float(((2**ideal - 1) * discounts[: len(ideal)]).sum())
    hits = np.flatnonzero(gains > 0)
    return {
        "ndcg_at_10": float(
            ((2**gains - 1) * discounts[: len(gains)]).sum() / ideal_dcg
        ),
        "recall_at_10": float(len(hits) / positive_count),
        "mrr_at_10": float(1 / (hits[0] + 1)) if len(hits) else 0.0,
    }


def summarize(plan, rows, receipt, original, quantized):
    row_ids = [r["id"] for r in rows]
    if len(set(row_ids)) != len(rows):
        raise ValueError("Duplicate fixture rows")
    if receipt.get("rows") != [
        {k: v for k, v in r.items() if k != "text"} for r in rows
    ]:
        raise ValueError("Receipt row identity/order mismatch")
    if original.shape != quantized.shape or original.shape != (len(rows), 768):
        raise ValueError("Incomplete or incorrect output shapes")
    index = {key: position for position, key in enumerate(row_ids)}
    a, b = normalized(original), normalized(quantized)
    cosine = np.clip(np.einsum("ij,ij->i", a, b), -1, 1)
    relative_l2 = np.linalg.norm(
        original.astype(np.float64) - quantized, axis=1
    ) / np.linalg.norm(original.astype(np.float64), axis=1)
    result = {
        "embedding_cosine": distribution(cosine),
        "cosine_distance": distribution(1 - cosine),
        "raw_relative_l2": distribution(relative_l2),
        "per_text": [
            {
                "id": key,
                "cosine": float(cosine[i]),
                "relative_l2": float(relative_l2[i]),
            }
            for i, key in enumerate(row_ids)
        ],
    }
    pairs = plan["sts_pairs"]
    scores_a = np.array([a[index[p["a"]]] @ a[index[p["b"]]] for p in pairs])
    scores_b = np.array([b[index[p["a"]]] @ b[index[p["b"]]] for p in pairs])
    gold = [p["score"] for p in pairs]
    rho_a, rho_b = spearman(scores_a, gold), spearman(scores_b, gold)
    result["stsbenchmark_test"] = {
        "pairs": len(pairs),
        "original_spearman": rho_a,
        "quantized_spearman": rho_b,
        "delta": rho_b - rho_a,
        "units": "correlation [-1,1]",
        "absolute_cosine_score_drift": distribution(abs(scores_b - scores_a)),
        "per_pair": [
            {
                "id": p["id"],
                "gold": p["score"],
                "original_cosine": float(x),
                "quantized_cosine": float(y),
            }
            for p, x, y in zip(pairs, scores_a, scores_b, strict=True)
        ],
    }
    corpus_ids = plan["scifact_corpus_ids"]
    judgments = plan["scifact_judgments"]
    query_ids = sorted(judgments)
    if len(set(corpus_ids)) != len(corpus_ids) or any(
        set(j) - set(corpus_ids) for j in judgments.values()
    ):
        raise ValueError("Invalid corpus/qrels identity")
    corpus_indices = [index[key] for key in corpus_ids]
    query_indices = [index[key] for key in query_ids]
    scores_a = a[query_indices] @ a[corpus_indices].T
    scores_b = b[query_indices] @ b[corpus_indices].T
    metrics_a = {key: {} for key in ("ndcg_at_10", "recall_at_10", "mrr_at_10")}
    metrics_b = {key: {} for key in metrics_a}
    per_query, agreement1, agreement10 = [], [], []
    for position, query_id in enumerate(query_ids):
        # Stable descending order breaks exact ties by the frozen corpus order.
        top_a = [
            corpus_ids[i] for i in np.argsort(-scores_a[position], kind="stable")[:10]
        ]
        top_b = [
            corpus_ids[i] for i in np.argsort(-scores_b[position], kind="stable")[:10]
        ]
        ma, mb = (
            retrieval_scores(top_a, judgments[query_id]),
            retrieval_scores(top_b, judgments[query_id]),
        )
        for metric in metrics_a:
            metrics_a[metric][query_id], metrics_b[metric][query_id] = (
                ma[metric],
                mb[metric],
            )
        agreement1.append(top_a[0] == top_b[0])
        agreement10.append(len(set(top_a) & set(top_b)) / len(top_a))
        per_query.append(
            {
                "id": query_id,
                "original": ma,
                "quantized": mb,
                "original_top10": top_a,
                "quantized_top10": top_b,
                "original_top10_cosine": [
                    float(scores_a[position, corpus_ids.index(key)]) for key in top_a
                ],
                "quantized_top10_cosine": [
                    float(scores_b[position, corpus_ids.index(key)]) for key in top_b
                ],
            }
        )
    qidentity = identity(
        {"query_ids": query_ids, "corpus_ids": corpus_ids, "qrels": judgments}
    )
    protocol = identity(
        {
            "fixture_sha256": plan["fixture_sha256"],
            "length": plan["sequence_length"],
            "metric": "cosine rank; stable corpus order ties; exponential gain ndcg@10; recall@10; reciprocal first relevant rank@10",
        }
    )
    retrieval = {
        "queries": len(query_ids),
        "corpus_documents": len(corpus_ids),
        "top1_agreement": float(np.mean(agreement1)),
        "top10_set_overlap_fraction": float(np.mean(agreement10)),
        "absolute_cosine_score_drift": distribution(abs(scores_b - scores_a)),
        "per_query": per_query,
        "units": "metrics [0,1]",
        "tie_policy": "frozen corpus order",
        "paired_delta_intervals": {},
    }
    for metric in metrics_a:
        retrieval["paired_delta_intervals"][metric] = bootstrap_paired_delta(
            metrics_a[metric],
            metrics_b[metric],
            metric=metric,
            baseline_query_identity_sha256=qidentity,
            candidate_query_identity_sha256=qidentity,
            baseline_protocol_identity_sha256=protocol,
            candidate_protocol_identity_sha256=protocol,
            resamples=2000,
            seed=0,
        )
    result["scifact_test"] = retrieval
    diagnostic_rows = rows[: plan["diagnostic_rows"]]
    if len(diagnostic_rows) == 88:
        # Small generated fixture, explicitly separate from independent public benchmarks.
        diag = {"scope": "88 handcrafted diagnostic texts, not an official benchmark"}
        for label, values in [("original", a[:88]), ("quantized", b[:88])]:
            english = [
                i for i, r in enumerate(diagnostic_rows) if r.get("domain") == "english"
            ]
            translation = [
                i
                for i, r in enumerate(diagnostic_rows)
                if r.get("domain") == "multilingual"
            ]
            if not translation:
                translation = list(range(8, 64))
            code_queries = list(range(64, 72))
            code_docs = list(range(72, 88))
            translation_hits = [
                diagnostic_rows[english[int(np.argmax(values[i] @ values[english].T))]][
                    "group"
                ]
                == diagnostic_rows[i]["group"]
                for i in translation
            ]
            code_hits = [
                diagnostic_rows[
                    code_docs[int(np.argmax(values[i] @ values[code_docs].T))]
                ]["group"]
                == diagnostic_rows[i]["group"]
                for i in code_queries
            ]
            diag[label] = {
                "translation_top1": float(np.mean(translation_hits)),
                "code_top1": float(np.mean(code_hits)),
            }
        result["curated_diagnostic"] = diag
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("plan", "fixture", "receipt", "vectors", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--role", choices=("yat", "mmbert"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite prior report")
    plan = pinned_json(args.plan, args.plan_sha256)
    rows = pinned_json(args.fixture, args.fixture_sha256)
    receipt = json.loads(args.receipt.read_text())
    hardware = receipt.get("hardware", {})
    if (
        plan.get("fixture_sha256") != args.fixture_sha256
        or plan.get("rows") != len(rows)
        or receipt.get("format") != "flaxchat-int8-native-tpu-quality-v1"
        or receipt.get("complete") is not True
        or receipt.get("role") != args.role
        or receipt.get("fixture_sha256") != args.fixture_sha256
        or "TPU" not in hardware.get("kind", "").upper()
        or hardware.get("utilized_devices") != 1
        or receipt.get("pooled_npz_sha256") != digest(args.vectors)
        or receipt.get("original_pooled_rows") != len(rows)
        or receipt.get("quantized_pooled_rows") != len(rows)
        or receipt.get("sequence_length") != plan.get("sequence_length")
        or receipt.get("batch_size") != plan.get("batch_size")
        or receipt.get("original_first_batch_repeat_bitwise_equal") is not True
        or receipt.get("quantized_outputs_changed") is not True
        or not receipt.get("artifacts_sha256")
        or not receipt.get("source_sha256")
        or not receipt.get("runtime", {}).get("packages", {}).get("libtpu")
        or len(receipt.get("input_int32_sha256", "")) != 64
    ):
        raise ValueError("Incomplete, unbound, or nonphysical TPU evidence")
    for variant in ("original", "quantized"):
        expected_start = 0
        for batch in (
            b for b in receipt.get("batches", []) if b.get("variant") == variant
        ):
            if batch.get("start") != expected_start or batch.get("rows") != min(
                8, len(rows) - expected_start
            ):
                raise ValueError("Batch coverage mismatch")
            expected_start += batch["rows"]
        if expected_start != len(rows):
            raise ValueError("Incomplete batch coverage")
    with np.load(args.vectors, allow_pickle=False) as values:
        if set(values.files) != {"original", "quantized"}:
            raise ValueError("Unexpected vector archive schema")
        report = summarize(plan, rows, receipt, values["original"], values["quantized"])
    report.update(
        format="flaxchat-quantization-quality-report-v1",
        role=args.role,
        complete=True,
        scope=receipt["scope"],
        reporting_only=True,
        serving_backend_qualified=False,
        full_mteb_qualified=False,
        parent_receipt_sha256=digest(args.receipt),
        vectors_sha256=digest(args.vectors),
        plan_sha256=args.plan_sha256,
        fixture_sha256=args.fixture_sha256,
        artifacts_sha256=receipt.get("artifacts_sha256"),
        source_sha256=receipt.get("source_sha256"),
        runtime=receipt.get("runtime"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
