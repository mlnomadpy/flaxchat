"""Exact cosine retrieval with bounded score blocks and explicit metric semantics."""

import numpy as np


def rank_cosine(
    queries,
    corpus,
    *,
    query_ids,
    document_ids,
    k=10,
    query_block=32,
    document_block=4096,
    score_dtype="float32",
):
    # Keep the historical mode explicit for frozen campaigns. Float64 avoids
    # float32 accumulation reversing small but real cosine margins; it is not
    # a guarantee of bitwise equality across every numerical platform.
    if score_dtype not in ("float32", "float64"):
        raise ValueError("Scoring dtype must be float32 or float64")
    dtype = np.dtype(score_dtype)
    if any(type(n) is not int or n < 1 for n in (k, query_block, document_block)):
        raise ValueError("Positive integer ranking bounds required")
    for values, ids in ((queries, query_ids), (corpus, document_ids)):
        if (
            values.ndim != 2
            or len(values) != len(ids)
            or not len(ids)
            or not np.issubdtype(values.dtype, np.floating)
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Nonempty float embeddings and unique string IDs required")
    if queries.shape[1] != corpus.shape[1] or queries.shape[1] < 1:
        raise ValueError("Embedding dimensions must match")

    def normalized(values):
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite embedding")
        scale = np.max(np.abs(values), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("Zero embedding cannot define cosine similarity")
        values = values / scale
        return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(dtype)

    # Stable secondary order uses document IDs, independent of input row order.
    order = np.argsort(np.asarray(document_ids))
    ids = [document_ids[int(i)] for i in order]
    k = min(k, len(corpus))
    rankings = {}
    for begin in range(0, len(queries), query_block):
        q = normalized(queries[begin : begin + query_block])
        scores = np.empty((len(q), 0), dtype)
        indices = np.empty((len(q), 0), np.int64)
        for start in range(0, len(corpus), document_block):
            end = min(len(corpus), start + document_block)
            block_scores = q @ normalized(corpus[order[start:end]]).T
            block_ids = np.broadcast_to(np.arange(start, end), block_scores.shape)
            scores = np.concatenate((scores, block_scores), axis=1)
            indices = np.concatenate((indices, block_ids), axis=1)
            selected = np.lexsort((indices, -scores), axis=1)[:, :k]
            scores = np.take_along_axis(scores, selected, axis=1)
            indices = np.take_along_axis(indices, selected, axis=1)
        for i, row in enumerate(indices):
            rankings[query_ids[begin + i]] = [ids[int(index)] for index in row]
    return rankings


def retrieval_metrics(rankings, qrels, *, document_ids, languages, cutoffs=(1, 10)):
    """Query means, per-language means and language macro; linear-gain nDCG.

    Missing judgments/positive documents are errors, never silently dropped.
    Unjudged retrieved documents count as nonrelevant. Every query needs a
    positive judgment. Tie policy is supplied by rank_cosine, not trec_eval.
    """
    if not rankings or set(rankings) != set(qrels) or set(languages) != set(qrels):
        raise ValueError(
            "Rankings, judgments and languages must cover identical queries"
        )
    if (
        not cutoffs
        or len(set(cutoffs)) != len(cutoffs)
        or any(type(k) is not int or k < 1 for k in cutoffs)
    ):
        raise ValueError("Distinct positive cutoffs required")
    universe = set(document_ids)
    if len(universe) != len(document_ids):
        raise ValueError("Duplicate document IDs")
    per_query = {}
    for query, ranked in rankings.items():
        judged = qrels[query]
        if (
            len(set(ranked)) != len(ranked)
            or not set(ranked) <= universe
            or len(ranked) < min(max(cutoffs), len(universe))
            or not set(judged) <= universe
            or not judged
            or any(type(g) is not int or g < 0 for g in judged.values())
            or not any(g > 0 for g in judged.values())
            or not isinstance(languages[query], str)
            or not languages[query]
        ):
            raise ValueError("Invalid or incomplete retrieval evidence")
        positive = sum(g > 0 for g in judged.values())
        ideal = sorted(judged.values(), reverse=True)
        metrics = {}
        for k in cutoffs:
            gains = np.asarray(
                [judged.get(doc, 0) for doc in ranked[:k]], dtype=np.float64
            )
            hits = np.flatnonzero(gains > 0)
            ideal_gains = np.asarray(ideal[:k], dtype=np.float64)
            dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))
            idcg = np.sum(ideal_gains / np.log2(np.arange(len(ideal_gains)) + 2))
            metrics.update(
                {
                    f"recall@{k}": len(hits) / positive,
                    f"mrr@{k}": 1 / (int(hits[0]) + 1) if len(hits) else 0.0,
                    f"ndcg@{k}": float(dcg / idcg),
                }
            )
        per_query[query] = metrics

    def average(rows):
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    per_language = {
        language: average(
            [row for q, row in per_query.items() if languages[q] == language]
        )
        for language in sorted(set(languages.values()))
    }
    return dict(
        query_count=len(per_query),
        per_query=per_query,
        per_language=per_language,
        query_mean=average(list(per_query.values())),
        language_macro=average(list(per_language.values())),
        ndcg_gain="linear relevance grade",
        unjudged_policy="nonrelevant",
    )
