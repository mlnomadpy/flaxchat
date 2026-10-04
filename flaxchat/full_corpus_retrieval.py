"""Host metrics over complete, already scored retrieval rankings; no encoders.

Exhaustive mode requires every document once per query. Scalable top-k mode
reports native metrics but explicitly leaves full-search qualification pending.
"""
from collections.abc import Mapping
import hashlib
import json
import math
import re


def identity(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Explicit artifact/protocol SHA256 required')
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not value or '\0' in value:
        raise ValueError('Nonempty string retrieval IDs required')
    return value


def validate_protocol(protocol):
    if not isinstance(protocol, Mapping) or protocol.get('format') != 'flaxchat-full-corpus-metrics-v1':
        raise ValueError('Explicit metric protocol required')
    if protocol.get('tie_break') != 'descending-score-ascending-document-id':
        raise ValueError('Explicit deterministic tie policy required')
    if protocol.get('ndcg_gain') not in {'linear', 'exponential'}:
        raise ValueError('Explicit nDCG gain required')
    if protocol.get('relevant_threshold') != 0 or type(protocol['relevant_threshold']) is not int:
        raise ValueError('Binary relevance is strictly positive judgment')
    cutoffs = protocol.get('cutoffs')
    if (not isinstance(cutoffs, list) or not cutoffs or len(cutoffs) > 20
            or any(type(k) is not int or not 1 <= k <= 10000 for k in cutoffs)
            or sorted(set(cutoffs)) != cutoffs):
        raise ValueError('Unique increasing finite metric cutoffs required')
    return cutoffs


def score_full_rankings(corpus_ids, query_ids, qrels, rankings, *, protocol,
                        corpus_sha256, queries_sha256, qrels_sha256,
                        rankings_sha256, max_rows=100_000_000, check_deadline=lambda: None,
                        ranking_mode='exhaustive', ranking_depth=None):
    """Consume query-contiguous full rankings, validating complete candidate sets.

    Artifact byte authentication belongs to the caller. These SHA values bind
    the receipt to that evidence, rather than claiming to encode or search data.
    """
    cutoffs = validate_protocol(protocol)
    if ranking_mode not in {'exhaustive', 'topk_unqualified'}:
        raise ValueError('Explicit supported ranking coverage mode required')
    if ranking_mode == 'topk_unqualified' and (type(ranking_depth) is not int or not 1 <= ranking_depth <= 10000):
        raise ValueError('Top-k retention must cover every metric cutoff')
    hashes = {name: identity(value) for name, value in
              [('corpus', corpus_sha256), ('queries', queries_sha256),
               ('qrels', qrels_sha256), ('rankings', rankings_sha256)]}
    corpus, queries = set(), set()
    for collection, ids in [(corpus, corpus_ids), (queries, query_ids)]:
        for value in ids:
            check_deadline()
            if identifier(value) in collection:
                raise ValueError('Duplicate corpus/query ID')
            collection.add(value)
        if not collection:
            raise ValueError('Complete nonempty corpus and queries required')
    if ranking_mode == 'topk_unqualified' and ranking_depth < min(cutoffs[-1], len(corpus)):
        raise ValueError('Top-k retention must cover every available metric cutoff')
    depth = len(corpus) if ranking_mode == 'exhaustive' else min(ranking_depth, len(corpus))
    if type(max_rows) is not int or max_rows < 1 or depth * len(queries) > max_rows:
        raise ValueError('Full corpus-query scoring exceeds declared row bound')
    judgments = {qid: {} for qid in queries}
    for row in qrels:
        check_deadline()
        qid, doc = identifier(row['query_id']), identifier(row['document_id'])
        grade = row['relevance']
        if (qid not in queries or doc not in corpus or type(grade) is not int
                or not 0 <= grade <= 20 or doc in judgments[qid]):
            raise ValueError('Foreign, duplicate or invalid relevance judgment')
        judgments[qid][doc] = grade
    if any(not any(grade > 0 for grade in values.values()) for values in judgments.values()):
        raise ValueError('Every declared query requires a positive relevance judgment')
    per_query, seen, top, previous, current = {}, set(), [], None, None
    rows = 0

    def finish():
        if (ranking_mode == 'exhaustive' and seen != corpus) or len(seen) != depth:
            raise ValueError('Ranking does not cover the complete corpus for query ' + current)
        values = judgments[current]
        positive = sum(grade > 0 for grade in values.values())
        ideal = sorted(values.values(), reverse=True)
        gain = (lambda grade: grade) if protocol['ndcg_gain'] == 'linear' else (lambda grade: 2**grade - 1)
        result = {}
        for k in cutoffs:
            grades = [values.get(doc, 0) for doc in top[:k]]
            found = [rank for rank, grade in enumerate(grades, 1) if grade > 0]
            dcg = math.fsum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
            idcg = math.fsum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal[:k], 1))
            result.update({f'ndcg_at_{k}': dcg / idcg, f'mrr_at_{k}': 1 / found[0] if found else 0.0,
                           f'recall_at_{k}': len(found) / positive})
        per_query[current] = result

    for row in rankings:
        check_deadline()
        rows += 1
        if rows > max_rows:
            raise ValueError('Scored ranking row bound exceeded')
        qid, doc = identifier(row['query_id']), identifier(row['document_id'])
        score = row['score']
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or qid not in queries or doc not in corpus):
            raise ValueError('Foreign IDs or nonfinite ranking score')
        if qid != current:
            if current is not None:
                finish()
            if qid in per_query:
                raise ValueError('Ranking query groups must be contiguous and unique')
            current, seen, top, previous = qid, set(), [], None
        if doc in seen:
            raise ValueError('Duplicate ranked document')
        if previous is not None and (score > previous[0] or (score == previous[0] and doc <= previous[1])):
            raise ValueError('Ranking order differs from explicit score/tie protocol')
        previous = (score, doc)
        seen.add(doc)
        if len(top) < cutoffs[-1]:
            top.append(doc)
    if current is not None:
        finish()
    if set(per_query) != queries or rows != depth * len(queries):
        raise ValueError('Complete declared query/corpus scoring coverage required')
    metrics = {name: math.fsum(per_query[qid][name] / len(queries) for qid in sorted(queries))
               for name in next(iter(per_query.values()))}
    comparison_identity = canonical_hash({key: hashes[key] for key in ('corpus', 'queries', 'qrels')})
    return {'format': 'flaxchat-full-corpus-retrieval-report-v1', 'reporting_only': True,
            'selection_policy_changed': False, 'model_execution': False,
            'artifact_sha256': hashes, 'protocol': dict(protocol), 'protocol_identity_sha256': canonical_hash(protocol),
            'query_identity_sha256': comparison_identity, 'corpus_documents': len(corpus),
            'queries': len(queries), 'scored_rows': rows, 'corpus_coverage': 'every-document-once-per-query-verified' if ranking_mode == 'exhaustive' else 'top-k-only; full-search-not-verified',
            'ranking_mode': ranking_mode, 'ranking_depth': depth,
            'full_search_qualified': False,
            'score_scale': 'native-fraction-0-to-1', 'aggregation': 'query_arithmetic_mean',
            'metrics': metrics, 'per_query': {qid: per_query[qid] for qid in sorted(per_query)},
            'limitations': 'Authenticates supplied rankings and candidate coverage; does not prove encoder or search numerical correctness, corpus source completeness, contamination freedom or model quality.'}
