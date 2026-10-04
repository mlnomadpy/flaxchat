"""Bounded exact cosine top-k over physically encoded TPU corpus blocks.

This scores supplied TPU embeddings, not tokens. Physical encoder/export identity
and numerical parity must be qualified separately; no CPU fallback is provided.
"""
import hashlib
import json
from pathlib import Path
import time

from flaxchat.full_corpus_retrieval import identifier, identity


def score_tpu_blocks(query_embeddings, query_ids, document_blocks, *, expected_document_ids,
                     corpus_sha256, queries_sha256, encoder_identity_sha256,
                     top_k=100, query_block_size=64, max_document_block=8192,
                     timeout_seconds=3600):
    """Stream (document IDs, TPU embeddings) blocks; retain only query×k scores.

    expected_document_ids must be independently authenticated complete corpus
    IDs in ascending string order. Blocks traverse that exact order once. The
    resulting coverage receipt is evidence of execution, not self-qualification.
    """
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != 'tpu' or jax.process_count() != 1:
        raise ValueError('Single-host physical TPU backend required; no CPU fallback')
    for value in (corpus_sha256, queries_sha256, encoder_identity_sha256):
        identity(value)
    if (type(top_k) is not int or not 1 <= top_k <= 10000
            or type(query_block_size) is not int or not 1 <= query_block_size <= 256
            or type(max_document_block) is not int or not 1 <= max_document_block <= 65536
            or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400):
        raise ValueError('Finite physical scorer bounds required')
    qids = [identifier(qid) for qid in query_ids]
    if not qids or len(qids) > 10000 or len(qids) != len(set(qids)):
        raise ValueError('Bounded unique query IDs required')

    def physical(array):
        if not isinstance(array, jax.Array) or not array.devices() or any(device.platform != 'tpu' for device in array.devices()):
            raise ValueError('Embeddings must already reside on physical TPU devices')
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise ValueError('Floating point TPU embeddings required')
        if not bool(jax.device_get(jnp.all(jnp.isfinite(array)))):
            raise ValueError('Finite TPU embeddings required')

    physical(query_embeddings)
    if query_embeddings.ndim != 2 or query_embeddings.shape[0] != len(qids):
        raise ValueError('Complete query embedding identity/shape required')
    deadline = time.monotonic() + timeout_seconds
    normalized_queries = query_embeddings.astype(jnp.float32)
    qnorm = jnp.linalg.norm(normalized_queries, axis=1, keepdims=True)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(qnorm) & (qnorm > 0)))):
        raise ValueError('Nonfinite or zero query norm')
    normalized_queries /= qnorm
    # Retain k results on TPU throughout; host receives only final query×k.
    best_scores = jnp.full((len(qids), top_k), -jnp.inf, dtype=jnp.float32)
    best_indices = jnp.full((len(qids), top_k), -1, dtype=jnp.int32)
    expected = iter(expected_document_ids)
    docs, count, blocks, previous = [], 0, 0, None
    id_digest = hashlib.sha256()

    @jax.jit
    def merge(q, p, scores, indices, offset, valid_documents):
        values = jnp.dot(q, p.T, precision=jax.lax.Precision.HIGHEST)
        columns = jnp.arange(p.shape[0], dtype=jnp.int32)
        valid = columns < valid_documents
        values = jnp.where(valid[None, :], values, -jnp.inf)
        candidates = jnp.broadcast_to(jnp.where(valid, columns + offset, -1), values.shape)
        values = jnp.concatenate((scores, values), axis=1)
        candidates = jnp.concatenate((indices, candidates), axis=1)
        # lax.top_k selects lower array indices for tied values. Earlier corpus
        # blocks have lexically earlier IDs, and prior winners preserve that
        # order within ties, so merge preserves the declared document-ID rule.
        result, columns = jax.lax.top_k(values, top_k)
        return result, jnp.take_along_axis(candidates, columns, axis=1)

    for block_ids, embeddings in document_blocks:
        if time.monotonic() >= deadline:
            raise TimeoutError('Physical corpus scorer deadline exhausted')
        block_ids = [identifier(doc) for doc in block_ids]
        physical(embeddings)
        if (not 1 <= len(block_ids) <= max_document_block or embeddings.ndim != 2
                or embeddings.shape != (len(block_ids), query_embeddings.shape[1])
                or count + len(block_ids) >= 2**31):
            raise ValueError('Document block identity/shape/bounds differ')
        for doc in block_ids:
            canonical_doc = next(expected, None)
            if canonical_doc != doc or (previous is not None and doc <= previous):
                raise ValueError('Physical traversal differs from complete authenticated corpus order')
            previous = doc
            # Reuse the authenticated ID object; avoid duplicating millions of
            # document strings parsed again by the encoder stream.
            docs.append(canonical_doc)
            id_digest.update(json.dumps(doc, ensure_ascii=False).encode() + b'\n')
        p = embeddings.astype(jnp.float32)
        norm = jnp.linalg.norm(p, axis=1, keepdims=True)
        if not bool(jax.device_get(jnp.all(jnp.isfinite(norm) & (norm > 0)))):
            raise ValueError('Nonfinite or zero document norm')
        p /= norm
        # Fixed document compute shape avoids a separate main scorer compile
        # for every text-memory-bounded chunk or partial final block.
        p = jnp.pad(p, ((0, max_document_block - len(block_ids)), (0, 0)))
        for start in range(0, len(qids), query_block_size):
            end = min(len(qids), start + query_block_size)
            scores, indices = merge(normalized_queries[start:end], p, best_scores[start:end],
                                    best_indices[start:end], jnp.int32(count), jnp.int32(len(block_ids)))
            best_scores = best_scores.at[start:end].set(scores)
            best_indices = best_indices.at[start:end].set(indices)
        best_scores.block_until_ready()
        visible = best_scores[:, :min(top_k, count + len(block_ids))]
        if not bool(jax.device_get(jnp.all(jnp.isfinite(visible)))):
            raise ValueError('Nonfinite merged cosine score')
        count += len(block_ids)
        blocks += 1
    if not count or next(expected, None) is not None:
        raise ValueError('Incomplete physical full-corpus traversal')
    if time.monotonic() >= deadline:
        raise TimeoutError('Physical corpus scorer deadline exhausted')
    retained_indices = best_indices[:, :min(top_k, count)]
    ordered_indices = jnp.sort(retained_indices, axis=1)
    if not bool(jax.device_get(jnp.all((retained_indices >= 0) & (retained_indices < count)))):
        raise ValueError('Final retained corpus indices out of bounds')
    if ordered_indices.shape[1] > 1 and not bool(jax.device_get(jnp.all(jnp.diff(ordered_indices, axis=1) != 0))):
        raise ValueError('Duplicate final retained corpus index')
    values, indices = jax.device_get((best_scores[:, :min(top_k, count)], best_indices[:, :min(top_k, count)]))
    if not bool(jax.device_get(jnp.all(jnp.isfinite(best_scores[:, :min(top_k, count)])))):
        raise ValueError('Nonfinite final cosine score')
    rankings = [{'query_id': qid, 'document_id': docs[int(index)], 'score': float(score)}
                for qid, row_scores, row_indices in zip(qids, values, indices, strict=True)
                for score, index in zip(row_scores, row_indices, strict=True)]
    return rankings, {
        'format': 'flaxchat-physical-full-corpus-scorer-v1', 'backend': 'tpu',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'corpus_sha256': corpus_sha256, 'queries_sha256': queries_sha256,
        'encoder_identity_sha256': encoder_identity_sha256,
        'corpus_documents_scored_per_query': count, 'corpus_blocks': blocks, 'query_count': len(qids),
        'document_id_traversal_sha256': id_digest.hexdigest(), 'retained_depth': min(top_k, count),
        'document_compute_capacity': max_document_block,
        'partial_document_policy': 'zero-pad-embeddings; mask-padded-scores-to-negative-infinity',
        'scoring': 'cosine-fp32-normalization-dot-highest',
        'deadline_enforcement': 'cooperative-between-blocks; independent outer process deadline required',
        'tie_break': 'descending-score-ascending-document-id',
        'producer_numerically_qualified': False, 'encoder_execution_verified': False,
        'limitations': 'Runtime coverage evidence only; physical encoder integration and independent scorer parity tests still required.'}
