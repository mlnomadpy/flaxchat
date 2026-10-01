"""Symmetric paired-encoder contrastive loss over one *global* data batch.

The caller must place the entire paired batch as global JAX arrays, sharded on
the leading data axis. Passing one host's local batch would silently reduce the
negative pool; ``expected_global_batch`` catches that integration mistake.
"""

import math

import jax
import jax.numpy as jnp


def symmetric_infonce(
    query_embeddings,
    document_embeddings,
    query_text_ids,
    document_text_ids,
    positive_group_ids,
    *,
    temperature: float = 0.05,
    expected_global_batch: int | None = None,
):
    """Return FP32 bidirectional InfoNCE with known false negatives excluded.

    Row ``i`` of each embedding array is an aligned positive pair. Text IDs
    identify exact normalized-text duplicates, while group IDs identify other
    known positive equivalents. Off-diagonal pairs sharing *any* of these IDs
    are removed from both denominators, never relabeled as negatives. IDs must
    be collision-free int32 values assigned by the data preparer.

    This function intentionally accepts globally shaped arrays rather than
    process-local slices; JAX's data sharding handles cross-device matmuls and
    their gradients. It does not silently stop gradients through remote rows.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    if query_embeddings.ndim != 2 or document_embeddings.shape != query_embeddings.shape:
        raise ValueError("Aligned query and document embeddings must have equal [batch, hidden] shape")
    batch, hidden = query_embeddings.shape
    if batch < 2 or hidden < 1:
        raise ValueError("Contrastive training requires at least two pairs and one feature")
    if expected_global_batch is not None and (
        type(expected_global_batch) is not int or expected_global_batch < 2
        or batch != expected_global_batch
    ):
        raise ValueError("Expected the complete global paired batch, not a process-local batch")
    for name, ids in (
        ("query_text_ids", query_text_ids),
        ("document_text_ids", document_text_ids),
        ("positive_group_ids", positive_group_ids),
    ):
        if ids.shape != (batch,) or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(f"{name} must contain one integer ID per global pair")

    queries = query_embeddings.astype(jnp.float32)
    documents = document_embeddings.astype(jnp.float32)
    queries = queries / jnp.maximum(jnp.linalg.norm(queries, axis=-1, keepdims=True), 1e-12)
    documents = documents / jnp.maximum(jnp.linalg.norm(documents, axis=-1, keepdims=True), 1e-12)
    logits = jnp.matmul(queries, documents.T, precision=jax.lax.Precision.HIGHEST) / temperature

    same_query = query_text_ids[:, None] == query_text_ids[None, :]
    same_document = document_text_ids[:, None] == document_text_ids[None, :]
    query_is_document = query_text_ids[:, None] == document_text_ids[None, :]
    same_positive_group = positive_group_ids[:, None] == positive_group_ids[None, :]
    known_positive = same_query | same_document | query_is_document | same_positive_group
    allowed = ~known_positive | jnp.eye(batch, dtype=bool)
    labels = jnp.arange(batch)
    query_to_document = jax.nn.log_softmax(jnp.where(allowed, logits, -jnp.inf), axis=1)
    document_to_query = jax.nn.log_softmax(jnp.where(allowed, logits, -jnp.inf), axis=0)
    loss_q = -query_to_document[labels, labels].mean()
    loss_d = -document_to_query[labels, labels].mean()
    return (loss_q + loss_d) * 0.5


def hard_negative_infonce(
    query_embeddings,
    positive_embeddings,
    negative_embeddings,
    query_text_ids,
    positive_text_ids,
    negative_text_ids,
    positive_group_ids,
    negative_valid,
    *,
    temperature: float = 0.05,
    expected_global_batch: int | None = None,
    known_positive_text_ids=None,
    return_metrics: bool = False,
):
    """Bidirectional in-batch loss with explicit mined negatives.

    The query-to-document denominator contains every allowed positive and
    mined negative in the global batch. The reverse denominator uses positive
    query/document pairs. Exact duplicate and known-positive texts are masked
    so a repeated passage is not silently taught as a false negative.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    if (query_embeddings.ndim != 2 or positive_embeddings.shape != query_embeddings.shape
            or negative_embeddings.shape != query_embeddings.shape):
        raise ValueError("All embedding arrays must have the same [batch, hidden] shape")
    batch, hidden = query_embeddings.shape
    if batch < 2 or hidden < 1 or (expected_global_batch is not None and
            (type(expected_global_batch) is not int or batch != expected_global_batch)):
        raise ValueError("Expected one complete global batch")
    for name, ids in (("query_text_ids", query_text_ids),
                      ("positive_text_ids", positive_text_ids),
                      ("negative_text_ids", negative_text_ids),
                      ("positive_group_ids", positive_group_ids)):
        if ids.shape != (batch,) or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(f"{name} must contain one integer ID per pair")
    if negative_valid.shape != (batch,) or negative_valid.dtype != jnp.bool_:
        raise ValueError("negative_valid must be a boolean flag per pair")

    def unit(vectors):
        vectors = vectors.astype(jnp.float32)
        return vectors / jnp.maximum(jnp.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)

    queries, positives, negatives = map(unit,
        (query_embeddings, positive_embeddings, negative_embeddings))
    positive_logits = jnp.matmul(queries, positives.T,
        precision=jax.lax.Precision.HIGHEST) / temperature
    negative_logits = jnp.matmul(queries, negatives.T,
        precision=jax.lax.Precision.HIGHEST) / temperature
    same_query = query_text_ids[:, None] == query_text_ids[None, :]
    same_positive = positive_text_ids[:, None] == positive_text_ids[None, :]
    query_is_positive = query_text_ids[:, None] == positive_text_ids[None, :]
    same_group = positive_group_ids[:, None] == positive_group_ids[None, :]
    allowed_positive = ~(same_query | same_positive | query_is_positive | same_group)
    allowed_positive |= jnp.eye(batch, dtype=bool)
    allowed_negative = (negative_valid[None, :]
                        & (query_text_ids[:, None] != negative_text_ids[None, :])
                        & (positive_text_ids[:, None] != negative_text_ids[None, :]))
    # Match against the complete preparation relevance set, including positives
    # outside this batch. Zero is reserved for padded membership entries.
    if known_positive_text_ids is None:
        related = same_query | same_group | jnp.eye(batch, dtype=bool)
        positive_is_negative = positive_text_ids[:, None] == negative_text_ids[None, :]
        known_negative = jnp.matmul(related.astype(jnp.int32), positive_is_negative.astype(jnp.int32)) > 0
        known_candidate = jnp.zeros((batch, batch), dtype=bool)
    else:
        if (known_positive_text_ids.ndim != 2 or known_positive_text_ids.shape[0] != batch
                or not jnp.issubdtype(known_positive_text_ids.dtype, jnp.integer)):
            raise ValueError("known_positive_text_ids must have [batch, membership] integer shape")
        known = known_positive_text_ids
        # Binary membership avoids a [batch,membership,batch] broadcast tensor.
        # Preparation may order zero padding either before or after IDs.
        if known.shape[1] < 1:
            raise ValueError("Known-positive membership width must be positive")
        known = jnp.sort(known, axis=1)
        def membership(candidates):
            def row_membership(row):
                index = jnp.searchsorted(row, candidates, side='left')
                value = row[jnp.minimum(index, row.shape[0] - 1)]
                return (candidates != 0) & (index < row.shape[0]) & (value == candidates)
            return jax.vmap(row_membership)(known)
        known_negative = membership(negative_text_ids)
        known_candidate = membership(positive_text_ids)
    allowed_negative &= ~known_negative
    allowed_positive &= ~known_candidate | jnp.eye(batch, dtype=bool)
    logits = jnp.concatenate((jnp.where(allowed_positive, positive_logits, -jnp.inf),
                              jnp.where(allowed_negative, negative_logits, -jnp.inf)), axis=1)
    labels = jnp.arange(batch)
    query_loss = -jax.nn.log_softmax(logits, axis=1)[labels, labels].mean()
    reverse_loss = -jax.nn.log_softmax(
        jnp.where(allowed_positive, positive_logits, -jnp.inf), axis=0
    )[labels, labels].mean()
    loss = 0.5 * (query_loss + reverse_loss)
    if return_metrics:
        return loss, {"masked_known_positive_negatives": jnp.sum(known_negative & negative_valid[None, :]),
                      "valid_explicit_candidates": jnp.sum(allowed_negative)}
    return loss
