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
