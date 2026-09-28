"""Loss semantics and sharded negative-pool checks for encoder contrastive work."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from flaxchat.contrastive import symmetric_infonce


def test_aligned_pairs_are_better_than_swapped_pairs():
    queries = jnp.eye(3, dtype=jnp.bfloat16)
    documents = jnp.eye(3, dtype=jnp.bfloat16)
    ids = jnp.arange(3, dtype=jnp.int32)
    aligned = symmetric_infonce(queries, documents, ids, ids, ids)
    swapped = symmetric_infonce(queries, documents[jnp.array([1, 0, 2])], ids, ids, ids)
    assert aligned.dtype == jnp.float32
    assert float(aligned) < float(swapped)


def test_duplicate_positive_is_not_treated_as_negative():
    # Rows 0 and 1 encode the same positive pair. The second copy must not
    # double the denominator of either direction's first two anchors.
    embeddings = jnp.array([[1., 0.], [1., 0.], [0., 1.]])
    distinct = jnp.arange(3, dtype=jnp.int32)
    duplicates = jnp.array([7, 7, 8], dtype=jnp.int32)
    masked = symmetric_infonce(embeddings, embeddings, duplicates, duplicates, duplicates)
    naive = symmetric_infonce(embeddings, embeddings, distinct, distinct, distinct)
    assert float(masked) < float(naive)
    assert np.isfinite(float(masked))
    all_duplicates = jnp.zeros(3, dtype=jnp.int32)
    assert float(symmetric_infonce(embeddings, embeddings, all_duplicates,
                                  all_duplicates, all_duplicates)) == pytest.approx(0, abs=1e-6)


def test_known_positive_group_excludes_false_negative_with_distinct_text():
    embeddings = jnp.array([[1., 0.], [1., 0.], [0., 1.]])
    distinct = jnp.arange(3, dtype=jnp.int32)
    groups = jnp.array([4, 4, 5], dtype=jnp.int32)
    grouped = symmetric_infonce(embeddings, embeddings, distinct, distinct, groups)
    ungrouped = symmetric_infonce(embeddings, embeddings, distinct, distinct, distinct)
    assert float(grouped) < float(ungrouped)


def test_query_equal_to_another_document_is_not_a_negative():
    embeddings = jnp.array([[1., 0.], [1., 0.], [0., 1.]])
    query_ids = jnp.array([7, 8, 9], dtype=jnp.int32)
    document_ids = jnp.array([10, 7, 11], dtype=jnp.int32)
    groups = jnp.arange(3, dtype=jnp.int32)
    masked = symmetric_infonce(embeddings, embeddings, query_ids, document_ids, groups)
    unmatched = symmetric_infonce(embeddings, embeddings, query_ids,
                                  document_ids + 100, groups)
    assert float(masked) < float(unmatched)


def test_global_batch_contract_and_finite_gradients():
    queries = jnp.array([[1., .1], [.1, 1.], [-1., .1]], dtype=jnp.bfloat16)
    documents = jnp.array([[.9, .2], [.2, .9], [-.9, .2]], dtype=jnp.bfloat16)
    ids = jnp.arange(3, dtype=jnp.int32)
    def loss(q, d):
        return symmetric_infonce(q, d, ids, ids, ids, expected_global_batch=3)
    q_grad, d_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))(queries, documents)
    assert np.isfinite(np.asarray(q_grad)).all()
    assert np.isfinite(np.asarray(d_grad)).all()
    assert np.any(np.asarray(q_grad) != 0)
    assert np.any(np.asarray(d_grad) != 0)
    with pytest.raises(ValueError, match='global paired batch'):
        symmetric_infonce(queries, documents, ids, ids, ids, expected_global_batch=6)
    with pytest.raises(ValueError, match='temperature'):
        symmetric_infonce(queries, documents, ids, ids, ids, temperature=0)


def test_sharded_global_negatives_and_gradients_match_unsharded_reference():
    if jax.device_count() < 2:
        pytest.skip('Requires at least two JAX devices; run on TPU topology')
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ('data',))
    batch = jax.device_count() * 2
    queries = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4) / 13 + 0.1
    documents = queries[:, ::-1].copy() + np.eye(batch, 4, dtype=np.float32) * .2
    ids = np.arange(batch, dtype=np.int32)
    def loss(q, d, qid, did, gid):
        return symmetric_infonce(q, d, qid, did, gid, expected_global_batch=batch)
    reference = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        jnp.asarray(queries), jnp.asarray(documents), jnp.asarray(ids),
        jnp.asarray(ids), jnp.asarray(ids))
    data_sharding = NamedSharding(mesh, P('data', None))
    id_sharding = NamedSharding(mesh, P('data'))
    sharded = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        jax.device_put(queries, data_sharding),
        jax.device_put(documents, data_sharding),
        jax.device_put(ids, id_sharding),
        jax.device_put(ids, id_sharding),
        jax.device_put(ids, id_sharding))
    np.testing.assert_allclose(np.asarray(sharded[0]), np.asarray(reference[0]), rtol=2e-4, atol=2e-4)
    for distributed, unsharded in zip(sharded[1], reference[1], strict=True):
        np.testing.assert_allclose(np.asarray(distributed), np.asarray(unsharded), rtol=2e-4, atol=2e-4)
