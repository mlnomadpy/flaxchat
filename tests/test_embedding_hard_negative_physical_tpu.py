"""Physical TPU admission: independent loss/gradient and relevance semantics.

Run FLAXCHAT_PHYSICAL_TPU=1 JAX_PLATFORMS=tpu python -m pytest ... on
an actual GCP TPU. This file never treats CPU execution as qualification.
"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('FLAXCHAT_PHYSICAL_TPU') != '1',
                                reason='Requires explicit physical TPU admission')


def reference(q, p, n, qids, pids, nids, groups, valid, known, temperature):
    vectors = [v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) for v in (q, p, n)]
    q, p, n = vectors
    pl, nl = q @ p.T / temperature, q @ n.T / temperature
    batch = len(q)
    allowed_p = np.zeros((batch, batch), dtype=bool)
    allowed_n = np.zeros((batch, batch), dtype=bool)
    for i in range(batch):
        relevant = set(known[i]) - {0}
        for j in range(batch):
            allowed_p[i, j] = i == j or (qids[i] != qids[j] and pids[i] != pids[j]
                and qids[i] != pids[j] and groups[i] != groups[j] and pids[j] not in relevant)
            allowed_n[i, j] = valid[j] and nids[j] not in {qids[i], pids[i]} | relevant
    def nll(values, target):
        peak = max(values)
        return peak + np.log(np.exp(values - peak).sum()) - target
    forward = [nll(np.concatenate([pl[i, allowed_p[i]], nl[i, allowed_n[i]]]), pl[i, i]) for i in range(batch)]
    backward = [nll(pl[allowed_p[:, i], i], pl[i, i]) for i in range(batch)]
    return .5 * (np.mean(forward) + np.mean(backward))


@pytest.mark.parametrize('explicit', [True, False])
@pytest.mark.parametrize('complete_membership', [True, False])
def test_physical_loss_gradient_complete_relevance(explicit, complete_membership):
    import jax
    import jax.numpy as jnp
    from flaxchat.contrastive import hard_negative_infonce
    assert jax.default_backend() == 'tpu', 'Admission refuses a CPU/GPU fallback'
    rng = np.random.default_rng(29)
    vectors = [rng.normal(size=(4, 3)).astype(np.float32) for _ in range(3)]
    qids = np.array([1, 1, 2, 3], np.int32)
    pids = np.array([11, 12, 13, 14], np.int32)
    nids = np.array([12, 15, 99, 14], np.int32)
    groups = np.array([1, 1, 2, 3], np.int32)
    valid = np.full(4, explicit, dtype=bool)
    # 99 is a known positive absent from every positive column in this batch.
    known = np.array([[11, 12, 99], [11, 12, 99], [13, 0, 0], [14, 0, 0]], np.int32)
    if not complete_membership:
        known[:2, -1] = 0  # Fallback has only query/group positives visible in this batch.
    args = (qids, pids, nids, groups, valid)
    def loss(q, p, n):
        return hard_negative_infonce(q, p, n, *map(jnp.asarray, args),
            known_positive_text_ids=jnp.asarray(known) if complete_membership else None, temperature=.2, expected_global_batch=4)
    value, gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(*map(jnp.asarray, vectors))
    expected = reference(*vectors, *args, known, .2)
    np.testing.assert_allclose(float(value), expected, rtol=2e-5, atol=2e-5)
    for vector_index, grad in enumerate(gradients):
        finite = np.zeros_like(vectors[vector_index])
        for coordinate in np.ndindex(finite.shape):
            plus = [v.astype(np.float64).copy() for v in vectors]
            minus = [v.astype(np.float64).copy() for v in vectors]
            plus[vector_index][coordinate] += 1e-4
            minus[vector_index][coordinate] -= 1e-4
            finite[coordinate] = (reference(*plus, *args, known, .2) - reference(*minus, *args, known, .2)) / 2e-4
        np.testing.assert_allclose(np.asarray(grad), finite, rtol=3e-4, atol=3e-5)
    _, metrics = hard_negative_infonce(*map(jnp.asarray, vectors), *map(jnp.asarray, args),
        known_positive_text_ids=jnp.asarray(known) if complete_membership else None, return_metrics=True, temperature=.2)
    assert int(metrics['masked_known_positive_negatives']) == ((5 if complete_membership else 3) if explicit else 0)
