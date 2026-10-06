"""Physical TPU admission for spherical exp-YAT InfoNCE; never a CPU benchmark."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLAXCHAT_PHYSICAL_TPU") != "1",
    reason="Requires explicit physical TPU admission",
)


@pytest.fixture
def runtime():
    import jax
    import jax.numpy as jnp

    assert jax.default_backend() == "tpu"
    return jax, jnp


def reference_scores(jnp, q, d, alpha, temperature):
    # Independent full-distance form; no spherical denominator shortcut.
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    dot = q @ d.T
    distance = jnp.sum((q[:, None, :] - d[None, :, :]) ** 2, axis=-1)
    return alpha * (1 + dot) ** 2 / (distance + 0.01) / temperature


@pytest.mark.parametrize("triplets", [False, True])
def test_yat_loss_gradients_and_masks(runtime, triplets):
    jax, jnp = runtime
    from flaxchat.contrastive import hard_negative_infonce, symmetric_infonce

    q = jnp.asarray([[1., 2., 0., -1.], [2., -1., 1., 0.], [0., 1., 3., 1.]])
    p = jnp.asarray([[2., 1., 0., 1.], [0., 1., -2., 1.], [1., 0., 1., 2.]])
    n = jnp.asarray([[0., 2., 1., 1.], [3., 0., 1., -1.], [1., 2., 1., 0.]])
    ids = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    pids = jnp.asarray([11, 12, 13], dtype=jnp.int32)
    groups = jnp.asarray([1, 1, 3], dtype=jnp.int32)
    nids = jnp.asarray([12, 22, 23], dtype=jnp.int32)
    valid = jnp.asarray([True, True, False])
    # Known positive 12 masks negative slot zero for every query. For query
    # zero it also removes positive candidate one. Groups mask (0,1)/(1,0).
    known = jnp.asarray([[11, 12], [12, 0], [13, 12]], dtype=jnp.int32)
    allowed_p = jnp.asarray([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=bool)
    if triplets:
        allowed_p = allowed_p.at[2, 1].set(False)
    allowed_n = jnp.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool)

    def actual(q, p, n, alpha):
        kwargs = dict(similarity="yat", yat_alpha=alpha, temperature=0.7,
                      expected_global_batch=3)
        if triplets:
            return hard_negative_infonce(q, p, n, ids, pids, nids, groups, valid,
                                         known_positive_text_ids=known, **kwargs)
        return symmetric_infonce(q, p, ids, pids, groups, **kwargs)

    def reference(q, p, n, alpha):
        positive = reference_scores(jnp, q, p, alpha, 0.7)
        scores = jnp.where(allowed_p, positive, -jnp.inf)
        reverse = -jax.nn.log_softmax(scores, axis=0)[jnp.arange(3), jnp.arange(3)].mean()
        if triplets:
            negative = reference_scores(jnp, q, n, alpha, 0.7)
            scores = jnp.concatenate([scores, jnp.where(allowed_n, negative, -jnp.inf)], axis=1)
        forward = -jax.nn.log_softmax(scores, axis=1)[jnp.arange(3), jnp.arange(3)].mean()
        return (forward + reverse) / 2

    alpha = jnp.asarray(0.02)
    got = jax.jit(jax.value_and_grad(actual, argnums=(0, 1, 2, 3)))(q, p, n, alpha)
    expected = jax.jit(jax.value_and_grad(reference, argnums=(0, 1, 2, 3)))(q, p, n, alpha)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5)
    assert np.isfinite(np.asarray(got[0]))
    assert abs(float(got[1][3])) > 1e-6


def test_yat_endpoints_roundoff_and_alpha(runtime):
    jax, jnp = runtime
    from flaxchat.contrastive import spherical_yat_logits

    score = jnp.asarray([-1.000001, -1., 0., 1., 1.000001])
    logits = jax.jit(spherical_yat_logits)(score, yat_alpha=0.02)
    expected = .02 / .05 * np.asarray([0., 0., 1 / 2.01, 400., 400.])
    np.testing.assert_allclose(logits, expected, rtol=1e-6)
    endpoint_gradients = jax.jit(jax.grad(lambda values: spherical_yat_logits(
        values, yat_alpha=.02).sum()))(score)
    assert np.isfinite(np.asarray(endpoint_gradients)).all()
    assert (np.asarray(endpoint_gradients) >= 0).all()
    derivative = jax.jit(jax.grad(lambda alpha: spherical_yat_logits(
        jnp.asarray(1.), yat_alpha=alpha).sum()))(jnp.asarray(.02))
    np.testing.assert_allclose(derivative, 400 / .05, rtol=1e-6)
    # Explicitly test positivity through the trainer's raw-alpha convention.
    raw_gradient = jax.grad(lambda raw: spherical_yat_logits(
        jnp.asarray(.3), yat_alpha=jax.nn.softplus(raw) + 1e-6).sum())(jnp.asarray(-4.))
    assert np.isfinite(float(raw_gradient)) and float(raw_gradient) > 0
    for alpha in (0., -1., float("nan"), float("inf")):
        assert np.isnan(np.asarray(spherical_yat_logits(jnp.asarray(0.), yat_alpha=alpha)))
    for value in (float("nan"), float("inf"), -float("inf")):
        assert np.isnan(np.asarray(spherical_yat_logits(jnp.asarray(value))))
    with pytest.raises(ValueError, match="scalar"):
        spherical_yat_logits(jnp.asarray(0.), yat_alpha=jnp.ones(2))


@pytest.mark.parametrize("bad", [0., 1e-14, float("nan"), float("inf")])
def test_yat_invalid_embeddings_fail_closed(runtime, bad):
    jax, jnp = runtime
    from flaxchat.contrastive import symmetric_infonce

    q = jnp.asarray([[bad, 0.], [0., 1.]])
    p = jnp.asarray([[1., 0.], [0., 1.]])
    ids = jnp.asarray([1, 2], dtype=jnp.int32)
    fn = jax.jit(lambda q: symmetric_infonce(q, p, ids, ids + 10, ids,
                                            similarity="yat", yat_alpha=.02))
    assert not np.isfinite(float(fn(q)))


def test_yat_padding_metrics_and_default_compatibility(runtime):
    jax, jnp = runtime
    from flaxchat.contrastive import hard_negative_infonce, symmetric_infonce

    q = jnp.asarray([[1., 2.], [-1., 1.]])
    p = jnp.asarray([[2., 1.], [1., -1.]])
    ids = jnp.asarray([1, 2], dtype=jnp.int32)
    kwargs = dict(similarity="yat", yat_alpha=.02, return_metrics=True)
    value, metrics = jax.jit(lambda: hard_negative_infonce(
        q, p, jnp.zeros_like(q), ids, ids + 10, ids + 20, ids,
        jnp.asarray([False, False]), **kwargs))()
    paired, paired_metrics = symmetric_infonce(q, p, ids, ids + 10, ids, **kwargs)
    np.testing.assert_allclose(value, paired, rtol=1e-6)
    assert int(metrics["valid_explicit_candidates"]) == 0
    assert bool(metrics["yat_logits_finite"])
    assert float(metrics["yat_logit_min"]) == float(paired_metrics["yat_logit_min"])
    assert float(metrics["yat_logit_max"]) == float(paired_metrics["yat_logit_max"])
    # Default remains exactly the pre-existing cosine objective.
    qnorm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    pnorm = p / jnp.linalg.norm(p, axis=-1, keepdims=True)
    logits = jnp.matmul(qnorm, pnorm.T, precision=jax.lax.Precision.HIGHEST) / .05
    expected = -.5 * (jnp.diag(jax.nn.log_softmax(logits, axis=0)).mean()
                      + jnp.diag(jax.nn.log_softmax(logits, axis=1)).mean())
    default = symmetric_infonce(q, p, ids, ids + 10, ids)
    np.testing.assert_array_equal(default, expected)
    np.testing.assert_array_equal(default, symmetric_infonce(q, p, ids, ids + 10, ids,
                                                             similarity="cosine"))


def test_yat_normalization_scale_invariance_and_invalid_negative(runtime):
    jax, jnp = runtime
    from flaxchat.contrastive import hard_negative_infonce, symmetric_infonce

    q = jnp.asarray([[1., 2.], [-1., 1.]])
    p = jnp.asarray([[2., 1.], [1., -1.]])
    ids = jnp.asarray([1, 2], dtype=jnp.int32)
    loss = jax.jit(lambda q, p: symmetric_infonce(
        q, p, ids, ids + 10, ids, similarity="yat", yat_alpha=.02))
    reference = loss(q, p)
    for scale in (1e-8, 1e30):
        np.testing.assert_allclose(loss(q * scale, p * scale), reference, rtol=2e-5, atol=2e-5)
    invalid = hard_negative_infonce(
        q, p, jnp.zeros_like(q), ids, ids + 10, ids + 20, ids,
        jnp.asarray([True, False]), similarity="yat", yat_alpha=.02)
    assert not np.isfinite(float(invalid))
    with pytest.raises(ValueError, match="global"):
        symmetric_infonce(q, p, ids, ids + 10, ids, expected_global_batch=8,
                          similarity="yat", yat_alpha=.02)
