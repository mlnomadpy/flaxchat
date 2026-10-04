"""Explicitly enabled physical TPU scorer oracles; no automatic CPU fallback."""

import math
import os

import pytest

if os.environ.get("FLAXCHAT_RUN_PHYSICAL_FULL_CORPUS") != "1":
    pytest.skip(
        "Explicit physical TPU corpus scorer admission required",
        allow_module_level=True,
    )

import jax
import jax.numpy as jnp

from flaxchat.full_corpus_tpu import score_tpu_blocks

if jax.default_backend() != "tpu" or jax.process_count() != 1:
    pytest.fail(
        "Physical corpus scorer oracle requires single-host TPU; no CPU fallback",
        pytrace=False,
    )


def score(q, blocks, expected=("a", "b", "c", "d", "e")):
    return score_tpu_blocks(
        q,
        ["q1", "q2"],
        blocks,
        expected_document_ids=expected,
        corpus_sha256="a" * 64,
        queries_sha256="b" * 64,
        encoder_identity_sha256="c" * 64,
        top_k=3,
        query_block_size=1,
        max_document_block=5,
        timeout_seconds=120,
    )


def test_ties_multiple_query_blocks_and_last_partial_document_block():
    q = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    documents = jnp.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32
    )
    blocks = [
        (["a", "b"], documents[:2]),
        (["c", "d"], documents[2:4]),
        (["e"], documents[4:]),
    ]
    rankings, receipt = score(q, blocks)
    assert [row["document_id"] for row in rankings] == ["a", "b", "e", "c", "e", "a"]
    expected = [1.0, 1.0, 1 / math.sqrt(2), 1.0, 1 / math.sqrt(2), 0.0]
    assert [row["score"] for row in rankings] == pytest.approx(
        expected, abs=2e-6, rel=2e-6
    )
    full, _ = score(q, [(["a", "b", "c", "d", "e"], documents)])
    assert [(r["query_id"], r["document_id"]) for r in rankings] == [
        (r["query_id"], r["document_id"]) for r in full
    ]
    assert [r["score"] for r in rankings] == pytest.approx(
        [r["score"] for r in full], abs=2e-6, rel=2e-6
    )
    assert (
        receipt["corpus_documents_scored_per_query"] == 5
        and receipt["corpus_blocks"] == 3
    )
    assert receipt["producer_numerically_qualified"] is False


@pytest.mark.parametrize(
    "bad", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], [3e38, 3e38]]
)
def test_zero_nonfinite_and_extreme_finite_query_vectors_fail(bad):
    q = jnp.asarray([bad, [0.0, 1.0]], dtype=jnp.float32)
    d = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    with pytest.raises(ValueError):
        score(q, [(["a"], d)], expected=("a",))


@pytest.mark.parametrize(
    "bad", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], [3e38, 3e38]]
)
def test_zero_nonfinite_and_extreme_finite_document_vectors_fail(bad):
    q = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    d = jnp.asarray([bad], dtype=jnp.float32)
    with pytest.raises(ValueError):
        score(q, [(["a"], d)], expected=("a",))


def test_missing_and_foreign_corpus_ids_fail_before_coverage_receipt():
    q = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    d = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    with pytest.raises(ValueError, match="Incomplete"):
        score(q, [(["a"], d)], expected=("a", "b"))
    with pytest.raises(ValueError, match="traversal"):
        score(q, [(["z"], d)], expected=("a",))
