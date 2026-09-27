import numpy as np
import pytest
from flaxchat.retrieval import rank_cosine, retrieval_metrics


@pytest.mark.parametrize("block", [1, 2, 10])
def test_chunked_cosine_matches_dense_with_stable_ties(block):
    queries = np.array([[1.0, 0], [0, 1], [-1, 1]])
    corpus = np.array([[1.0, 0], [1, 0], [0, 1], [-1, 0]])
    ids = ["z", "a", "c", "b"]
    actual = rank_cosine(
        queries,
        corpus,
        query_ids=["q1", "q2", "q3"],
        document_ids=ids,
        k=3,
        query_block=2,
        document_block=block,
    )
    scores = (queries / np.linalg.norm(queries, axis=1, keepdims=True)) @ (
        corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    ).T
    expected = {
        q: sorted(ids, key=lambda doc: (-scores[i, ids.index(doc)], doc))[:3]
        for i, q in enumerate(["q1", "q2", "q3"])
    }
    assert actual == expected


def test_graded_ndcg_and_language_macro_include_all_queries():
    result = retrieval_metrics(
        {"q1": ["b", "a", "c"], "q2": ["c", "b", "a"], "q3": ["a", "b", "c"]},
        {"q1": {"a": 2, "b": 1}, "q2": {"a": 1}, "q3": {"a": 1}},
        document_ids=["a", "b", "c"],
        languages={"q1": "ar", "q2": "ar", "q3": "en"},
        cutoffs=(1, 2),
    )
    assert result["per_query"]["q1"]["recall@1"] == 0.5
    assert result["per_query"]["q2"]["mrr@2"] == 0
    expected = (1 + 2 / np.log2(3)) / (2 + 1 / np.log2(3))
    assert result["per_query"]["q1"]["ndcg@2"] == pytest.approx(expected)
    assert result["query_mean"]["recall@1"] == 0.5
    assert result["language_macro"]["ndcg@1"] == 0.625


@pytest.mark.parametrize(
    "case",
    [
        "missing_query",
        "missing_document",
        "empty_positives",
        "duplicate_rank",
        "short_ranking",
    ],
)
def test_invalid_judgments_cannot_inflate_results(case):
    ranks = {"q": ["a", "b"]}
    rels = {"q": {"a": 1}}
    langs = {"q": "ar"}
    if case == "missing_query":
        rels["other"] = {"a": 1}
    if case == "missing_document":
        rels["q"] = {"absent": 1}
    if case == "empty_positives":
        rels["q"] = {"a": 0}
    if case == "duplicate_rank":
        ranks["q"] = ["a", "a"]
    if case == "short_ranking":
        ranks["q"] = ["a"]
    with pytest.raises(ValueError):
        retrieval_metrics(
            ranks, rels, document_ids=["a", "b"], languages=langs, cutoffs=(2,)
        )


@pytest.mark.parametrize(
    "bad",
    [np.array([[0.0, 0.0]]), np.array([[np.nan, 1.0]]), np.array([[np.inf, 1.0]])],
)
def test_invalid_embeddings_rejected(bad):
    with pytest.raises(ValueError):
        rank_cosine(bad, np.ones((2, 2)), query_ids=["q"], document_ids=["a", "b"])


def test_large_finite_embeddings_normalize_without_overflow():
    assert rank_cosine(
        np.array([[1e308, 1e308]]),
        np.array([[1e308, 1e308]]),
        query_ids=["q"],
        document_ids=["a"],
    ) == {"q": ["a"]}


def test_embedding_artifact_identity_and_no_false_quality_claim(tmp_path):
    import json
    from flaxchat.encoder_data import file_hash
    from scripts.evaluate_retrieval_embeddings import evaluate

    np.save(tmp_path / "queries.npy", np.array([[1.0, 0.0]]))
    np.save(tmp_path / "corpus.npy", np.array([[0.0, 1.0], [1.0, 0.0]]))
    manifest = dict(
        format="flaxchat-retrieval-embeddings-v1",
        split="validation",
        dataset="synthetic",
        revision="test-v1",
        model_identity="test-fixture",
        pooling="mean",
        queries_sha256=file_hash(tmp_path / "queries.npy"),
        corpus_sha256=file_hash(tmp_path / "corpus.npy"),
        query_ids=["q"],
        document_ids=["a", "b"],
        qrels={"q": {"b": 1}},
        languages={"q": "ar"},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = evaluate(tmp_path, cutoffs=(1,))
    assert report["metrics"]["query_mean"]["recall@1"] == 1
    assert report["quality_qualified"] is False
    np.save(tmp_path / "queries.npy", np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="checksum"):
        evaluate(tmp_path, cutoffs=(1,))


@pytest.mark.parametrize('query_block,document_block', [(1, 1), (32, 4096)])
def test_float64_ranking_preserves_real_near_tie(query_block, document_block):
    import math
    query = np.array([[1., 0.]], dtype=np.float32)
    low = np.nextafter(np.float32(.1), np.float32(1.))
    corpus = np.array([[1., low], [1., .1]], dtype=np.float32)
    reference = [1 / math.sqrt(math.fsum(float(v) ** 2 for v in row)) for row in corpus]
    assert 0 < reference[1] - reference[0] < 1e-8
    for order in ([0, 1], [1, 0]):
        ids = [(['a', 'z'])[i] for i in order]
        ranked = rank_cosine(query, corpus[order], query_ids=['q'], document_ids=ids,
                             k=1, query_block=query_block, document_block=document_block,
                             score_dtype='float64')
        assert ranked == {'q': ['z']}


def test_ranking_rejects_unsupported_scoring_precision():
    with pytest.raises(ValueError, match='dtype'):
        rank_cosine(np.ones((1, 2)), np.ones((1, 2)), query_ids=['q'], document_ids=['d'], score_dtype='float16')
