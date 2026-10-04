"""Literal ranking/receipt metrics only, never a CPU model or embedding score."""

import hashlib
import json
import math
import subprocess
import sys

import pytest

from flaxchat.full_corpus_retrieval import score_full_rankings
from scripts.report_full_corpus_retrieval import report

PROTOCOL = {
    "format": "flaxchat-full-corpus-metrics-v1",
    "cutoffs": [1, 2],
    "tie_break": "descending-score-ascending-document-id",
    "ndcg_gain": "linear",
    "relevant_threshold": 0,
}


def fixture():
    return dict(
        corpus_ids=["d1", "d2", "d3"],
        query_ids=["q1", "q2"],
        qrels=[
            {"query_id": "q1", "document_id": "d2", "relevance": 2},
            {"query_id": "q1", "document_id": "d3", "relevance": 1},
            {"query_id": "q2", "document_id": "d3", "relevance": 1},
        ],
        rankings=[
            {"query_id": qid, "document_id": doc, "score": score}
            for qid, docs in [("q1", ["d1", "d2", "d3"]), ("q2", ["d3", "d2", "d1"])]
            for doc, score in zip(docs, [3.0, 2.0, 1.0], strict=True)
        ],
        protocol=PROTOCOL,
        corpus_sha256="a" * 64,
        queries_sha256="b" * 64,
        qrels_sha256="c" * 64,
        rankings_sha256="d" * 64,
    )


def test_imports_no_model_or_numerical_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flaxchat.full_corpus_tpu; import scripts.report_full_corpus_retrieval; assert not {'jax','numpy','torch','flax'} & set(sys.modules)",
        ],
        check=True,
        timeout=10,
    )


def test_hand_derived_native_graded_metrics_and_complete_coverage():
    result = score_full_rankings(**fixture())
    q = result["per_query"]["q1"]
    assert q["ndcg_at_1"] == 0 and q["mrr_at_2"] == 0.5 and q["recall_at_2"] == 0.5
    assert q["ndcg_at_2"] == pytest.approx((2 / math.log2(3)) / (2 + 1 / math.log2(3)))
    assert result["metrics"]["recall_at_2"] == 0.75
    assert result["scored_rows"] == 6 and result["corpus_documents"] == 3
    assert result["corpus_coverage"] == "every-document-once-per-query-verified"
    assert result["full_search_qualified"] is False


def test_topk_scores_keep_full_qrels_denominator_but_no_search_claim():
    args = fixture()
    args["rankings"] = [
        row for index, row in enumerate(args["rankings"]) if index not in {2, 5}
    ]
    result = score_full_rankings(
        **args, ranking_mode="topk_unqualified", ranking_depth=2
    )
    assert result["per_query"]["q1"]["recall_at_2"] == 0.5
    assert result["corpus_coverage"] == "top-k-only; full-search-not-verified"
    assert result["full_search_qualified"] is False


def test_selected_positive_only_pool_is_not_complete_corpus():
    args = fixture()
    args["rankings"] = [row for row in args["rankings"] if row["document_id"] != "d1"]
    with pytest.raises(ValueError, match="complete corpus"):
        score_full_rankings(**args)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("nan", "nonfinite"),
        ("bool", "nonfinite"),
        ("foreign", "Foreign"),
        ("duplicate", "Duplicate"),
        ("unsorted", "order"),
        ("missingquery", "Complete"),
    ],
)
def test_inconsistent_ranking_evidence_rejected(mutation, match):
    args = fixture()
    if mutation == "nan":
        args["rankings"][0]["score"] = float("nan")
    elif mutation == "bool":
        args["rankings"][0]["score"] = True
    elif mutation == "foreign":
        args["rankings"][0]["document_id"] = "outside"
    elif mutation == "duplicate":
        args["rankings"][1]["document_id"] = "d1"
    elif mutation == "unsorted":
        args["rankings"][1]["score"] = 4
    else:
        args["rankings"] = args["rankings"][:3]
    with pytest.raises(ValueError, match=match):
        score_full_rankings(**args)


def test_ties_follow_explicit_document_id_policy():
    args = fixture()
    args["rankings"][0]["score"] = 2
    assert score_full_rankings(**args)["scored_rows"] == 6
    args["rankings"][0]["document_id"], args["rankings"][1]["document_id"] = "d2", "d1"
    with pytest.raises(ValueError, match="tie protocol"):
        score_full_rankings(**args)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "foreign", "negative", "no-positive"]
)
def test_invalid_qrels_never_silently_change_query_denominator(mutation):
    args = fixture()
    if mutation == "duplicate":
        args["qrels"].append(dict(args["qrels"][0]))
    elif mutation == "foreign":
        args["qrels"][0]["document_id"] = "outside"
    elif mutation == "negative":
        args["qrels"][0]["relevance"] = -1
    else:
        args["qrels"][2]["relevance"] = 0
    with pytest.raises(ValueError):
        score_full_rankings(**args)


def test_declared_work_bounds_and_cutoffs_enforced():
    with pytest.raises(ValueError, match="row bound"):
        score_full_rankings(**fixture(), max_rows=5)
    with pytest.raises(ValueError, match="cutoff"):
        score_full_rankings(
            **fixture(), ranking_mode="topk_unqualified", ranking_depth=1
        )


def artifacts(tmp_path):
    args = fixture()
    values = {
        "corpus": [{"id": doc, "text": doc} for doc in args["corpus_ids"]],
        "queries": [{"id": query, "text": query} for query in args["query_ids"]],
        "qrels": args["qrels"],
        "rankings": args["rankings"],
    }
    contract = {
        "format": "flaxchat-full-corpus-report-contract-v1",
        "purpose": "reporting_only",
        "source_split": "dev",
        "source_identity": {"local_snapshot": "fixture"},
        "model_identity": {"literal_scored_fixture": True},
        "ranking_mode": "exhaustive",
        "protocol": PROTOCOL,
        "max_rows": 20,
        "max_ids": 10,
        "bootstrap": {"resamples": 10, "seed": 0},
        "files": {},
    }
    for role, rows in values.items():
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path = tmp_path / (role + ".jsonl")
        path.write_bytes(raw)
        contract["files"][role] = {
            "path": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_artifact_authentication_and_bound_native_uncertainty(tmp_path):
    path, digest = artifacts(tmp_path)
    result = report(path, digest, tmp_path / "report.json")
    assert result["metrics"]["recall_at_2"] == 0.75
    assert result["uncertainty"]["recall_at_2"]["estimate"] == 0.75
    assert (
        result["uncertainty"]["recall_at_2"]["query_identity_sha256"]
        == result["query_identity_sha256"]
    )
    assert result["encoder_execution_verified"] is False
    with pytest.raises(FileExistsError):
        report(path, digest, tmp_path / "report.json")


def test_rehashed_or_corrupted_input_cannot_bypass_external_contract(tmp_path):
    path, digest = artifacts(tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(corpus.read_bytes().replace(b"d1", b"e1"))
    with pytest.raises(ValueError, match="SHA256"):
        report(path, digest, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()
    value = json.loads(path.read_text())
    value["files"]["corpus"]["sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Externally pinned"):
        report(path, digest, tmp_path / "report.json")


def test_symlink_and_deadline_rejected_before_report_commit(tmp_path, monkeypatch):
    path, digest = artifacts(tmp_path)
    target = tmp_path / "corpus.jsonl"
    target.rename(tmp_path / "real.jsonl")
    target.symlink_to(tmp_path / "real.jsonl")
    with pytest.raises(ValueError, match="Regular"):
        report(path, digest, tmp_path / "report.json")
    target.unlink()
    (tmp_path / "real.jsonl").rename(target)
    ticks = iter([0, 601])
    monkeypatch.setattr(
        "scripts.report_full_corpus_retrieval.time.monotonic", lambda: next(ticks)
    )
    with pytest.raises(TimeoutError):
        report(path, digest, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_cutoffs_larger_than_complete_small_corpus_are_valid():
    args = fixture()
    args['protocol'] = {**PROTOCOL, 'cutoffs': [1, 10]}
    result = score_full_rankings(**args, ranking_mode='topk_unqualified', ranking_depth=3)
    assert result['metrics']['recall_at_10'] == 1
    assert result['ranking_depth'] == 3
    assert result['full_search_qualified'] is False
