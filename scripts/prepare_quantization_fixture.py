"""Freeze held-out STS/SciFact rows and a labelled multilingual/code diagnostic."""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "sts": (
        "mteb/stsbenchmark-sts",
        "b0fddb56ed78048fa8b90373c8a3cfc37b684831",
        ("test.jsonl.gz",),
    ),
    "scifact": (
        "mteb/scifact",
        "d56462d0e63a25450459c4f213e49ffdb866f7f9",
        ("corpus.jsonl", "queries.jsonl", "qrels/test.jsonl"),
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, diagnostic, diagnostic_sha):
    output, diagnostic = Path(output), Path(diagnostic)
    if digest(diagnostic) != diagnostic_sha:
        raise ValueError("Diagnostic input identity differs")
    output.mkdir(parents=True, exist_ok=False)
    sources, loaded = {}, {}
    for task, (repo, revision, names) in SOURCES.items():
        for name in names:
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(16 * 1024 * 1024 + 1)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError("Dataset file exceeds admission size")
            destination = output / (task + "-" + name.replace("/", "-"))
            destination.write_bytes(data)
            sources[task + "/" + name] = {
                "repo": repo,
                "revision": revision,
                "file": name,
                "sha256": digest(destination),
                "bytes": len(data),
            }
            text = gzip.decompress(data) if name.endswith(".gz") else data
            loaded[task + "/" + name] = [
                json.loads(line) for line in text.decode().splitlines() if line.strip()
            ]
    rows = json.loads(diagnostic.read_text())
    diagnostics = len(rows)
    sts = []
    for i, pair in enumerate(loaded["sts/test.jsonl.gz"]):
        ids = []
        for side in ("sentence1", "sentence2"):
            ident = f"sts:test:{i}:{side}"
            rows.append(
                {
                    "id": ident,
                    "text": pair[side],
                    "domain": "sts_test",
                    "group": f"sts:{i}",
                    "variant": side,
                    "language": "en",
                }
            )
            ids.append(ident)
        sts.append(
            {"id": f"sts:{i}", "a": ids[0], "b": ids[1], "score": float(pair["score"])}
        )
    corpus = loaded["scifact/corpus.jsonl"]
    corpus_ids = []
    for doc in corpus:
        ident = "scifact:doc:" + str(doc["_id"])
        rows.append(
            {
                "id": ident,
                "text": (doc.get("title", "") + " " + doc["text"]).strip(),
                "domain": "scifact_corpus",
                "group": ident,
                "variant": "document",
                "language": "en",
            }
        )
        corpus_ids.append(ident)
    judgments = {}
    for judgment in loaded["scifact/qrels/test.jsonl"]:
        qid = "scifact:query:" + str(judgment["query-id"])
        judgments.setdefault(qid, {})["scifact:doc:" + str(judgment["corpus-id"])] = (
            int(judgment["score"])
        )
    queries = {str(q["_id"]): q["text"] for q in loaded["scifact/queries.jsonl"]}
    for qid in sorted(judgments):
        rows.append(
            {
                "id": qid,
                "text": queries[qid.removeprefix("scifact:query:")],
                "domain": "scifact_test_query",
                "group": qid,
                "variant": "query",
                "language": "en",
            }
        )
    if len(corpus) != 5183 or len(judgments) != 300 or len(sts) != 1379:
        raise ValueError("Frozen full test row counts differ")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(
        set(j) - set(corpus_ids) for j in judgments.values()
    ):
        raise ValueError("Duplicate IDs or judgments outside corpus")
    fixture = output / "inputs.json"
    fixture.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    plan = {
        "schema": "yat-quantization-quality-plan-v1",
        "sources": sources,
        "fixture_sha256": digest(fixture),
        "rows": len(rows),
        "diagnostic_rows": diagnostics,
        "diagnostic_sha256": diagnostic_sha,
        "sts_pairs": sts,
        "scifact_corpus_ids": corpus_ids,
        "scifact_judgments": judgments,
        "sequence_length": 256,
        "batch_size": 8,
        "pooling": "mean nonpadding including special tokens",
        "normalization": "FP32 L2",
        "prompts": "none",
        "scope": "full STS test; full SciFact test queries and corpus at length256; generated multilingual/code diagnostic",
        "reporting_only": True,
        "publication_qualified": False,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return {
        "fixture_sha256": digest(fixture),
        "plan_sha256": digest(output / "plan.json"),
        "rows": len(rows),
        "sts_pairs": len(sts),
        "scifact_queries": len(judgments),
        "corpus": len(corpus),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--diagnostic", required=True, type=Path)
    p.add_argument("--diagnostic-sha256", required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.output, a.diagnostic, a.diagnostic_sha256)))


if __name__ == "__main__":
    main()
