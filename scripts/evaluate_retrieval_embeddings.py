"""Score checksum-pinned query/corpus embeddings; not a model-quality certificate."""

import argparse
import json
from pathlib import Path

import numpy as np
from flaxchat.encoder_data import file_hash
from flaxchat.retrieval import rank_cosine, retrieval_metrics


def evaluate(
    directory,
    *,
    cutoffs=(1, 10),
    query_block=32,
    document_block=4096,
    include_rankings=False,
    score_dtype="float32",
):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    identity = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != "flaxchat-retrieval-embeddings-v1"
        or manifest.get("split") not in ("validation", "test")
        or not all(
            isinstance(manifest.get(key), str) and manifest[key]
            for key in ("dataset", "revision", "model_identity", "pooling")
        )
    ):
        raise ValueError(
            "Pinned held-out dataset and embedding model identity required"
        )
    paths = [directory / "queries.npy", directory / "corpus.npy"]
    hashes = [file_hash(path) for path in paths]
    if hashes != [manifest.get("queries_sha256"), manifest.get("corpus_sha256")]:
        raise ValueError("Embedding checksum mismatch")
    queries, corpus = [
        np.load(path, mmap_mode="r", allow_pickle=False) for path in paths
    ]
    if not cutoffs or any(type(k) is not int or k < 1 for k in cutoffs):
        raise ValueError("Positive cutoffs required")
    ranks = rank_cosine(
        queries,
        corpus,
        query_ids=manifest["query_ids"],
        document_ids=manifest["document_ids"],
        k=max(cutoffs),
        query_block=query_block,
        document_block=document_block,
        score_dtype=score_dtype,
    )
    metrics = retrieval_metrics(
        ranks,
        manifest["qrels"],
        document_ids=manifest["document_ids"],
        languages=manifest["languages"],
        cutoffs=cutoffs,
    )
    if file_hash(manifest_path) != identity or hashes != [
        file_hash(path) for path in paths
    ]:
        raise ValueError("Inputs changed during evaluation")
    result = dict(
        metrics=metrics,
        manifest_sha256=identity,
        model_identity=manifest["model_identity"],
        dataset=manifest["dataset"],
        revision=manifest["revision"],
        split=manifest["split"],
        corpus_documents=len(corpus),
        similarity="cosine",
        tie_policy="ascending document ID",
        quality_qualified=False,
        limitations=[
            "Embedding model identity is supplied provenance, not verified checkpoint execution.",
            "No benchmark acceptance threshold, confidence interval or contamination clearance implied.",
        ],
    )
    if include_rankings:
        result["rankings"] = ranks
    if score_dtype != "float32":
        result["score_dtype"] = score_dtype
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[1, 10])
    parser.add_argument(
        "--score-dtype", choices=["float32", "float64"], default="float32"
    )
    args = parser.parse_args(argv)
    report = evaluate(
        args.data, cutoffs=tuple(args.cutoffs), score_dtype=args.score_dtype
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}))


if __name__ == "__main__":
    main()
