"""Score all declared bitext subsets from verified exported embedding artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from flaxchat.bitext import bitext_metrics
from flaxchat.encoder_data import file_hash
from scripts.evaluate_retrieval_embeddings import evaluate


def evaluate_subset(directory, *, score_dtype="float32"):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    identity = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("subset"), str) or not manifest["subset"]:
        raise ValueError("Explicit bitext subset required")
    queries, documents = manifest["query_ids"], manifest["document_ids"]
    if len(queries) != len(documents):
        raise ValueError("Paired bitext requires equal query and corpus counts")
    for query, document in zip(queries, documents, strict=True):
        if manifest["qrels"].get(query) != {document: 1}:
            raise ValueError("Require original one-to-one aligned bitext judgments")
    report = evaluate(
        directory, cutoffs=(1,), include_rankings=True, score_dtype=score_dtype
    )
    if report["manifest_sha256"] != identity or file_hash(manifest_path) != identity:
        raise ValueError("Manifest changed during bitext evaluation")
    indices = {document: i for i, document in enumerate(documents)}
    predictions = [indices[report["rankings"][query][0]] for query in queries]
    return dict(
        subset=manifest["subset"],
        dataset=manifest["dataset"],
        revision=manifest["revision"],
        split=manifest["split"],
        model_identity=manifest["model_identity"],
        pooling=manifest["pooling"],
        manifest_sha256=identity,
        pairs=len(queries),
        predictions=predictions,
        metrics=bitext_metrics(
            predictions, list(range(len(queries))), corpus_size=len(documents)
        ),
    )


def evaluate_campaign(root, inventory, *, score_dtype="float32"):
    root, inventory = Path(root), Path(inventory)
    identity = file_hash(inventory)
    declared = json.loads(inventory.read_text())
    entries = declared["shards"]
    subsets = [entry["subset"] for entry in entries]
    if (
        not subsets
        or len(set(subsets)) != len(subsets)
        or any(
            not isinstance(s, str) or Path(s).name != s or s in (".", "..")
            for s in subsets
        )
    ):
        raise ValueError("Unique safe subset names required")
    expected = {entry["subset"]: entry["pairs"] for entry in entries}
    reports = [
        evaluate_subset(root / subset, score_dtype=score_dtype) for subset in subsets
    ]
    for report in reports:
        if (
            report["dataset"] != declared["dataset"]
            or report["revision"] != declared["revision"]
            or report["split"] != "test"
            or report["pairs"] != expected[report["subset"]]
            or report["model_identity"] != reports[0]["model_identity"]
            or report["pooling"] != reports[0]["pooling"]
        ):
            raise ValueError(
                "Mixed model, dataset, split, pooling or incomplete subset coverage"
            )
    if file_hash(inventory) != identity:
        raise ValueError("Inventory changed during evaluation")
    result = dict(
        dataset=declared["dataset"],
        revision=declared["revision"],
        inventory_sha256=identity,
        model_identity=reports[0]["model_identity"],
        subset_count=len(reports),
        pair_count=sum(r["pairs"] for r in reports),
        per_subset=reports,
        subset_macro={
            key: float(np.mean([r["metrics"][key] for r in reports]))
            for key in reports[0]["metrics"]
        },
        similarity="cosine",
        tie_policy="ascending document ID",
        official_mteb_parity=False,
        quality_qualified=False,
        limitations=[
            "Deterministic tie policy is not upstream torch.topk tie parity.",
            "Supplied embedding provenance requires separate checkpoint execution evidence.",
            "No acceptance threshold or contamination clearance implied.",
        ],
    )
    if score_dtype != "float32":
        result["score_dtype"] = score_dtype
        source_root = Path(__file__).resolve().parents[1]
        result["scoring_source_sha256"] = {
            name: file_hash(source_root / name)
            for name in (
                "scripts/evaluate_encoder_bitext.py",
                "scripts/evaluate_retrieval_embeddings.py",
                "flaxchat/retrieval.py",
                "flaxchat/bitext.py",
            )
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--score-dtype", choices=["float32", "float64"], default="float32"
    )
    args = parser.parse_args()
    result = evaluate_campaign(
        args.embeddings, args.inventory, score_dtype=args.score_dtype
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_subset"}))


if __name__ == "__main__":
    main()
