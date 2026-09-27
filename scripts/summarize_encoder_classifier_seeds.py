"""Aggregate every frozen XNLI seed from raw matched evaluations."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from scripts.validate_encoder_classifier_pair import compare


def summarize(frozen, pairs) -> dict[str, Any]:
    seeds = frozen["seeds"]
    if (
        len(seeds) < 2
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
        or set(pairs) != set(seeds)
    ):
        raise ValueError("Require every distinct frozen seed, with no extras")
    comparisons = []
    identity = None
    for seed in seeds:
        baseline, candidate = pairs[seed]
        comparisons.append(compare(baseline, candidate, frozen, seed))
        # Within-pair matching is insufficient if an entire seed uses different code.
        metadata = baseline["training_metadata"]
        recipe = metadata["resolved_config"]
        current = dict(
            evaluator=baseline["evaluator_source_sha256"],
            runtime=baseline["runtime"],
            training_source=metadata["source_python_sha256"],
            train_data=metadata["data_manifest_identity"],
            tokenizer=metadata["tokenizer_identity"],
            dataset=metadata["dataset"],
            revision=metadata["dataset_revision"],
            recipe={k: v for k, v in recipe.items() if k != "seed"},
            candidate_origin=candidate["training_metadata"]["resolved_config"][
                "origin"
            ],
        )
        if identity is not None and current != identity:
            raise ValueError("Unmatched provenance or recipe across seeds")
        identity = current

    def stats(values):
        return dict(
            mean=statistics.mean(values),
            sample_stddev=statistics.stdev(values),
            minimum=min(values),
            maximum=max(values),
        )

    return dict(
        scope="matched_full_xnli_development_all_seeds",
        seeds=seeds,
        comparisons=comparisons,
        baseline_macro_accuracy=stats(
            [r["baseline_macro_accuracy"] for r in comparisons]
        ),
        candidate_macro_accuracy=stats(
            [r["candidate_macro_accuracy"] for r in comparisons]
        ),
        paired_macro_delta_percentage_points=stats(
            [r["macro_delta_percentage_points"] for r in comparisons]
        ),
        paired_language_delta_percentage_points={
            language: stats(
                [r["language_delta_percentage_points"][language] for r in comparisons]
            )
            for language in frozen["languages"]
        },
        production_quality_qualified=False,
        limitations=[
            "Development evaluation; no untouched test-set claim.",
            "Seed dispersion is not a confidence interval or significance test.",
            "XNLI alone does not establish production multilingual quality.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("SEED", "BASELINE", "CANDIDATE"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs = {}
    hashes = {}

    def read(path):
        raw = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    frozen = read(args.plan)
    for seed, baseline, candidate in args.pair:
        seed = int(seed)
        if seed in pairs:
            raise ValueError("Duplicate seed")
        pairs[seed] = (read(baseline), read(candidate))
    report = summarize(frozen, pairs)
    report["input_sha256"] = hashes
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
