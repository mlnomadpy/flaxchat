"""Compare matching frozen-encoder probes without claiming production acceptance."""

import argparse
import json
import math
from pathlib import Path


def compare(reference, candidate, *, max_macro_drop_pp=None, max_language_drop_pp=None):
    limits = (max_macro_drop_pp, max_language_drop_pp)
    if (max_macro_drop_pp is None) != (max_language_drop_pp is None) or any(
        x is not None and (not math.isfinite(x) or x < 0) for x in limits
    ):
        raise ValueError("Declare both finite nonnegative regression limits")
    if (
        reference.get("scope") != "xnli_frozen_encoder_probe"
        or candidate.get("scope") != reference["scope"]
        or not reference.get("dataset")
        or reference["dataset"] != candidate.get("dataset")
    ):
        raise ValueError("Require identical pinned datasets and probe scope")
    languages = reference["dataset"]["languages"]
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("Require distinct declared languages")
    for report in (reference, candidate):
        if set(report["scores"]) != set(languages):
            raise ValueError("Missing or unexpected language scores")
        for score in report["scores"].values():
            if (
                type(score["examples"]) is not int
                or score["examples"] <= 0
                or not math.isfinite(score["accuracy"])
                or not 0 <= score["accuracy"] <= 1
            ):
                raise ValueError("Invalid score evidence")
        expected = sum(report["scores"][x]["accuracy"] for x in languages) / len(
            languages
        )
        if not math.isclose(report["macro_accuracy"], expected, abs_tol=1e-12):
            raise ValueError("Macro accuracy disagrees with language evidence")
    deltas = {}
    for language in languages:
        left, right = reference["scores"][language], candidate["scores"][language]
        if left["examples"] != right["examples"]:
            raise ValueError("Evaluation sample count changed")
        deltas[language] = 100 * (right["accuracy"] - left["accuracy"])
    macro_delta = 100 * (candidate["macro_accuracy"] - reference["macro_accuracy"])
    gate = None
    if max_macro_drop_pp is not None and max_language_drop_pp is not None:
        gate = macro_delta >= -max_macro_drop_pp and all(
            x >= -max_language_drop_pp for x in deltas.values()
        )
    return dict(
        scope="matched_xnli_probe_comparison",
        production_quality_qualified=False,
        reference_macro_accuracy=reference["macro_accuracy"],
        candidate_macro_accuracy=candidate["macro_accuracy"],
        macro_delta_percentage_points=macro_delta,
        language_delta_percentage_points=deltas,
        regression_gate_passed=gate,
        declared_limits=dict(
            macro_drop_pp=max_macro_drop_pp, language_drop_pp=max_language_drop_pp
        ),
        limitations=[
            "Optional regression limits must be chosen before observing test results; passing them is not production acceptance or statistical significance.",
            "The frozen probe does not reproduce published fine-tuned XNLI results.",
            "Matching dataset metadata does not establish matching evaluator source; retain source bundle identities.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-macro-drop-pp", type=float)
    parser.add_argument("--max-language-drop-pp", type=float)
    args = parser.parse_args()
    report = compare(
        json.loads(Path(args.reference).read_text()),
        json.loads(Path(args.candidate).read_text()),
        max_macro_drop_pp=args.max_macro_drop_pp,
        max_language_drop_pp=args.max_language_drop_pp,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(report["regression_gate_passed"] is False)


if __name__ == "__main__":
    raise SystemExit(main())
