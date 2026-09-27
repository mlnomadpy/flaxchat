"""Require each proposed YAT variant to pass, independent of baseline success.

This validates a sweep report's coverage and status, not its tensor evidence.
Independent numerical replay is still required before accepting hardware results.
"""

import argparse
import json
import math
from pathlib import Path


def require_candidates(report, candidates, *, backend):
    """Reject missing cases, skipped candidates and incomplete timing matrices."""
    if not candidates:
        raise ValueError("At least one required candidate is needed")
    if report.get("complete") is not True or report.get("backend") != backend:
        raise ValueError("Incomplete report or unexpected backend")
    required_saved = {(radius, layer) for radius in (None, 64) for layer in (0, 7, 14, 21)}
    for candidate in candidates:
        if candidate in report.get("failures", {}):
            raise ValueError(f"Required candidate failed: {candidate}")
        rows = [r for r in report.get("rows", []) if r.get("variant") == candidate]
        if len(rows) != 8 or {(r["radius"], r["layer"]) for r in rows} != required_saved:
            raise ValueError(f"Missing saved-model coverage: {candidate}")
        for row in rows:
            checks = row.get("checks", {})
            if (row.get("passed") is not True or
                    set(checks) != {"output", "q", "k", "v", "alpha"} or
                    any(c.get("passed") is not True for c in checks.values())):
                raise ValueError(f"Saved-model gate failed: {candidate}")
        timings = report.get("timings", [])
        if len(timings) != 2 or {r["radius"] for r in timings} != {None, 128}:
            raise ValueError("Missing global/local performance coverage")
        for row in timings:
            checks = row.get("checks", {}).get(candidate, {})
            samples = row.get("samples_seconds", {}).get(candidate, [])
            if (set(checks) != {"output", "q", "k", "v", "alpha"} or
                    any(c.get("passed") is not True for c in checks.values()) or
                    len(samples) != 20 or
                    any(not math.isfinite(x) or x <= 0 for x in samples)):
                raise ValueError(f"Missing or failed performance gate: {candidate}")
    return dict(required_candidates=list(candidates), backend=backend, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--require", action="append", required=True)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    args = parser.parse_args()
    result = require_candidates(json.loads(args.report.read_text()), args.require, backend=args.backend)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
