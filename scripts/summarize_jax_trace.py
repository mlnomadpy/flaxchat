"""Summarize TPU JAX XPlane intervals without treating nested time as additive.

Category intervals can overlap. Collective duration includes potential waiting;
this report does not infer communication bandwidth or exclusive kernel time.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def union_ns(intervals):
    """Elapsed time covered by a collection of half-open intervals."""
    total = 0
    end = float("-inf")
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("Event ends before it starts")
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def overlap_ns(left, right):
    """Intersection of two interval unions, without double-counting nesting."""
    return union_ns(left) + union_ns(right) - union_ns([*left, *right])


def compile_operation_groups(operation_groups):
    """Validate named selectors; custom groups may deliberately overlap."""
    reserved = {"modules", "all_ops", "collectives", "pallas_named_ops"}
    compiled = {}
    for name, pattern in (operation_groups or {}).items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in reserved:
            raise ValueError(f"Invalid or reserved operation group: {name}")
        if not pattern:
            raise ValueError(f"Empty operation pattern: {name}")
        compiled[name] = re.compile(pattern)
    return compiled


def summarize(profile, *, top=20, operation_groups=None, max_name_chars=500,
              module_pattern=None):
    if top < 1:
        raise ValueError("top must be positive")
    if max_name_chars < 1:
        raise ValueError("max_name_chars must be positive")
    selectors = compile_operation_groups(operation_groups)
    module_selector = re.compile(module_pattern) if module_pattern is not None else None
    devices = []
    for plane in profile.planes:
        if not plane.name.startswith("/device:TPU:"):
            continue
        groups = defaultdict(list)
        operations = defaultdict(list)
        operation_events = defaultdict(int)
        windows = sorted(
            (event.start_ns, event.end_ns)
            for line in plane.lines if line.name == "XLA Modules"
            for event in line.events
            if module_selector is not None and module_selector.search(event.name)
        )
        if any(stop < start for start, stop in windows):
            raise ValueError("Module ends before it starts")
        # Merge windows so overlapping/nested modules cannot multiply op time.
        merged = []
        for start, stop in windows:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        if module_selector is not None and not windows:
            raise ValueError(f"No matching TPU modules on {plane.name}: {module_pattern}")
        for line in plane.lines:
            for event in line.events:
                interval = (event.start_ns, event.end_ns)
                if interval[1] < interval[0]:
                    raise ValueError("Event ends before it starts")
                intervals = [interval] if module_selector is None else [
                    (max(interval[0], start), min(interval[1], stop))
                    for start, stop in merged
                    if interval[0] < stop and interval[1] > start
                ]
                if line.name == "XLA Modules":
                    if module_selector is None or module_selector.search(event.name):
                        groups["modules"].extend(intervals)
                elif line.name == "XLA Ops":
                    if not intervals:
                        continue
                    operations[event.name].extend(intervals)
                    operation_events[event.name] += 1
                    groups["all_ops"].extend(intervals)
                    for name, selector in selectors.items():
                        if selector.search(event.name):
                            groups[name].extend(intervals)
                    if re.search(
                        r"\b(all-reduce|all-gather|reduce-scatter|collective-permute)\b",
                        event.name,
                    ):
                        groups["collectives"].extend(intervals)
                    if re.search(
                        r"tpu_custom_call|mosaic|pallas", event.name, re.IGNORECASE
                    ):
                        groups["pallas_named_ops"].extend(intervals)
        rows: list[dict[str, Any]] = [
            dict(
                name=name,
                events=operation_events[name],
                union_ms=union_ns(intervals) / 1e6,
                summed_event_ms=sum(b - a for a, b in intervals) / 1e6,
            )
            for name, intervals in operations.items()
        ]
        rows.sort(key=lambda row: (-row["union_ms"], row["name"]))
        operation_count = len(rows)
        # HLO loop names can contain the entire model's parameter signature.
        # Group and sort by the full name, then bound only the presentation.
        rows = rows[:top]
        for row in rows:
            name = row["name"]
            if len(name) > max_name_chars:
                row["name_sha256"] = hashlib.sha256(name.encode()).hexdigest()
                row["name_length"] = len(name)
                row["name_truncated"] = True
                row["name"] = name[:max_name_chars]
        names = ["collectives", "pallas_named_ops", *selectors]
        devices.append(
            dict(
                device=plane.name,
                matched_module_events=len(windows) if module_selector is not None else None,
                interval_union_ms={
                    name: union_ns(groups[name]) / 1e6
                    for name in (
                        "modules",
                        "all_ops",
                        "collectives",
                        "pallas_named_ops",
                        *selectors,
                    )
                },
                operation_names=operation_count,
                category_overlap_ms={
                    left: {
                        right: overlap_ns(groups[left], groups[right]) / 1e6
                        for right in names[i + 1:]
                    }
                    for i, left in enumerate(names)
                },
                top_operations=rows,
            )
        )
    return dict(
        scope="Per-device interval unions; category_overlap_ms reports pairwise temporal intersections, not causality or exclusive execution. Categories overlap. Summed event time is not exclusive time. Pallas identification uses operation names only. Collective time may include waiting.",
        operation_group_patterns=dict(operation_groups or {}),
        module_pattern=module_pattern,
        max_operation_name_chars=max_name_chars,
        devices=devices,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Completed .xplane.pb file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--module-pattern", help="Regex selecting TPU module time windows; excludes checkpoint and other work outside those windows")
    parser.add_argument("--max-operation-name-chars", type=int, default=500,
                        help="Bound displayed HLO names; truncated names retain a SHA256 identity")
    parser.add_argument(
        "--op-group", action="append", default=[], metavar="NAME=REGEX",
        help="Add an XLA operation interval-union group; repeat for multiple groups",
    )
    args = parser.parse_args(argv)
    if args.top < 1 or args.max_operation_name_chars < 1:
        parser.error("top and max-operation-name-chars must be positive")
    operation_groups = {}
    for spec in args.op_group:
        name, separator, pattern = spec.partition("=")
        if not separator or name in operation_groups:
            parser.error("Each --op-group requires a unique NAME=REGEX")
        operation_groups[name] = pattern
    try:
        compile_operation_groups(operation_groups)
        if args.module_pattern is not None:
            re.compile(args.module_pattern)
    except (ValueError, re.error) as error:
        parser.error(str(error))
    from jax.profiler import ProfileData

    result = summarize(ProfileData.from_file(str(args.trace)), top=args.top,
                       operation_groups=operation_groups,
                       max_name_chars=args.max_operation_name_chars,
                       module_pattern=args.module_pattern)
    if not result["devices"]:
        raise ValueError("Trace contains no supported TPU device planes")
    result["trace"] = str(args.trace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
