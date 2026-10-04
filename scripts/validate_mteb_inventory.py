"""Inspect actual pinned MTEB task metadata without data loading or model work."""

from __future__ import annotations
import argparse
import inspect
from pathlib import Path
from scripts.evaluation_contract import digest, task_inventory, write_atomic


def inspect_inventory(task_names):
    import mteb

    if mteb.__version__ != "2.21.8":
        raise ValueError("Requires pinned MTEB2.21.8")
    if not task_names or len(task_names) != len(set(task_names)):
        raise ValueError("Unique nonempty task selection required")
    tasks = mteb.get_tasks(tasks=task_names)
    if {task.metadata.name for task in tasks} != set(task_names):
        raise ValueError("Unknown task or incomplete registry selection")
    inventory = {task.metadata.name: task_inventory(task) for task in tasks}
    return {
        "scope": "actual_pinned_package_task_metadata_only",
        "physical_tpu_qualified": False,
        "datasets_loaded": False,
        "model_executed": False,
        "mteb_version": mteb.__version__,
        "inventory": inventory,
        "task_source_sha256": {
            task.metadata.name: digest(Path(inspect.getfile(type(task))))
            for task in tasks
        },
        "remaining_checks": [
            "actual dataset split/subset loading",
            "emitted result row agreement",
            "physical model evaluation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "STSBenchmark",
            "MIRACLRetrievalHardNegatives",
            "WebLINXCandidatesReranking",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_atomic(args.output, inspect_inventory(args.tasks))


if __name__ == "__main__":
    main()
