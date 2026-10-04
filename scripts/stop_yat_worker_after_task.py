"""Stop a TPU benchmark worker once its current task has a durable receipt.

Used when redistributing later tasks from a worker occupied by a long task.
The worker's report is written atomically by evaluate_yat_public_mteb.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()

    while True:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            return
        try:
            report = json.loads(args.report.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        if args.task in report["results"] or args.task in report["failures"]:
            os.kill(args.pid, signal.SIGTERM)
            print(f"Stopped PID {args.pid} after {args.task}", flush=True)
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
