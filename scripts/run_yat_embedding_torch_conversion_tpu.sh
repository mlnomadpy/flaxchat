#!/usr/bin/env bash
# Historical fixed-bucket conversion recipe is retired. Qualification now runs
# an explicitly frozen campaign; conversion is a separate authenticated stage.
set -euo pipefail
manifest=${FLAXCHAT_RUN_MANIFEST:?Set FLAXCHAT_RUN_MANIFEST to verified run.json}
run_root=${FLAXCHAT_RUN_ROOT:?Set FLAXCHAT_RUN_ROOT to the unique run directory}
python3 - "$manifest" <<'CHECK'
import sys
from pathlib import Path
from scripts.representation_run import load_manifest
manifest = load_manifest(Path(sys.argv[1]))
workload = manifest['workload']
if len(workload) < 3 or workload[1:3] != ['-m', 'scripts.run_yat_parity_campaign']:
    raise SystemExit('Require the bounded durable parity campaign as frozen workload')
for option in ('--source', '--target', '--jax-python', '--torch-python', '--evidence',
               '--host-ram-budget-bytes', '--scratch-budget-bytes', '--evidence-budget-bytes'):
    if workload.count(option) != 1 or workload.index(option) + 1 >= len(workload):
        raise SystemExit('Missing/ambiguous campaign option: ' + option)
CHECK
python3 -m scripts.representation_run setup --manifest "$manifest" --root "$run_root"
exec python3 -m scripts.representation_run run --manifest "$manifest" --root "$run_root" "$@"
