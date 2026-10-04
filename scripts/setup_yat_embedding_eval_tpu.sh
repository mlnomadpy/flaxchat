#!/usr/bin/env bash
# Manifest-driven replacement; old fixed deployment recipes are intentionally retired.
set -euo pipefail
manifest=${FLAXCHAT_RUN_MANIFEST:?Set FLAXCHAT_RUN_MANIFEST to verified run.json}
run_root=${FLAXCHAT_RUN_ROOT:?Set FLAXCHAT_RUN_ROOT to the unique run directory}
exec python3 -m scripts.representation_run setup --manifest "$manifest" --root "$run_root" "$@"
