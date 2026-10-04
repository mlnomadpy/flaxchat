#!/usr/bin/env bash
set -euo pipefail
index=${1:?MIRACL shard index is required}
shift
exec python3 -m scripts.representation_workflow --kind miracl --index "$index" "$@"
