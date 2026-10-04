#!/usr/bin/env bash
set -euo pipefail

# Eight independent one-chip JAX evaluators on a v5e-8 single-host TPU.
# Chip task files are frozen JSON arrays, one per chip.
cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
MODEL_DIR=${MODEL_DIR:-$HOME/yat-public-step12000}
SHARD_DIR=${SHARD_DIR:-$HOME/yat-shards}
REPORT_DIR=${REPORT_DIR:-$HOME/yat-parallel-reports}
REPORT_PREFIX=${REPORT_PREFIX:-gs://azettaai-yat-eval-0929/mteb-v2-parallel}
PYTHON="$HOME/yat-eval-env/bin/python"
mkdir -p "$REPORT_DIR"
PIDS=()

sync_reports() {
  gcloud storage rsync "$REPORT_DIR" "$REPORT_PREFIX" --recursive
}
on_exit() {
  status=$?
  trap - EXIT
  for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "${SYNC_PID:-}" ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  sync_reports || status=1
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(
  while sleep 60; do
    sync_reports || echo 'Periodic GCS report sync failed' >&2
  done
) &
SYNC_PID=$!

for chip in {0..7}; do
  files=("$SHARD_DIR/chip-$chip-"*.json)
  if [[ ${#files[@]} -ne 1 || ! -f ${files[0]} ]]; then
    echo "Exactly one task file required for chip $chip" >&2
    exit 2
  fi
  file=${files[0]}
  suite=${file##*chip-$chip-}
  suite=${suite%.json}
  if [[ "$suite" != multilingual && "$suite" != coir ]]; then
    echo "Unexpected suite for chip $chip: $suite" >&2
    exit 2
  fi
  TPU_VISIBLE_CHIPS="$chip" \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  "$PYTHON" -m scripts.evaluate_yat_public_mteb \
    --model-dir "$MODEL_DIR" --benchmark "$suite" \
    --tasks-file "$file" --batch-size 16 \
    --output "$REPORT_DIR/chip-$chip-$suite.json" \
    > "$HOME/yat-chip-$chip.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
sync_reports
exit "$status"
