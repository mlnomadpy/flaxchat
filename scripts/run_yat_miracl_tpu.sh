#!/usr/bin/env bash
set -euo pipefail

# Eight disjoint language-subset evaluators on a v5e-8 single-host TPU.
cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
PYTHON="$HOME/yat-eval-env/bin/python"
REPORT_DIR="$HOME/yat-miracl-reports"
REPORT_PREFIX=gs://azettaai-yat-eval-0929/miracl-subsets
MODEL_DIR="$HOME/yat-public-step12000"
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
    sync_reports || echo 'Periodic MIRACL report sync failed' >&2
  done
) &
SYNC_PID=$!

for chip in {0..7}; do
  file="$HOME/yat-miracl-shards/miracl-chip-$chip.json"
  if [[ ! -s "$file" ]]; then
    echo "Missing MIRACL subset manifest for chip $chip" >&2
    exit 2
  fi
  TPU_VISIBLE_CHIPS="$chip" \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  "$PYTHON" -m scripts.evaluate_yat_mteb_subsets \
    --model-dir "$MODEL_DIR" --task MIRACLRetrievalHardNegatives \
    --subsets-file "$file" --batch-size 16 \
    --output "$REPORT_DIR/chip-$chip.json" \
    > "$HOME/yat-miracl-chip-$chip.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
sync_reports
exit "$status"
