#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
MODEL_DIR="$HOME/yat-embedding-v1"
SHARD_DIR="$HOME/yat-shards"
REPORT_DIR="$HOME/yat-embedding-eval-reports"
REPORT_PREFIX=gs://azettaai-yat-eval-0929/embedding-finetune-v1/evaluation
PYTHON="$HOME/yat-eval-env/bin/python"
mkdir -p "$REPORT_DIR"
PIDS=()

sync_reports() {
  gcloud storage rsync "$REPORT_DIR" "$REPORT_PREFIX" --recursive
}
on_exit() {
  status=$?
  trap - EXIT
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  kill "${SYNC_PID:-}" 2>/dev/null || true
  wait "${SYNC_PID:-}" 2>/dev/null || true
  sync_reports || status=1
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(while sleep 60; do sync_reports || true; done) &
SYNC_PID=$!

for chip in {0..7}; do
  files=("$SHARD_DIR/chip-$chip-"*.json)
  if [[ ${#files[@]} -ne 1 || ! -f ${files[0]} ]]; then
    echo "Exactly one task manifest required for chip $chip" >&2
    exit 2
  fi
  file=${files[0]}
  suite=${file##*chip-$chip-}
  suite=${suite%.json}
  if [[ "$suite" != english && "$suite" != coir ]]; then
    echo "Unexpected benchmark suite: $suite" >&2
    exit 2
  fi
  TPU_VISIBLE_CHIPS="$chip" \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  "$PYTHON" -m scripts.evaluate_yat_public_mteb \
    --model-dir "$MODEL_DIR" \
    --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
    --benchmark "$suite" --tasks-file "$file" --batch-size 16 \
    --output "$REPORT_DIR/chip-$chip-$suite.json" \
    > "$HOME/yat-embedding-chip-$chip.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
sync_reports
exit "$status"
