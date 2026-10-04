#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

PREFIX=gs://azettaai-yat-eval-0929/embedding-finetune-v1/full-mteb-v1
PLAN_DIR="$HOME/yat-full-mteb-plan"
REPORT_DIR="$HOME/yat-full-mteb-reports"
SPLIT_DIR="$HOME/yat-full-mteb-weblinx"
PYTHON="$HOME/yat-eval-env/bin/python"
MODEL_DIR="$HOME/yat-embedding-v1"
mkdir -p "$PLAN_DIR" "$REPORT_DIR" "$SPLIT_DIR"
gcloud storage rsync "$PREFIX/plan" "$PLAN_DIR" --recursive
gcloud storage rsync "$PREFIX/reports" "$REPORT_DIR" --recursive || true
gcloud storage rsync "$PREFIX/weblinx" "$SPLIT_DIR" --recursive || true
gcloud storage cp "$PREFIX/code/evaluate_yat_mteb_splits.py" \
  "$HOME/flaxchat/scripts/evaluate_yat_mteb_splits.py"

sync_reports() {
  gcloud storage rsync "$REPORT_DIR" "$PREFIX/reports" --recursive
  gcloud storage rsync "$SPLIT_DIR" "$PREFIX/weblinx" --recursive
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
PIDS=()

run_chip() {
  chip=$1
  export TPU_VISIBLE_CHIPS="$chip"
  export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
  export TPU_PROCESS_BOUNDS=1,1,1
  if [[ "$chip" == 0 ]]; then
    "$PYTHON" -m scripts.evaluate_yat_public_mteb \
      --model-dir "$MODEL_DIR" \
      --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
      --benchmark english --tasks-file "$PLAN_DIR/chip-0-english.json" \
      --batch-size 64 --output "$REPORT_DIR/chip-0-english.json"
  else
    for suite in english multilingual; do
      "$PYTHON" -m scripts.evaluate_yat_public_mteb \
        --model-dir "$MODEL_DIR" \
        --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
        --benchmark "$suite" --tasks-file "$PLAN_DIR/chip-$chip-$suite.json" \
        --batch-size 16 --output "$REPORT_DIR/chip-$chip-$suite.json"
    done
  fi
  splits=(test_vis test_cat test_geo test_web test_iid validation)
  if (( chip < ${#splits[@]} )); then
    split=${splits[chip]}
    "$PYTHON" -m scripts.evaluate_yat_mteb_splits \
      --model-dir "$MODEL_DIR" \
      --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
      --task WebLINXCandidatesReranking --split "$split" \
      --batch-size 32 --output "$SPLIT_DIR/weblinx-$split.json"
  fi
}
for chip in {0..7}; do
  run_chip "$chip" > "$HOME/yat-full-mteb-chip-$chip.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
sync_reports
exit "$status"
