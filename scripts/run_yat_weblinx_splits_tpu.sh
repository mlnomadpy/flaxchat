#!/usr/bin/env bash
set -euo pipefail

# Run the six official WebLINX splits on otherwise idle chips of a v5e-8 VM.
cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MODEL_DIR=${MODEL_DIR:-$HOME/yat-public-step12000}
REPORT_DIR=${REPORT_DIR:-$HOME/yat-parallel-reports}
PYTHON="$HOME/yat-eval-env/bin/python"
mkdir -p "$REPORT_DIR"

splits=(validation test_iid test_cat test_geo test_vis test_web)
chips=(0 1 2 3 5 7)
pids=()
for index in "${!splits[@]}"; do
  split=${splits[$index]}
  chip=${chips[$index]}
  TPU_VISIBLE_CHIPS="$chip" \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
    timeout --signal=TERM --kill-after=30s 9600s \
    "$PYTHON" -m scripts.evaluate_yat_mteb_splits \
      --model-dir "$MODEL_DIR" \
      --task WebLINXCandidatesReranking \
      --split "$split" \
      --batch-size 32 \
      --output "$REPORT_DIR/weblinx-$split.json" \
      > "$HOME/yat-weblinx-$split.log" 2>&1 &
  pids+=("$!")
  echo "$split chip=$chip pid=$!"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
