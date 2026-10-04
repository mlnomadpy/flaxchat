#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HOME/yat-extra-shards" "$HOME/yat-embedding-eval-reports"
gcloud storage rsync \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/eval-extra-shards \
  "$HOME/yat-extra-shards" --recursive

PIDS=()
for entry in '4 english' '5 multilingual' '6 multilingual'; do
  read -r chip suite <<< "$entry"
  file="$HOME/yat-extra-shards/chip-$chip-$suite-extra.json"
  TPU_VISIBLE_CHIPS="$chip" \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  "$HOME/yat-eval-env/bin/python" -m scripts.evaluate_yat_public_mteb \
    --model-dir "$HOME/yat-embedding-v1" \
    --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
    --benchmark "$suite" --tasks-file "$file" --batch-size 16 \
    --output "$HOME/yat-embedding-eval-reports/chip-$chip-$suite-extra.json" \
    > "$HOME/yat-embedding-chip-$chip-extra.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
gcloud storage rsync "$HOME/yat-embedding-eval-reports" \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/evaluation --recursive
exit "$status"
