#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HOME/yat-miracl-shards" "$HOME/yat-embedding-miracl-reports"
gcloud storage cp \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/eval-code/evaluate_yat_mteb_subsets.py \
  "$HOME/flaxchat/scripts/evaluate_yat_mteb_subsets.py"
gcloud storage rsync \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/eval-miracl-shards \
  "$HOME/yat-miracl-shards" --recursive

PIDS=()
for chip in 3 4 6 7; do
  TPU_VISIBLE_CHIPS="$chip" \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  "$HOME/yat-eval-env/bin/python" -m scripts.evaluate_yat_mteb_subsets \
    --model-dir "$HOME/yat-embedding-v1" \
    --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
    --task MIRACLRetrievalHardNegatives \
    --subsets-file "$HOME/yat-miracl-shards/miracl-chip-$chip.json" \
    --batch-size 16 \
    --output "$HOME/yat-embedding-miracl-reports/chip-$chip.json" \
    > "$HOME/yat-embedding-miracl-chip-$chip.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
gcloud storage rsync "$HOME/yat-embedding-miracl-reports" \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/eval-miracl --recursive
exit "$status"
