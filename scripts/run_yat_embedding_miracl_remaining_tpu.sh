#!/usr/bin/env bash
set -euo pipefail

gcloud storage cp \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/eval-code/run_yat_embedding_miracl_shard_tpu.sh \
  "$HOME/run_yat_embedding_miracl_shard_tpu.sh"
PIDS=()
for mapping in '0 3' '1 4' '5 6'; do
  read -r shard device <<< "$mapping"
  bash "$HOME/run_yat_embedding_miracl_shard_tpu.sh" "$shard" "$device" \
    > "$HOME/yat-embedding-miracl-chip-$shard.log" 2>&1 &
  PIDS+=("$!")
done
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
exit "$status"
