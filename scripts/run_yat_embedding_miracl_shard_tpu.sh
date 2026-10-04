#!/usr/bin/env bash
set -euo pipefail

chip=${1:?MIRACL shard index required}
if [[ ! "$chip" =~ ^[0-7]$ ]]; then
  echo "Shard index must be 0 through 7" >&2
  exit 2
fi
device=${2:-$chip}
if [[ ! "$device" =~ ^[0-7]$ ]]; then
  echo "TPU chip index must be 0 through 7" >&2
  exit 2
fi
cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HOME/yat-embedding-miracl-reports"
TPU_VISIBLE_CHIPS="$device" \
TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
TPU_PROCESS_BOUNDS=1,1,1 \
"$HOME/yat-eval-env/bin/python" -m scripts.evaluate_yat_mteb_subsets \
  --model-dir "$HOME/yat-embedding-v1" \
  --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
  --task MIRACLRetrievalHardNegatives \
  --subsets-file "$HOME/yat-miracl-shards/miracl-chip-$chip.json" \
  --batch-size 16 \
  --output "$HOME/yat-embedding-miracl-reports/chip-$chip.json"
gcloud storage cp "$HOME/yat-embedding-miracl-reports/chip-$chip.json" \
  "gs://azettaai-yat-eval-0929/embedding-finetune-v1/eval-miracl/chip-$chip.json"
