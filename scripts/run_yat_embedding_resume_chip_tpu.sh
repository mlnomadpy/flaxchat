#!/usr/bin/env bash
set -euo pipefail

chip=${1:?TPU chip index required}
if [[ ! "$chip" =~ ^[0-7]$ ]]; then
  echo "TPU chip index must be 0 through 7" >&2
  exit 2
fi
cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
export TPU_VISIBLE_CHIPS="$chip"
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_PROCESS_BOUNDS=1,1,1
# The 100 GiB VM boot disk cannot hold several full MTEB dataset caches.
# Keep new disposable downloads in RAM; reports stay on disk and sync to GCS.
export HF_HOME=/dev/shm/yat-embedding-full-mteb-hf
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

PYTHON="$HOME/yat-eval-env/bin/python"
MODEL_DIR="$HOME/yat-embedding-v1"
PLAN_DIR="$HOME/yat-full-mteb-plan"
REPORT_DIR="$HOME/yat-full-mteb-reports"
SPLIT_DIR="$HOME/yat-full-mteb-weblinx"
MODEL_ID=gs://azettaai-yat-eval-0929/embedding-finetune-v1/release

if [[ "$chip" == 0 ]]; then
  "$PYTHON" -m scripts.evaluate_yat_public_mteb \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --benchmark english --tasks-file "$PLAN_DIR/chip-0-english.json" \
    --batch-size 64 --output "$REPORT_DIR/chip-0-english.json"
else
  for suite in english multilingual; do
    "$PYTHON" -m scripts.evaluate_yat_public_mteb \
      --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
      --benchmark "$suite" --tasks-file "$PLAN_DIR/chip-$chip-$suite.json" \
      --batch-size 16 --output "$REPORT_DIR/chip-$chip-$suite.json"
  done
fi

splits=(test_vis test_cat test_geo test_web test_iid validation)
if (( chip < ${#splits[@]} )); then
  split=${splits[chip]}
  "$PYTHON" -m scripts.evaluate_yat_mteb_splits \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --task WebLINXCandidatesReranking --split "$split" \
    --batch-size 32 --output "$SPLIT_DIR/weblinx-$split.json"
fi
