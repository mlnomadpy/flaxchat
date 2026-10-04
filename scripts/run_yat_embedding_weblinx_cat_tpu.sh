#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
export TPU_VISIBLE_CHIPS=6
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_PROCESS_BOUNDS=1,1,1
export HF_HOME=/dev/shm/yat-embedding-full-mteb-hf
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
"$HOME/yat-eval-env/bin/python" -m scripts.evaluate_yat_mteb_splits \
  --model-dir "$HOME/yat-embedding-v1" \
  --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
  --task WebLINXCandidatesReranking --split test_cat \
  --batch-size 32 --output "$HOME/yat-full-mteb-weblinx/weblinx-test_cat.json"
