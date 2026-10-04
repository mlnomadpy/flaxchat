#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/flaxchat"
report="$HOME/yat-full-mteb-reports/chip-6-multilingual.json"
while true; do
  done_count=$("$HOME/yat-eval-env/bin/python" - "$report" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(len(d['results'])+len(d['failures']))
except (OSError,ValueError,KeyError):
    print(0)
PY
)
  if (( done_count >= 17 )); then break; fi
  sleep 30
done
sleep 10
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
export TPU_VISIBLE_CHIPS=6
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_PROCESS_BOUNDS=1,1,1
export HF_HOME=/dev/shm/yat-embedding-full-mteb-hf
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
"$HOME/yat-eval-env/bin/python" -m scripts.evaluate_yat_public_mteb \
  --model-dir "$HOME/yat-embedding-v1" \
  --model-id gs://azettaai-yat-eval-0929/embedding-finetune-v1/release \
  --benchmark multilingual --task TwitterHjerneRetrieval \
  --batch-size 16 \
  --output "$HOME/yat-full-mteb-reports/chip-6-twitter-multilingual-retry.json"
