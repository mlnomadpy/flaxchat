#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/yat-data"
INPUT="$HOME/yat-input"
BUCKET="gs://azettaai-yat-eval-0929/embedding-data"
PYTHON="$HOME/yat-data-env/bin/python"
mkdir -p "$ROOT"

sync_log() {
  gcloud storage cp "$HOME/yat-data-prep.log" "$BUCKET/preparation.log" || true
}
trap sync_log EXIT

for source in msmarco miracl code; do
  if [[ -e "$ROOT/$source" ]]; then
    echo "Refusing to overwrite existing prepared source: $source" >&2
    exit 1
  fi
  echo "Starting $source at $(date -u +%FT%TZ)"
  "$PYTHON" "$INPUT/prepare_yat_embedding_finetune.py" \
    --source "$source" \
    --output "$ROOT/$source" \
    --tokenizer "$INPUT/tokenizer.json" \
    --query-length 128 \
    --document-length 256
  gcloud storage rsync "$ROOT/$source" "$BUCKET/$source" --recursive
  sync_log
  echo "Completed $source at $(date -u +%FT%TZ)"
done

echo 'All three prepared sources uploaded.'
