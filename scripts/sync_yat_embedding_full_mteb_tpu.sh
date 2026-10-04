#!/usr/bin/env bash
set -u

prefix=gs://azettaai-yat-eval-0929/embedding-finetune-v1/full-mteb-v1
while true; do
  gcloud storage rsync "$HOME/yat-full-mteb-reports" "$prefix/reports" --recursive || true
  gcloud storage rsync "$HOME/yat-full-mteb-weblinx" "$prefix/weblinx" --recursive || true
  sleep 60
done
