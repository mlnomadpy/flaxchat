#!/usr/bin/env bash
set -uo pipefail

timeout --signal=TERM --kill-after=30s 4h \
  bash "$HOME/yat-input/run_yat_embedding_data_prep_gce.sh"
status=$?
echo "Data preparation exited with status $status at $(date -u +%FT%TZ)"
gcloud storage cp "$HOME/yat-data-prep.log" \
  gs://azettaai-yat-eval-0929/embedding-data/preparation.log || true
sudo shutdown -h now
exit "$status"
