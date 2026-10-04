#!/usr/bin/env bash
set -euo pipefail

# Execute on the azettaai TPU VM after this checkout and the evaluator script
# have been copied there. MODEL_DIR is downloaded directly from Hugging Face.
MODEL_DIR=${MODEL_DIR:-$HOME/yat-public-step12000}
REPORT_DIR=${REPORT_DIR:-$HOME/yat-eval-reports}
REPORT_PREFIX=${REPORT_PREFIX:-gs://azettaai-yat-eval-0929/mteb-v2}
BENCHMARK=${1:-english}
shift || true
if [[ "$BENCHMARK" != english && "$BENCHMARK" != multilingual && "$BENCHMARK" != coir ]]; then
  echo 'Usage: run_yat_public_mteb_tpu.sh {english|multilingual|coir} [evaluator args]' >&2
  exit 2
fi
cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
PYTHON="$HOME/yat-eval-env/bin/python"
mkdir -p "$REPORT_DIR"

sync_reports() {
  gcloud storage rsync "$REPORT_DIR" "$REPORT_PREFIX" --recursive
}
on_exit() {
  status=$?
  trap - EXIT
  if [[ -n "${SYNC_PID:-}" ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  sync_reports || status=1
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(
  while sleep 60; do
    sync_reports || echo 'Periodic GCS sync failed' >&2
  done
) &
SYNC_PID=$!

"$PYTHON" -m scripts.evaluate_yat_public_mteb \
  --model-dir "$MODEL_DIR" --benchmark "$BENCHMARK" \
  --output "$REPORT_DIR/${BENCHMARK}.json" "$@"
sync_reports
