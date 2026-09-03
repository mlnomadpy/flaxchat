#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID explicitly}"

FLAXCHAT_ZONE="${FLAXCHAT_ZONE:-us-west4-a}"
FLAXCHAT_ACCELERATOR="${FLAXCHAT_ACCELERATOR:-v5litepod-8}"
FLAXCHAT_RUNTIME="${FLAXCHAT_RUNTIME:-v2-alpha-tpuv5-lite}"
FLAXCHAT_QUEUE="${FLAXCHAT_QUEUE:-flaxchat-flex}"
FLAXCHAT_NODE="${FLAXCHAT_NODE:-flaxchat-tpu}"
FLAXCHAT_MAX_RUN="${FLAXCHAT_MAX_RUN:-4h}"
FLAXCHAT_VALID_UNTIL="${FLAXCHAT_VALID_UNTIL:-4h}"

case "${1:-}" in
  create)
    gcloud alpha compute tpus queued-resources create "$FLAXCHAT_QUEUE" \
      --project="$PROJECT_ID" --zone="$FLAXCHAT_ZONE" \
      --node-id="$FLAXCHAT_NODE" --accelerator-type="$FLAXCHAT_ACCELERATOR" \
      --runtime-version="$FLAXCHAT_RUNTIME" --provisioning-model=flex-start \
      --max-run-duration="$FLAXCHAT_MAX_RUN" \
      --valid-until-duration="$FLAXCHAT_VALID_UNTIL" \
      --labels=project=flaxchat
    ;;
  status)
    gcloud alpha compute tpus queued-resources describe "$FLAXCHAT_QUEUE" \
      --project="$PROJECT_ID" --zone="$FLAXCHAT_ZONE"
    ;;
  delete)
    gcloud alpha compute tpus queued-resources delete "$FLAXCHAT_QUEUE" \
      --project="$PROJECT_ID" --zone="$FLAXCHAT_ZONE" --force --quiet
    ;;
  *)
    echo "usage: PROJECT_ID=... $0 {create|status|delete}" >&2
    exit 2
    ;;
esac
