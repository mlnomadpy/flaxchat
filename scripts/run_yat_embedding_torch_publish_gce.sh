#!/usr/bin/env bash
set -euo pipefail
# Token delivery remains explicit; never place credentials in manifest artifacts.
trap 'rm -f "$HOME/hf_token"' EXIT
chmod 600 "$HOME/hf_token"
python3 -m scripts.representation_workflow --kind publish-torch "$@"
