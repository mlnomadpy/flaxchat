#!/usr/bin/env bash
# Run through gcp_tpu_run --setup --directory /tmp with a remote timeout.
set -euo pipefail
if [[ ( $# != 3 && $# != 4 ) || ! "$1" =~ ^gs:// || ! "$2" =~ ^[0-9a-f]{64}$ || ! "$3" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: setup_validation.sh gs://bucket/source.tar.gz SHA256 SOURCE_COMMIT [validation|training]' >&2
  exit 2
fi
archive_uri="$1"
archive_sha="$2"
source_commit="$3"
setup_profile="${4:-validation}"
if [[ "$setup_profile" != validation && "$setup_profile" != training ]]; then
  echo "Unknown setup profile" >&2
  exit 2
fi
cd /tmp
if [[ ! -d flaxchat-validation/.git ]]; then
  git clone https://github.com/mlnomadpy/flaxchat.git flaxchat-validation
fi
cd flaxchat-validation
git checkout "$source_commit"
gcloud storage cp "$archive_uri" /tmp/flaxchat-source.tar.gz
printf '%s  %s\n' "$archive_sha" /tmp/flaxchat-source.tar.gz | sha256sum -c -
tar xzf /tmp/flaxchat-source.tar.gz
if [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
"$HOME/.local/bin/uv" venv --allow-existing --python 3.12 .venv
if [[ "$setup_profile" == training ]]; then
  "$HOME/.local/bin/uv" pip install --python .venv/bin/python -c infra/tpu/validation-lock.txt -r infra/tpu/validation-environment.txt '.[data]' pip gcsfs transformers
else
  "$HOME/.local/bin/uv" pip install --python .venv/bin/python -c infra/tpu/validation-lock.txt -r infra/tpu/validation-environment.txt '.[dev,web,data]' httpx pip gcsfs transformers
  "$HOME/.local/bin/uv" pip install --python .venv/bin/python -c infra/tpu/validation-lock.txt torch --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip freeze > /tmp/flaxchat-env.txt
test -f /tmp/flaxchat-env.txt
