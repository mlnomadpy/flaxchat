#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/yat-embedding-export"
mkdir -p "$ROOT/source" "$ROOT/checkpoint" "$ROOT/parent" "$ROOT/release"

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv
python3 -m venv "$ROOT/bootstrap"
"$ROOT/bootstrap/bin/python" -m pip install --quiet uv
"$ROOT/bootstrap/bin/uv" python install 3.12
"$ROOT/bootstrap/bin/uv" venv --clear --python 3.12 "$ROOT/env"
"$ROOT/bootstrap/bin/uv" pip install --python "$ROOT/env/bin/python" \
  'jax==0.11.2' 'flax==0.12.10' 'orbax-checkpoint==0.12.6'

gcloud storage cp gs://azettaai-yat-eval-0929/training-input/yat-embedding-src-0929.tar.gz \
  "$ROOT/source.tar.gz"
tar -xzf "$ROOT/source.tar.gz" -C "$ROOT/source"
"$ROOT/bootstrap/bin/uv" pip install --python "$ROOT/env/bin/python" \
  -e "$ROOT/source[encoder]"

gcloud storage cp --recursive \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/checkpoints/14000/model \
  "$ROOT/checkpoint/"
gcloud storage cp \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/checkpoints/14000/metadata/metadata \
  "$ROOT/checkpoint/metadata.json"
gcloud storage cp \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/checkpoints/14000/manifest/metadata \
  "$ROOT/checkpoint/manifest.json"
gcloud storage cp gs://azettaai-yat-eval-0929/model/tokenizer.json \
  "$ROOT/parent/tokenizer.json"

cd "$ROOT/source"
"$ROOT/env/bin/python" -m scripts.export_public_encoder \
  "$ROOT/checkpoint/model" "$ROOT/checkpoint/metadata.json" \
  "$ROOT/checkpoint/manifest.json" "$ROOT/parent/tokenizer.json" \
  "$ROOT/release" --model-family yat_embedding_finetune --step 14000

"$ROOT/env/bin/python" - "$ROOT/release" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
names = ("config.json", "tokenizer.json", "model.safetensors", "export.json")
hashes = {}
for name in names:
    with (root / name).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        hashes[name] = digest.hexdigest()
(root / "files.sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
print(json.dumps({"event": "export_verified", "files": hashes}), flush=True)
PY

gcloud storage cp --recursive "$ROOT/release" \
  gs://azettaai-yat-eval-0929/embedding-finetune-v1/
echo EXPORT_GCS_COMPLETE
