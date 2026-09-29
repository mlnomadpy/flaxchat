#!/usr/bin/env bash
set -euo pipefail

cd "$HOME"
if [ ! -d flaxchat/.git ]; then
  git clone --depth 1 https://github.com/mlnomadpy/flaxchat.git
fi
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 "$HOME/yat-jax-env"
uv pip install --python "$HOME/yat-jax-env/bin/python" -e "$HOME/flaxchat[encoder,tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

uv venv --python 3.12 "$HOME/yat-torch-env"
uv pip install --python "$HOME/yat-torch-env/bin/python" \
  'torch==2.9.0' 'torch_xla[tpu]==2.9.0' safetensors tokenizers huggingface_hub \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

"$HOME/yat-jax-env/bin/python" -c 'import jax; print("JAX", jax.__version__, jax.devices())'
PYLIB="$("$HOME/yat-torch-env/bin/python" -c 'import sys; print(sys.base_prefix + "/lib")')"
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$HOME/yat-torch-env/bin/python" -c 'import torch,torch_xla; print("PyTorch",torch.__version__,"XLA",torch_xla.__version__)'
