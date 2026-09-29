#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/flaxchat"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PJRT_DEVICE=TPU
PYLIB="$("$HOME/yat-torch-env/bin/python" -c 'import sys; print(sys.base_prefix + "/lib")')"
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "${1:-}" in
  smoke)
    "$HOME/yat-torch-env/bin/python" -c 'import torch, torch_xla; print(torch.__version__, torch_xla.__version__, torch_xla.real_devices())'
    ;;
  convert)
    "$HOME/yat-torch-env/bin/python" -m torch_port.convert_yat_encoder \
      "$HOME/yat-flax-release" "$HOME/yat-torch-release"
    ;;
  inputs)
    "$HOME/yat-torch-env/bin/python" -m scripts.validate_yat_torch_parity inputs \
      "$HOME/yat-flax-release/tokenizer.json" "$HOME/parity/inputs.npy"
    ;;
  jax)
    "$HOME/yat-jax-env/bin/python" -m scripts.validate_yat_torch_parity jax \
      "$HOME/yat-flax-release" "$HOME/parity/inputs.npy" "$HOME/parity/jax.npz"
    ;;
  torch)
    "$HOME/yat-torch-env/bin/python" -m scripts.validate_yat_torch_parity torch \
      "$HOME/yat-torch-release" "$HOME/parity/inputs.npy" "$HOME/parity/torch.npz"
    ;;
  compare)
    "$HOME/yat-torch-env/bin/python" -m scripts.validate_yat_torch_parity compare \
      "$HOME/parity/jax.npz" "$HOME/parity/torch.npz" "$HOME/parity/report.json"
    ;;
  *)
    echo 'Usage: run_yat_torch_tpu_stage.sh {smoke|convert|inputs|jax|torch|compare}' >&2
    exit 2
    ;;
esac
