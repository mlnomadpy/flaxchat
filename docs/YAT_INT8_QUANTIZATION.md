# YAT encoder INT8 weight export

The candidate uses symmetric INT8 storage with one FP32 scale per embedding row
and linear output channel. This is weight-only compression: features and distance
arithmetic remain BF16, residuals and attention scores remain FP32. It does not
implement INT8 matrix multiplication and does not establish an inference speedup.

`torch_port/yat_quantized.py` gathers vocabulary rows before reconstruction and
reconstructs projection weights on demand. Each FFN reconstructs its `wi` weight
once, sharing it between projection and YAT prototype distance. Norms, trained
alpha parameters and decoder bias retain FP32 storage. Fixed bias 1 and epsilon
0.01 remain unchanged. The inference loader freezes all parameters; fine-tuning
requires the original floating-point training checkpoint.

Export a separately authenticated PyTorch parent without executing a model:

```sh
python -m scripts.quantize_yat_encoder \
  --source /PATH/TO/ORIGINAL/PYTORCH/EXPORT \
  --output /PATH/TO/NEW/INT8/EXPORT \
  --expected-sha256 d5da8ef6bc834d286c2946b1eebee4c9060fe3924bd444a95f359652189fd421
```

The command refuses an existing output directory and never modifies the parent.
Its receipt binds weight/config/tokenizer bytes and the shipped loader sources;
`physical_tpu_validated=false` explicitly records that exporting bytes is not a
quality qualification. INT8 values use nearest-even rounding and [-127,127]; a
zero row has scale 1. Nonfinite input weights are rejected.

Repository usage (on an admitted physical TPU with a separate Torch/XLA runtime):

```python
import torch_xla
from torch_port.yat_quantized import load_quantized

torch_xla.runtime.set_device_type("TPU")
assert any("TPU" in kind for kind in torch_xla.real_devices())
model = load_quantized("/PATH/TO/INT8/EXPORT", device=torch_xla.device())
# Tokenize with the preserved tokenizer, then model.pool(ids).
```

The original parent is `mlnomad/yat-mmbert-base-embedding-v1-pytorch` at immutable
revision `2fbed4e13e33722726a63c1290669806c019be55`. Later README updates do not
change those weight bytes. Store compressed candidates separately; do not replace
its public model weights or copy its benchmark scores onto a quantized release.

Three data-only tests cover genuine integer storage, unchanged scalar bytes,
zero-row handling, chunk invariance, rounding bounds, invalid values, parent hash
rejection and refusal to overwrite. They construct no model. Native JAX weight-perturbation numerical and retrieval checks subsequently passed
(the linked October 5 receipt records that separate scope). Direct compressed
PyTorch numerical, memory and synchronized latency checks remain unrun. Before publication,
freeze quantization-specific quality gates and run a matched original-versus-INT8
TPU comparison on independent multilingual, code, STS and retrieval development
data, padding/empty cases, batches 1/8 and supported context lengths. Preserve all
failed gates. Existing exact-conversion release gates remain unchanged; compression
alone does not qualify production use or the published parent's benchmark scores.

## October 5 export

The full pinned public PyTorch parent produced a 310,252,696-byte INT8 weight
file, compared with 1,231,162,360 bytes for the original: a 74.80% weight-file
reduction. Original model files were not modified. Creation used a 900-second
Cloud Shell data-only job; no model forward pass or paid TPU allocation ran.
Its first upload exceeded a 120-second transfer timeout. The failed terminal
receipt is retained, with owned process and scratch cleanup verified. Transfer
repair uses the retained compressed files and a separate finite lease rather than
regenerating the model. This result establishes compression only.

The repaired single-worker transfer subsequently passed full SHA256 verification
for all six GCS files, including the manifest. The compressed weight SHA256 is
`237c907448a99de44c98ec6919ec02d900226115b17c9e894aa28c9106fec4a9`.
The candidate is retained at
`gs://azettaai-yat-eval-0929/quantization-1005/int8/`. Neither a public quantized
release nor a TPU quality/speed acceptance is claimed. Original failed transfer
receipts remain preserved. The export's owned work scratch was removed.

The [October 5 TPU quality check](YAT_INT8_QUALITY_2026-10-05.md) subsequently passed for native JAX reconstruction of the actual compressed weights. STS and SciFact changed minimally for YAT. This clears that weight-perturbation scope; compressed PyTorch loader parity, HBM and speed remain unrun. The export receipt retains `physical_tpu_validated=false` because it is the original creation receipt; the separate quality receipts bind the later measurements.

### QAT master checkpoints and compressed releases

A native QAT checkpoint stores **FP32 training masters**. Its configuration retains
`weight_quantization=int8_per_channel_ste`, so native inference applies the same
weight rounding used during training. `export.json` explicitly records FP32
master storage and `compressed_artifact=false`; exporting masters does not create
an INT8 model or qualify its PyTorch loader.

The ordinary PyTorch loader and ordinary Flax-to-PyTorch converter reject these
QAT configurations rather than silently dropping fake quantization. Only the
compressed loader may explicitly accept that policy, after authenticating the
INT8 artifact and matching its source policy to `quantization.json`. Unknown
quantization policies always fail. This admission support is not a physical
loader-parity qualification: a QAT release still needs its own TPU quality,
conversion, and compressed-loader receipts before publication. Existing dated
INT8 receipts remain bound to their original source and artifact hashes.
