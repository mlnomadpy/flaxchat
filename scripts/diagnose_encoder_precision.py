"""Compare projection losses with frozen encoder features on a real TPU batch."""

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.common import replicate_on_mesh
from flaxchat.training import place_host_batch
from flaxchat.mlm import mask_tokens
from scripts.train_encoder import load_pretrained
from flaxchat.fused_cross_entropy import sharded_fused_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        raise ValueError("Diagnostic requires four physical TPU devices")
    config = EncoderConfig.from_hf(
        json.loads(Path("artifacts/mmbert-base/config.json").read_text()),
        compute_dtype="bfloat16",
        residual_dtype="float32",
        mlm_projection="masked",
    )
    model = ModernBert(config, rngs=nnx.Rngs(0))
    load_pretrained(model, "artifacts/mmbert-base")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    root = Path(a.data)
    manifest = json.loads((root / "manifest.json").read_text())
    x, y = mask_tokens(
        np.load(root / "tokens.npy")[:4],
        seed=42,
        step=0,
        example_ids=np.arange(4),
        vocab_size=config.vocab_size,
        mask_token_id=config.mask_token_id,
        special_token_ids=manifest["special_token_ids"],
        probability=0.15,
    )
    x, y = place_host_batch(x, mesh), place_host_batch(y, mesh)
    full = nnx.jit(lambda m, x, y: m(x, y))
    full_gradient = nnx.jit(nnx.value_and_grad(lambda m, x, y: m(x, y)))
    losses = {}
    gradient_losses = {}
    for backend in ("xla", "xla_full", "xla_local", "pallas"):
        model.config = replace(config, mlm_loss_backend=backend)
        losses[backend] = float(full(model, x, y))
        value, gradient = full_gradient(model, x, y)
        gradient_losses[backend] = float(value)
        del gradient
    model.config = config
    hidden = nnx.jit(lambda m, x: m.encode(x))(model, x)
    indices = jax.vmap(lambda row: jnp.nonzero(row >= 0, size=128, fill_value=0)[0])(y)
    hidden = jnp.take_along_axis(hidden, indices[..., None], axis=1)
    labels = jnp.take_along_axis(y, indices, axis=1)
    labels = jnp.where(
        jnp.arange(128)[None] < (y >= 0).sum(1, keepdims=True), labels, -1
    )
    features = nnx.jit(lambda m, h: m.prediction_features(h))(model, hidden).astype(
        jnp.bfloat16
    )
    weight = model.embedding.embedding[...].astype(jnp.bfloat16)
    bias = model.decoder_bias[...]
    frozen = {}
    for backend in ("xla_full", "xla_local", "pallas"):
        fn = jax.jit(partial(sharded_fused_loss, backend=backend))
        frozen[backend] = float(fn(features, weight, bias, labels))
    # Materialize frozen real features on one chip to isolate dot/cast lowering.
    from jax.experimental import pallas as pl
    from flaxchat.fused_cross_entropy import _logits
    selected = np.unique(np.concatenate((np.asarray(labels).ravel(), np.arange(1024))))
    selected = selected[selected >= 0][:1024]
    selected = np.pad(selected, (0, 1024 - len(selected)))
    device = jax.devices()[0]
    h = jax.device_put(np.asarray(features)[0], device)
    w = jax.device_put(np.asarray(weight)[selected], device)
    b = jax.device_put(np.asarray(bias)[selected], device)
    def kernel(hr, wr, br, out):
        out[...] = _logits(hr[...], wr[...], br[...])
    pallas_logits = jax.jit(pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((128, 1024), jnp.float32)))(h, w, b)
    reference_logits = jax.jit(lambda h, w, b: (h @ w.T).astype(jnp.float32) + b)(h, w, b)
    explicit_logits = jax.jit(_logits)(h, w, b)
    fp32_logits = jax.jit(lambda h, w, b: jnp.matmul(h, w.T, preferred_element_type=jnp.float32) + b)(h, w, b)
    arrays = {name: np.asarray(value) for name, value in
        [('reference', reference_logits), ('explicit', explicit_logits), ('pallas', pallas_logits), ('fp32', fp32_logits)]}
    np.savez_compressed(Path(a.output).with_suffix('.npz'), h=np.asarray(h), w=np.asarray(w), b=np.asarray(b),
        reference=arrays['reference'], explicit=arrays['explicit'], pallas=arrays['pallas'], fp32=arrays['fp32'])
    dot_comparison = {name: dict(max_abs=float(np.max(np.abs(value-arrays['reference']))),
        differing=int(np.count_nonzero(value != arrays['reference']))) for name, value in arrays.items()}
    result = dict(
        scope="precision_diagnostic_only",
        quality_qualified=False,
        dot_comparison=dot_comparison,
        full_losses=losses,
        value_and_grad_losses=gradient_losses,
        frozen_prediction_feature_losses=frozen,
    )
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
