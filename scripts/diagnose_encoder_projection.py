"""Localize released-weight projection drift; diagnostic output is not qualification."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from jax.sharding import Mesh

from flaxchat.common import replicate_on_mesh
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.mlm import mask_tokens
from flaxchat.training import place_host_batch
from scripts.train_encoder import load_pretrained
from scripts.validate_encoder_projection import gradient_diagnostics


def comparison(reference, candidate):
    rows = gradient_diagnostics(reference[1], candidate[1])
    norm = sum(r['reference_l2'] ** 2 for r in rows)
    difference = sum(r['difference_l2'] ** 2 for r in rows)
    return dict(reference_loss=float(reference[0]), candidate_loss=float(candidate[0]),
                gradient_relative_l2=(difference / max(norm, 1e-30)) ** .5,
                parameters=rows)


def diagnose(output, data_root, prefix=None):
    jax.config.update('jax_default_matmul_precision', 'highest')
    if jax.default_backend() != 'tpu':
        raise RuntimeError('Physical TPU required')
    config = EncoderConfig.from_hf(json.loads(Path('artifacts/mmbert-base/config.json').read_text()),
        compute_dtype='bfloat16', residual_dtype='float32', mlm_projection='masked')
    model = ModernBert(config, rngs=nnx.Rngs(0))
    identity = load_pretrained(model, 'artifacts/mmbert-base')
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    path = Path(data_root) / 'data-512'
    tokens = np.load(path / 'tokens.npy')[:4]
    manifest = json.loads((path / 'manifest.json').read_text())
    x, y = mask_tokens(tokens, seed=42, step=0, example_ids=np.arange(4),
        vocab_size=config.vocab_size, mask_token_id=config.mask_token_id,
        special_token_ids=manifest['special_token_ids'], probability=.15)
    x, y = place_host_batch(x, mesh), place_host_batch(y, mesh)
    report: dict[str, Any] = dict(scope='diagnosis_not_qualification', device_kind=jax.devices()[0].device_kind,
                  initial_weights_sha256=identity, matmul_precision='highest', stages={})
    def save():
        Path(output).write_text(json.dumps(report, indent=2) + '\n')
        if prefix:
            subprocess.run(['gcloud', 'storage', 'cp', str(output), prefix.rstrip('/') + '/'], check=True)
        print(json.dumps(dict(completed_stages=list(report['stages']))), flush=True)
    full = nnx.jit(nnx.value_and_grad(lambda m, x, y: m(x, y)))
    base = full(model, x, y)
    bf16_results = {'xla': base}
    for backend in ('xla_full', 'pallas'):
        model.config = replace(config, mlm_loss_backend=backend)
        value = full(model, x, y)
        bf16_results[backend] = value
        report['stages']['full_bf16_' + backend] = comparison(base, value)
        save()
    # Freeze encoder outputs, then compare only prediction heads and their input gradients.
    model.config = config
    hidden = nnx.jit(lambda m, x: m.encode(x))(model, x)
    indices = jax.vmap(lambda row: jnp.nonzero(row >= 0, size=128, fill_value=0)[0])(y)
    hidden = jnp.take_along_axis(hidden, indices[..., None], axis=1)
    labels = jnp.take_along_axis(y, indices, axis=1)
    labels = jnp.where(jnp.arange(128)[None] < (y >= 0).sum(1, keepdims=True), labels, -1)
    head = nnx.jit(nnx.value_and_grad(lambda m, h, y: m._projection_loss(h, y), argnums=(0, 1)))
    base_head = head(model, hidden, labels)
    for backend in ('xla_full', 'pallas'):
        model.config = replace(config, mlm_loss_backend=backend)
        value = head(model, hidden, labels)
        report['stages']['frozen_hidden_' + backend] = comparison(base_head, value)
        save()
    # Build an actual FP32 model: replacing config alone does not change Linear.dtype.
    fp32 = ModernBert(replace(config, compute_dtype='float32'), rngs=nnx.Rngs(0))
    nnx.update(fp32, nnx.state(model))
    reference = full(fp32, x, y)
    for backend, value in bf16_results.items():
        report['stages']['fp32_vs_bf16_' + backend] = comparison(reference, value)
    save()
    fp32.config = replace(fp32.config, mlm_loss_backend='xla_full')
    report['stages']['full_fp32_xla_full'] = comparison(reference, full(fp32, x, y))
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='artifacts/encoder-validation-0922')
    parser.add_argument('--prefix')
    args = parser.parse_args()
    diagnose(args.output, args.data_root, args.prefix)
