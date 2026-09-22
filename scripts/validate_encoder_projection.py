"""Released mmBERT loss/all-parameter-gradient gate for a projection backend."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

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


def gradient_diagnostics(reference, candidate):
    """Device reductions only; never copy full released-model gradients to host."""
    @jax.jit
    def measure(a, b):
        pairs = list(zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
        return jnp.stack([jnp.stack((jnp.sum(jnp.square(x.astype(jnp.float32))),
            jnp.sum(jnp.square(y.astype(jnp.float32))),
            jnp.sum(jnp.square(x.astype(jnp.float32)-y.astype(jnp.float32))),
            jnp.sum(x.astype(jnp.float32)*y.astype(jnp.float32)))) for x,y in pairs])
    values = np.asarray(measure(reference, candidate))
    paths = [jax.tree_util.keystr(path) for path,_ in jax.tree_util.tree_flatten_with_path(reference)[0]]
    rows = []
    for path, (a2,b2,d2,ab) in zip(paths, values, strict=True):
        rows.append(dict(parameter=path, reference_l2=float(np.sqrt(a2)),
            candidate_l2=float(np.sqrt(b2)), difference_l2=float(np.sqrt(d2)),
            relative_l2=float(np.sqrt(d2/max(a2,1e-30))),
            cosine_similarity=float(ab/max(np.sqrt(a2*b2),1e-30))))
    return sorted(rows, key=lambda r:r['difference_l2'], reverse=True)


def validate(backend, tile, data_root='artifacts/encoder-validation-0922'):
    jax.config.update('jax_default_matmul_precision', 'highest')
    if jax.default_backend() != 'tpu':
        raise RuntimeError('Physical TPU required')
    config = EncoderConfig.from_hf(json.loads(Path('artifacts/mmbert-base/config.json').read_text()),
        compute_dtype='bfloat16', residual_dtype='float32', mlm_projection='masked')
    model = ModernBert(config, rngs=nnx.Rngs(0))
    identity = load_pretrained(model, 'artifacts/mmbert-base')
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    data = Path(data_root) / 'data-512'
    tokens = np.load(data/'tokens.npy')[:4]
    manifest = json.loads((data/'manifest.json').read_text())
    x, y = mask_tokens(tokens, seed=42, step=0, example_ids=np.arange(4),
        vocab_size=config.vocab_size, mask_token_id=config.mask_token_id,
        special_token_ids=manifest['special_token_ids'], probability=.15)
    x, y = place_host_batch(x, mesh), place_host_batch(y, mesh)
    value_grad = nnx.jit(nnx.value_and_grad(lambda m, x, y:m(x, y)))
    baseline, baseline_grad = value_grad(model, x, y)
    model.config = replace(config, mlm_loss_backend=backend, mlm_vocab_tile=tile)
    candidate, candidate_grad = value_grad(model, x, y)
    @jax.jit
    def errors(a, b):
        difference = jax.tree.map(lambda x, y:jnp.sum(jnp.square(x-y)),a,b)
        norm = jax.tree.map(lambda x:jnp.sum(jnp.square(x)),a)
        finite = jax.tree.reduce(lambda a,b:a & jnp.all(jnp.isfinite(b)),b,jnp.array(True))
        return jnp.sqrt(jax.tree.reduce(jnp.add,difference)/jnp.maximum(jax.tree.reduce(jnp.add,norm),1e-30)), finite
    relative, finite = errors(baseline_grad, candidate_grad)
    loss_error = abs(float(candidate)-float(baseline))
    # Declared before measurement. BF16 allows numerical drift, never NaNs.
    passed = bool(finite) and loss_error < 1e-3 and float(relative) < .03
    return dict(passed=passed, baseline_loss=float(baseline), candidate_loss=float(candidate),
        absolute_loss_error=loss_error, gradient_relative_l2=float(relative),
        maximum_loss_error=1e-3, maximum_gradient_relative_l2=.03,
        backend=backend, tile=tile, initial_weights_sha256=identity, matmul_precision='highest',
        parameter_gradient_diagnostics=gradient_diagnostics(baseline_grad, candidate_grad),
        devices=jax.device_count(), device_kind=jax.devices()[0].device_kind,
        quality_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['pallas','xla_full'], required=True)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', default='artifacts/encoder-validation-0922')
    args=parser.parse_args()
    report=validate(args.backend,args.tile,args.data_root)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    raise SystemExit(0 if report['passed'] else 1)
