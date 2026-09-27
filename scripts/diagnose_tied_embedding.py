"""Separate tied-embedding derivatives without changing the production model.

This diagnostic supports the native chunked XLA loss used by the captured
failure. Stop-gradient variants are instrumentation, not training algorithms.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.runtime import runtime_identity
from flaxchat.profiling import TrainingTrace
from flaxchat.training import gradients_for_local_mlm_microbatches
from scripts.numerical_evidence import read_numerical_evidence, write_numerical_evidence
from scripts.replay_mlm_accumulation import flatten, per_example_reference, restore_parameters


class StoppedInput(nnx.Embed):
    def __call__(self, inputs):
        return jax.lax.stop_gradient(super().__call__(inputs))


class StoppedDecoder(ModernBert):
    def decode(self, features):
        dtype = getattr(jnp, self.config.compute_dtype)
        return jnp.matmul(features.astype(dtype),
            jax.lax.stop_gradient(self.embedding.embedding[...]).astype(dtype).T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST) + self.decoder_bias[...]


def variants(model):
    if model.config.mlm_loss_backend != 'xla':
        raise ValueError('Tied-gradient diagnostic supports only the native xla decoder')
    for name in ('full', 'input_only', 'decoder_only'):
        cls = StoppedDecoder if name == 'input_only' else ModernBert
        current = cls(model.config, rngs=nnx.Rngs(0))
        if name == 'decoder_only':
            current.embedding = StoppedInput(model.config.vocab_size,
                model.config.hidden_size, dtype=model.embedding.dtype, rngs=nnx.Rngs(0))
        current.embedding.dtype = model.embedding.dtype
        nnx.update(current, jax.tree.map(lambda a: a.copy(), nnx.state(model, nnx.Param)))
        yield name, current


def error_summary(residual):
    finite = bool(np.isfinite(residual).all())
    return dict(finite=finite, absolute_max=float(np.max(abs(residual))) if finite else None)


def diagnose(model, inputs, targets, mesh, *, profile_directory=None):
    arrays, rows = {}, {}
    key = "['embedding']['embedding'].value"
    for name, current in variants(model):
        fn = nnx.jit(lambda m, x, y: gradients_for_local_mlm_microbatches(m, x, y, mesh))
        loss, grads = fn(current, inputs, targets)
        reference_loss, reference, count = per_example_reference(current, inputs, targets)
        actual, expected = flatten(grads)[key], reference[key]
        arrays[name + '_batched'] = actual
        arrays[name + '_per_example'] = expected
        rows[name] = dict(loss=float(loss) if np.isfinite(float(loss)) else None,
            reference_loss=reference_loss if np.isfinite(reference_loss) else None,
            masked_targets=count, **error_summary(actual - expected))
        if profile_directory is not None:
            trace = TrainingTrace(Path(profile_directory) / name, skip=1, steps=3)
            try:
                for step in range(4):
                    with trace.step(step):
                        jax.block_until_ready(fn(current, inputs, targets))
            finally:
                trace.close()
    closure = {}
    for mode in ('batched', 'per_example'):
        residual = arrays['full_' + mode] - arrays['input_only_' + mode] - arrays['decoder_only_' + mode]
        arrays['closure_' + mode] = residual
        closure[mode] = error_summary(residual)
    return arrays, dict(backend=jax.default_backend(), config=asdict(model.config),
        embedding_dtype=str(model.embedding.dtype), modes=rows, closure=closure,
        runtime=runtime_identity(), devices=[str(d) for d in jax.devices()],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [*Path('flaxchat').glob('*.py'), Path(__file__),
                                 Path('scripts/replay_mlm_accumulation.py')]},
        scope='Instrumented derivatives; check unchanged losses and decomposition closure before attribution. CPU results do not qualify TPU arithmetic.',
        production_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-directory', type=Path)
    parser.add_argument('--embedding-order', choices=('gather_first', 'cast_first'),
                        default='gather_first')
    args = parser.parse_args()
    arrays, receipt = read_numerical_evidence(args.capture)
    metadata = receipt['metadata']
    if len(metadata['devices']) != jax.device_count():
        raise ValueError('Diagnostic requires the captured device count')
    model = ModernBert(EncoderConfig(**metadata['config']), rngs=nnx.Rngs(0))
    if args.embedding_order == 'cast_first':
        model.embedding.dtype = getattr(jnp, model.config.residual_dtype)
    restore_parameters(model, arrays, metadata['leaf_paths'])
    values, report = diagnose(model, jnp.asarray(arrays['inputs']),
                             jnp.asarray(arrays['targets']), Mesh(np.asarray(jax.devices()), ('data',)),
                             profile_directory=args.profile_directory)
    report['capture_sha256'] = receipt['archive_sha256']
    report['embedding_order'] = args.embedding_order
    write_numerical_evidence(args.output, values, metadata=report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
