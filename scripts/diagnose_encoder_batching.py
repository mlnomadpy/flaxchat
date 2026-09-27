"""Capture small-fixture forward batching drift with instrumentation controls.

Intermediate outputs can change compilation. Uninstrumented outputs are retained
separately; neither path proves the behavior of compiled training derivatives.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.runtime import runtime_identity
from scripts.numerical_evidence import read_numerical_evidence, write_numerical_evidence
from scripts.replay_mlm_accumulation import restore_parameters


def stages(model, ids):
    segments = jnp.where(ids == model.config.pad_token_id, -1, 0)
    positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    x = model.embedding(ids).astype(getattr(jnp, model.config.residual_dtype))
    result = {'embedding': x}
    x = model.embedding_norm(x)
    result['embedding_norm'] = x
    for index, layer in enumerate(model.layers):
        if model.config.use_remat:
            x = nnx.remat(lambda block, h, s, p: block(h, s, p, packed=False))(
                layer, x, segments, positions)
        else:
            x = layer(x, segments, positions, packed=False)
        result[f'layer_{index:03d}'] = x
    x = jnp.where((segments >= 0)[..., None], model.final_norm(x), 0)
    result['encoder_output'] = x
    result['head_features'] = model.prediction_features(x)
    result['logits'] = model.decode(result['head_features'])
    return result


def comparison(a, b, active):
    # Preserve host sums of scalar losses as well as exact BF16/FP32 values.
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    delta = abs(a - b)
    return dict(finite=finite, exact=bool(np.array_equal(a, b)),
        max_abs=float(delta.max()) if finite else None,
        active_rows_max_abs=float(delta[active].max()) if finite and active.any() else (0.0 if finite else None))


def diagnose(model, inputs, targets, *, device_count):
    x, y = np.asarray(inputs), np.asarray(targets)
    if (x.ndim != 3 or x.shape != y.shape or not all(x.shape)
            or device_count < 1 or x.shape[1] % device_count):
        raise ValueError('Require nonempty matching microstep/batch/sequence arrays divisible by device count')
    if x.size * model.config.vocab_size > 1_000_000:
        raise ValueError('Forward diagnostic requires a small fixture: at most one million logits')
    graph, state = nnx.split(model)
    current = nnx.merge(graph, state)
    current.config = replace(model.config, yat_local_shards=False)
    current._local_projection_loss = True
    for layer in current.layers:
        layer.config = current.config
    instrumented = nnx.jit(stages)
    plain = nnx.jit(lambda m, ids: m(ids))
    loss_only = nnx.jit(lambda m, ids, labels: m(ids, labels, loss_reduction='sum'))
    differentiated = nnx.jit(nnx.value_and_grad(
        lambda m, ids, labels: m(ids, labels, loss_reduction='sum')))
    # JAX flattens dictionary outputs in key order, not execution order.
    stage_order = ['embedding', 'embedding_norm',
                   *[f'layer_{i:03d}' for i in range(len(current.layers))],
                   'encoder_output', 'head_features', 'logits']
    batch_values, single_values, batch_plain, single_plain = {}, {}, [], []
    loss_controls = []
    for microstep, labels in zip(x, y, strict=True):
        for batch, batch_labels in zip(np.split(microstep, device_count),
                                      np.split(labels, device_count), strict=True):
            traced = instrumented(current, jnp.asarray(batch))
            individual = [instrumented(current, jnp.asarray(row[None])) for row in batch]
            for name in stage_order:
                value = traced[name]
                batch_values.setdefault(name, []).append(np.asarray(value))
                single_values.setdefault(name, []).append(np.concatenate([np.asarray(v[name]) for v in individual]))
            batch_plain.append(np.asarray(plain(current, jnp.asarray(batch))))
            single_plain.extend(np.asarray(plain(current, jnp.asarray(row[None]))) for row in batch)
            # Request and synchronize the complete derivative output: a
            # forward-only executable need not use the same compiled arithmetic.
            forward_loss = loss_only(current, jnp.asarray(batch), jnp.asarray(batch_labels))
            ad_loss, grads = differentiated(current, jnp.asarray(batch), jnp.asarray(batch_labels))
            jax.block_until_ready((ad_loss, grads))
            reference_loss = sum(float(loss_only(current, jnp.asarray(row[None]),
                                  jnp.asarray(label[None])))
                                 for row, label in zip(batch, batch_labels, strict=True))
            loss_controls.append([float(forward_loss), float(ad_loss), reference_loss])
    arrays = {'inputs': x, 'targets': y, 'plain_batched': np.concatenate(batch_plain),
              'plain_per_example': np.concatenate(single_plain),
              'loss_controls': np.asarray(loss_controls, dtype=np.float64)}
    active = ((y >= 0) & (x != model.config.pad_token_id)).any(axis=-1).reshape(-1)
    rows = {}
    for name in batch_values:
        a, b = np.concatenate(batch_values[name]), np.concatenate(single_values[name])
        arrays['batched_' + name], arrays['per_example_' + name] = a, b
        rows[name] = comparison(a, b, active)
    controls = {mode: comparison(arrays[mode + '_logits'], arrays['plain_' + mode], active)
                for mode in ['batched', 'per_example']}
    loss_rows = arrays['loss_controls']
    active_groups = active.reshape(-1, x.shape[1] // device_count).any(axis=1)
    return arrays, dict(backend=jax.default_backend(), runtime=runtime_identity(),
        config=asdict(model.config), embedding_dtype=str(model.embedding.dtype),
        local_batch_size=x.shape[1] // device_count, captured_device_count=device_count,
        stage_order=list(rows), stages=rows, instrumentation_controls=controls,
        loss_control_columns=['forward_sum', 'autodiff_sum', 'per_example_forward_sum'],
        loss_controls={
            'forward_vs_autodiff': comparison(loss_rows[:, 0], loss_rows[:, 1], active_groups),
            'batched_vs_per_example_forward': comparison(loss_rows[:, 0], loss_rows[:, 2], active_groups)},
        uninstrumented_batching=comparison(arrays['plain_batched'], arrays['plain_per_example'], active),
        first_observed_stage_with_active_row_drift=next((k for k, v in rows.items() if v['active_rows_max_abs'] is not None and v['active_rows_max_abs'] > 0), None),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [*Path('flaxchat').glob('*.py'), Path(__file__)]},
        scope='Local batch shapes evaluated on one device. Full-logit stages and separate configured-MLM loss/autodiff controls; not the distributed accumulated training executable. Intermediate drift does not establish the cause of compiled training-gradient errors.',
        production_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arrays, receipt = read_numerical_evidence(args.capture)
    metadata = receipt['metadata']
    model = ModernBert(EncoderConfig(**metadata['config']), rngs=nnx.Rngs(0))
    restore_parameters(model, arrays, metadata['leaf_paths'])
    values, report = diagnose(model, arrays['inputs'], arrays['targets'], device_count=len(metadata['devices']))
    report['capture_sha256'] = receipt['archive_sha256']
    write_numerical_evidence(args.output, values, metadata=report)


if __name__ == '__main__':
    main()
