"""Synchronized YAT forward/backward microbenchmarks; no TPU provisioning.

Run with python -m scripts.benchmark_yat --output report.json on the target
machine. CPU results do not qualify TPU performance or training quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from flaxchat.encoder import bidirectional_attention, yat_glu
from scripts.numerical_evidence import write_numerical_evidence


def measure(operation, inputs, *, repeats, failure_directory=None):
    forward = jax.jit(operation)
    output = jax.block_until_ready(forward(*inputs))
    train = jax.jit(jax.value_and_grad(
        lambda *xs: jnp.mean(operation(*xs).astype(jnp.float32) ** 2),
        argnums=tuple(range(len(inputs)))))
    start = time.perf_counter()
    executable = train.lower(*inputs).compile()
    compile_seconds = time.perf_counter() - start
    for _ in range(3):
        result = jax.block_until_ready(executable(*inputs))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(executable(*inputs))
        samples.append(time.perf_counter() - start)
    memory = executable.memory_analysis()
    finite = all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves((output, result)))
    if not finite:
        if failure_directory is not None:
            arrays = {f'input_{i}': value for i, value in enumerate(inputs)}
            arrays['output'] = output
            arrays['loss'] = result[0]
            arrays.update({f'gradient_{i}': value for i, value in enumerate(result[1])})
            write_numerical_evidence(failure_directory, arrays, metadata={
                'reason': 'Nonfinite YAT output or gradient',
                'backend': jax.default_backend(), 'jax_version': jax.__version__,
                'devices': [str(device) for device in jax.devices()],
            })
        raise ValueError('Nonfinite YAT output or gradient')
    return output, {
        'forward_backward_median_seconds': float(np.median(samples)),
        'forward_backward_samples_seconds': samples,
        'compile_seconds': compile_seconds,
        'temporary_bytes': memory.temp_size_in_bytes if memory else None,
        'forward_stablehlo_has_f32': 'f32' in str(forward.lower(*inputs).compiler_ir(dialect='stablehlo')),
        'finite': finite,
    }


def benchmark(*, repeats=15, collision_stress=False, failure_directory=None):
    x = jax.random.normal(jax.random.key(41), (128, 768)).astype(jnp.bfloat16)
    w = (jax.random.normal(jax.random.key(42), (768, 2304)) * .02).astype(jnp.bfloat16)
    q = jax.random.normal(jax.random.key(43), (1, 128, 4, 64)).astype(jnp.bfloat16)
    k = jax.random.normal(jax.random.key(44), q.shape).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.key(45), q.shape).astype(jnp.bfloat16)
    segments = jnp.zeros((1, 128), jnp.int32)
    alpha = jnp.bfloat16(1)
    cases = [('ffn_random', (x, w, alpha)),
             ('ffn_one_close_pair', (x, w.at[:, 0].set(x[0]), alpha)),
             ('attention_random', (q, k, v, alpha)),
             ('attention_equal_qk', (q, q, v, alpha))]
    if collision_stress:
        columns = jnp.array([0, 127, 256, 511, 768, 1151])
        rows = jnp.array([0, 24, 48, 72, 96, 127])
        cases.extend([
            ('ffn_scattered_collisions', (x, w.at[:, columns].set(x[rows].T), alpha)),
            ('ffn_dense_collisions', (x, w.at[:, :1152].set(jnp.tile(x.T, (1, 9))), alpha)),
        ])
    results = {}
    for case, inputs in cases:
        results[case] = {}
        reference = None
        for mode in ('mixed', 'bf16_adaptive', 'bf16'):
            if case.startswith('ffn'):
                def ffn_operation(x, w, alpha, mode=mode):
                    return yat_glu(x, w, alpha=alpha, compute_mode=mode)
                operation = ffn_operation
            else:
                def attention_operation(q, k, v, alpha, mode=mode):
                    return bidirectional_attention(q, k, v, segments, score='yat_softmax',
                                                   alpha=alpha, yat_compute_mode=mode)
                operation = attention_operation
            output, report = measure(operation, inputs, repeats=repeats,
                failure_directory=(Path(failure_directory) / case / mode
                                   if failure_directory is not None else None))
            output = output.astype(jnp.float32)
            if reference is None:
                reference = output
            report['relative_forward_l2_vs_mixed'] = float(
                jnp.linalg.norm(output - reference) / jnp.maximum(jnp.linalg.norm(reference), 1e-30))
            results[case][mode] = report
    root = Path(__file__).resolve().parents[1]
    return {'backend': jax.default_backend(), 'jax_version': jax.__version__,
            'devices': [str(d) for d in jax.devices()], 'repeats': repeats,
            'collision_stress': collision_stress,
            'scope': 'single-device microbenchmark; not end-to-end training qualification',
            'source_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                              for p in ('flaxchat/yat.py', 'flaxchat/encoder.py', 'scripts/benchmark_yat.py')},
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--collision-stress', action='store_true',
                        help='Include scattered and dense FFN near-collision patterns')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('--repeats must be at least 3')
    report = benchmark(repeats=args.repeats, collision_stress=args.collision_stress,
                       failure_directory=args.output.with_suffix('.failures'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'backend': report['backend']}))


if __name__ == '__main__':
    main()
