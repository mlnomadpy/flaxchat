"""Matched TPU MLM benchmarks, isolated by process and saved after every case.

Run under gcp_spot_supervisor. A failed kernel never qualifies for training.
Reports separate steady update throughput, compilation, checkpointing and cost.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


def preflight_cases(cases, data_root):
    """Reject undersized fixtures on CPU before renting accelerator time."""
    import numpy as np
    checked = []
    for batch, length in cases:
        directory = Path(data_root) / f'data-{length}'
        tokens = np.load(directory / 'tokens.npy', mmap_mode='r', allow_pickle=False)
        if tokens.ndim != 2 or tokens.dtype != np.int32 or tokens.shape[1] != length or len(tokens) < batch:
            raise ValueError(f'Case {batch}:{length} requires at least {batch} int32 rows of length {length}; got {tokens.shape}')
        checked.append(dict(batch=batch, length=length, available_rows=len(tokens)))
    return checked


def summarize_steps(records, warmup=5):
    steps = [r for r in records if r.get('event') == 'train_step']
    measured = steps[warmup:]
    if not measured or any(not r['updated'] for r in steps):
        raise ValueError('Insufficient accepted training updates')
    seconds = [r['seconds'] for r in measured]
    tokens = sum(r['tokens'] for r in measured)
    return dict(measured_steps=len(measured), warmup_steps=warmup,
        steady_tokens_per_second=tokens / sum(seconds), median_step_seconds=statistics.median(seconds),
        p90_step_seconds=sorted(seconds)[int(.9 * (len(seconds) - 1))],
        first_step_compile_and_execute_seconds=steps[0]['seconds'],
        checkpoint_seconds=sum(r['seconds'] for r in records if r.get('event') == 'checkpoint'),
        first_loss=steps[0]['loss'], final_loss=steps[-1]['loss'],
        masked_tokens=sum(r['masked_tokens'] for r in measured),
        overflow_steps=sum(r.get('projection_dense_fallback', False) for r in measured),
        memory=[r for r in records if r.get('event') == 'device_memory'])


def kernel_benchmark(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flaxchat.fused_cross_entropy import fused_cross_entropy
    if jax.default_backend() != 'tpu':
        raise RuntimeError('Physical TPU required')
    def reference(h, w, b, y):
        logits = (h @ w.T).astype(jnp.float32) + b
        target = jnp.take_along_axis(logits, jnp.maximum(y, 0)[:, None], axis=-1)[:, 0]
        return jnp.where(y >= 0, jax.nn.logsumexp(logits, -1) - target, 0).sum() / jnp.maximum((y >= 0).sum(), 1)
    def fused(h, w, b, y):
        return fused_cross_entropy(h, w, b, y, tile=args.tile).sum() / jnp.maximum((y >= 0).sum(), 1)
    checks = []
    # Irregular vocabulary, ignored labels and nonuniform bias exercise tails.
    for vocab in (4097, 256000):
        h = jax.random.normal(jax.random.key(1), (args.rows, 768)).astype(jnp.bfloat16)
        w = (jax.random.normal(jax.random.key(2), (vocab, 768)) * .02).astype(jnp.bfloat16)
        b = jnp.linspace(-.1, .1, vocab)
        y = (jnp.arange(args.rows) * 31 % vocab).at[0].set(-1).at[-1].set(vocab - 1)
        values = []
        for fn in (reference, fused) if args.tile else (reference,):
            values.append(jax.block_until_ready(jax.jit(jax.value_and_grad(fn, argnums=(0, 1, 2)))(h, w, b, y)))
        if args.tile:
            a, ga = values[0]
            z, gz = values[1]
            errors = [float(np.linalg.norm(np.asarray(x, dtype=np.float32) - np.asarray(v, dtype=np.float32)) /
                            max(float(np.linalg.norm(np.asarray(x, dtype=np.float32))), 1e-12)) for x, v in zip(ga, gz, strict=True)]
            passed = abs(float(a) - float(z)) < 1e-3 and max(errors) < .015
            checks.append(dict(vocabulary=vocab, loss_reference=float(a), loss_fused=float(z),
                               gradient_relative_l2=errors, passed=passed))
            if not passed:
                raise AssertionError(json.dumps(checks))
    fn = jax.jit(jax.value_and_grad(fused if args.tile else reference, argnums=(0, 1, 2)))
    start = time.perf_counter()
    compiled = fn.lower(h, w, b, y).compile()
    compile_seconds = time.perf_counter() - start
    for _ in range(5):
        jax.block_until_ready(compiled(h, w, b, y))
    times = []
    for _ in range(50):
        start = time.perf_counter()
        jax.block_until_ready(compiled(h, w, b, y))
        times.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    report = dict(kind='projection_forward_backward', tile=args.tile, checks=checks,
        passed=True, device_kind=jax.devices()[0].device_kind, jax_version=jax.__version__, median_seconds=statistics.median(times), compile_seconds=compile_seconds,
        projected_rows=args.rows, vocabulary=256000, hidden=768, devices_used=1,
        peak_memory_including_correctness_oracle=jax.devices()[0].memory_stats(),
        compilation_cache=os.environ.get('FLAXCHAT_COMPILATION_CACHE_DIR'),
        compiled_memory={name: getattr(memory, name, None) for name in
                         ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')})
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def campaign(args):
    preflight_cases([(4, 512), *args.cases], args.data_root)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()
    env = os.environ | {'JAX_PLATFORMS': 'tpu', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}
    def save():
        report = dict(scope='matched_synthetic_performance_not_quality', started_epoch=started,
            elapsed_seconds=time.time()-started, hourly_usd=args.hourly_usd,
            rate_is_estimate=True, results=results, planned_cases=args.cases, kernel_rows=args.kernel_rows,
            completed_training_cases=sum(r['name'].startswith('train-') for r in results))
        (root/'summary.json').write_text(json.dumps(report, indent=2))
        if args.prefix:
            subprocess.run(['gcloud','storage','cp',*map(str,root.glob('*.json')),
                            *map(str,root.glob('*.log')),args.prefix.rstrip('/')+'/'],check=True)
    def run(name, command, timeout=600):
        start = time.monotonic()
        try:
            with (root/f'{name}.log').open('w') as log:
                p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = 124
        result: dict[str, Any] = dict(name=name, returncode=code, seconds=time.monotonic()-start)
        results.append(result)
        if code:
            result['passed'] = False
        return result
    try:
        for rows in args.kernel_rows:
            for tile in (0, *args.tiles):
                name = f'kernel-{tile}-rows{rows}'
                r = run(name, [sys.executable,'-m','scripts.benchmark_encoder_projection','--kind','kernel',
                               '--tile',str(tile),'--rows',str(rows),'--output',str(root/f'{name}.json')], timeout=240)
                if r['returncode'] == 0:
                    r.update(json.loads((root/f'{name}.json').read_text()))
                save()
        candidates = {}
        for tile in args.tiles:
            passing = [r for r in results if r.get('tile') == tile and r.get('passed')]
            if {r['projected_rows'] for r in passing} == set(args.kernel_rows):
                candidates[tile] = sum(r['median_seconds'] / r['projected_rows'] for r in passing)
        best = min(candidates, key=lambda tile:candidates[tile]) if candidates else None
        variants = [('dense','xla',1024),('masked','xla',1024)]
        for backend, tile in [('xla_full',1024)] + ([('pallas',best)] if best else []):
            name = f'model-check-{backend}'
            r = run(name, [sys.executable,'-m','scripts.validate_encoder_projection','--backend',backend,
                           '--tile',str(tile),'--data-root',args.data_root,
                           '--output',str(root/f'{name}.json')], timeout=300)
            if (root/f'{name}.json').exists():
                r.update(json.loads((root/f'{name}.json').read_text()))
            r['passed'] = r['returncode'] == 0 and bool(r.get('passed'))
            if r.get('passed'):
                variants.append(('masked',backend,tile))
            save()
        for batch, length in args.cases:
            for projection, backend, tile in variants:
                if time.time() - started > args.max_seconds - 180:
                    save()
                    return False
                name = f'train-{projection}-{backend}-b{batch}-s{length}'
                command = [sys.executable,'-m','scripts.train_encoder','--config','artifacts/mmbert-base/config.json',
                    '--pretrained','artifacts/mmbert-base','--data',str(Path(args.data_root)/f'data-{length}'),
                    '--output',args.prefix.rstrip('/')+'/checkpoints/'+name,'--steps','55','--save-every','55',
                    '--batch-size',str(batch),'--mlm-projection',projection,'--mlm-loss-backend',backend,
                    '--mlm-vocab-tile',str(tile)]
                r = run(name, command)
                if r['returncode'] == 0:
                    records=[]
                    for line in (root/f'{name}.log').read_text().splitlines():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                    r.update(summarize_steps(records))
                    r.update(batch=batch, length=length, projection=projection, backend=backend, passed=True)
                    r['steady_usd_per_billion_input_tokens'] = args.hourly_usd*1e9/(3600*r['steady_tokens_per_second'])
                    r['invocation_usd_per_billion_input_tokens'] = args.hourly_usd*r['seconds']*1e9/(3600*batch*length*55)
                save()
    finally:
        save()
    from scripts.compare_encoder_projection import compare
    return compare(json.loads((root/'summary.json').read_text()))['complete']


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['kernel','campaign'], default='campaign')
    p.add_argument('--tile', type=int, default=1024)
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--kernel-rows', nargs='+', type=int, default=[128,512,1024])
    p.add_argument('--tiles', nargs='+', type=int, default=[512,1024,2048])
    p.add_argument('--cases', nargs='+', default=['4:512','16:512','32:512','4:2048'])
    p.add_argument('--output', required=True)
    p.add_argument('--prefix')
    p.add_argument('--data-root', default='artifacts/encoder-validation-0922')
    p.add_argument('--preflight-only', action='store_true', help='Validate all fixture sizes on CPU before provisioning')
    p.add_argument('--hourly-usd', type=float, default=4.013452)
    p.add_argument('--max-seconds', type=int, default=1500)
    a=p.parse_args()
    a.cases = [tuple(map(int, case.split(':'))) for case in a.cases]
    if any(len(case) != 2 or min(case) <= 0 for case in a.cases) or a.hourly_usd <= 0:
        p.error('Positive batch:sequence cases and hourly rate required')
    if a.preflight_only:
        print(json.dumps(preflight_cases([(4, 512), *a.cases], a.data_root), indent=2))
        raise SystemExit(0)
    if a.kind == 'campaign' and not a.prefix:
        p.error('campaign requires a fresh GCS prefix')
    if a.kind == 'kernel':
        kernel_benchmark(a)
    else:
        raise SystemExit(0 if campaign(a) else 1)
