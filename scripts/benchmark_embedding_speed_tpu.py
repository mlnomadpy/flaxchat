"""Small, physical-TPU-only inference comparison using the JAX harness.

Run once per model in a separate process. This measures one physical TPU device,
not whole-slice throughput or upstream Hugging Face GPU/FlashAttention speed.
Tokenization, download and loading are outside steady-state inference timing.
An external finite setup/workload lease and independent TPU cleanup are required.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 128), (8, 128), (1, 512), (8, 512))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('yat', 'mmbert'), required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--expected-weights-sha256', required=True)
    parser.add_argument('--model-directory', type=Path,
                        help='Existing authenticated snapshot, or omit to download on the TPU worker')
    parser.add_argument('--download-root', type=Path, default=Path('/tmp/embedding-speed-models'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    expected_repo = {'yat': 'mlnomad/yat-mmbert-base-embedding-v1', 'mmbert': 'jhu-clsp/mmBERT-base'}[args.role]
    if (args.repo != expected_repo or not re.fullmatch('[0-9a-f]{40}', args.revision)
            or not re.fullmatch('[0-9a-f]{64}', args.expected_weights_sha256)
            or not 1 <= args.warmups <= 10 or not 3 <= args.repeats <= 30):
        parser.error('Require exact supported public repository, immutable revision, weight SHA and bounded repetitions')
    if args.output.exists():
        parser.error('Refusing to overwrite a prior benchmark receipt')

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.public_encoder import load_public_encoder
    from scripts.train_encoder import load_pretrained

    if jax.default_backend() != 'tpu' or jax.process_count() != 1:
        raise RuntimeError('Benchmark requires a physical single-host TPU; CPU fallback is forbidden')
    devices = jax.local_devices()
    if not devices or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('Physical TPU devices required')
    device = devices[0]
    source_paths = sorted(ROOT.glob('flaxchat/**/*.py')) + [Path(__file__), ROOT / 'scripts/train_encoder.py', ROOT / 'scripts/convert_encoder_checkpoint.py']
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    packages = {}
    for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'safetensors', 'huggingface-hub', 'libtpu', 'orbax-checkpoint', 'torch'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {'format': 'flaxchat-light-embedding-speed-v1', 'complete': False,
              'role': args.role, 'repo': args.repo, 'revision': args.revision,
              'scope': 'single-physical-TPU-device/JAX-harness/encoder-plus-mean-pool-and-L2',
              'implementation': 'FlaxChat ModernBert; not upstream HF GPU implementation',
              'model_artifacts_sha256': {}, 'source_files_sha256': sources,
              'runtime': {'python': platform.python_version(), 'packages': packages,
                          'jax_default_matmul_precision': str(jax.config.jax_default_matmul_precision),
                          'jax_enable_x64': bool(jax.config.jax_enable_x64),
                          'xla_flags': os.environ.get('XLA_FLAGS', ''),
                          'jax_platforms': os.environ.get('JAX_PLATFORMS', '')},
              'hardware': {'measured_device': str(device), 'device_kind': device.device_kind,
                           'local_devices': [str(d) for d in devices], 'process_count': jax.process_count(),
                           'utilized_devices': 1},
              'protocol': {'shapes': [list(s) for s in SHAPES], 'warmups': args.warmups,
                           'repeats': args.repeats, 'inputs': 'deterministic full-length nonpadding token IDs',
                           'pooling': 'mean of nonpadding states including special tokens',
                           'normalization': 'L2 FP32', 'tokenization_included': False,
                           'host_output_transfer_included': False, 'synchronization': 'output.block_until_ready'},
              'cases': []}
    write_report(args.output, report)
    try:
        directory = args.model_directory
        if directory is None:
            from huggingface_hub import snapshot_download
            directory = Path(snapshot_download(repo_id=args.repo, revision=args.revision,
                             local_dir=args.download_root / args.role,
                             allow_patterns=['config.json', 'tokenizer.json', 'model.safetensors', 'pytorch_model.bin'], token=False))
        directory = Path(directory)
        source_weight = 'model.safetensors' if args.role == 'yat' else 'pytorch_model.bin'
        if digest(directory / source_weight) != args.expected_weights_sha256:
            raise ValueError('Actual public trained weight bytes differ from independently retained SHA256')
        if args.role == 'mmbert':
            from scripts.convert_encoder_checkpoint import convert
            if not (directory / 'model.safetensors').exists():
                conversion = convert(directory)  # Tensor format conversion only; no CPU model forward.
            else:
                conversion = json.loads((directory / 'conversion.json').read_text())
            if (conversion['source_sha256'] != args.expected_weights_sha256
                    or conversion['safetensors_sha256'] != digest(directory / 'model.safetensors')):
                raise ValueError('Converted mmBERT bytes are not bound to the pinned public PyTorch source')
            report['conversion'] = conversion
        for name in ('config.json', 'tokenizer.json', 'model.safetensors', source_weight):
            report['model_artifacts_sha256'][name] = digest(directory / name)
        original_config = json.loads((directory / 'config.json').read_text())
        begin = time.perf_counter()
        with jax.default_device(device):
            if args.role == 'yat':
                model = load_public_encoder(directory)
                config = model.config
                if (config.compute_dtype != 'bfloat16' or config.residual_dtype != 'float32'
                        or config.ffn_type != 'yat_glu' or config.attention_score != 'yat_softmax'):
                    raise ValueError('Require actual released YAT BF16/FP32-residual configuration')
            else:
                config = EncoderConfig.from_hf(original_config, compute_dtype='bfloat16',
                                              residual_dtype='float32', attention_backend='xla')
                model = ModernBert(config, rngs=nnx.Rngs(0))
                load_pretrained(model, directory)
            jax.block_until_ready(nnx.state(model))
            report['load_and_device_placement_seconds'] = time.perf_counter() - begin
            report['effective_encoder_config'] = asdict(config)
            report['original_snapshot_config'] = original_config
            report['precision_policy'] = 'BF16 feature computation / FP32 residuals and L2; model-specific YAT arithmetic retained'
            if config.vocab_size < 1024 or config.max_position_embeddings < 512:
                raise ValueError('Unexpected vocabulary/context for the requested shape matrix')

            @nnx.jit
            def forward(model, ids):
                pooled = model.pool(ids).astype(jnp.float32)
                return pooled / jnp.maximum(jnp.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12)

            for batch, length in SHAPES:
                raw = (512 + np.arange(batch * length, dtype=np.int32).reshape(batch, length) % 251)
                if np.any(raw == config.pad_token_id):
                    raise ValueError('Synthetic speed fixture must contain no padding')
                ids = jax.device_put(raw, device)
                ids.block_until_ready()
                first_start = time.perf_counter()
                values = forward(model, ids)
                values.block_until_ready()
                first_seconds = time.perf_counter() - first_start
                # Numerical checks and D2H copy are deliberately outside timed loops.
                host = np.asarray(values)
                norms = np.linalg.norm(host, axis=-1)
                if (host.shape != (batch, config.hidden_size) or not np.isfinite(host).all()
                        or not np.allclose(norms, 1, rtol=1e-4, atol=1e-4)):
                    raise ValueError('Nonfinite, collapsed, or malformed normalized embeddings')
                for _ in range(args.warmups):
                    forward(model, ids).block_until_ready()
                elapsed = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    forward(model, ids).block_until_ready()
                    elapsed.append(time.perf_counter() - start)
                median = statistics.median(elapsed)
                report['cases'].append({'batch_size': batch, 'sequence_length': length,
                    'input_int32_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'first_call_compile_and_execute_seconds': first_seconds,
                    'steady_seconds': elapsed, 'median_latency_ms': median * 1000,
                    'mean_latency_ms': statistics.mean(elapsed) * 1000,
                    'min_latency_ms': min(elapsed) * 1000,
                    'sequences_per_second': batch / median, 'tokens_per_second': batch * length / median,
                    'normalized_output_sha256': hashlib.sha256(host.tobytes()).hexdigest(),
                    'finite_output': True, 'output_shape': list(host.shape),
                    'device_memory_stats': device.memory_stats()})
                write_report(args.output, report)
                print(json.dumps({'role': args.role, **report['cases'][-1]}, sort_keys=True), flush=True)
        report['complete'] = True
        write_report(args.output, report)
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        write_report(args.output, report)
        raise


if __name__ == '__main__':
    main()
