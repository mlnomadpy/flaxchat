"""Bounded embedding admission worker for an already allocated physical TPU slice.

Launch this identical command on EVERY TPU VM worker, under the cloud supervisor
and an outer timeout. It never provisions hardware. Requires a fresh GCS output.
All model computation, including replicated references, stays on physical TPUs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import tempfile
import time
from urllib.parse import urlparse


def validate_ownership(records, *, devices, processes, batch):
    """Host-only admission of actual rank/device/row inventories."""
    if processes < 2 or devices < processes or batch % devices or devices % processes:
        raise ValueError('Need a declared genuine multi-host topology and aligned batch')
    if len(records) != processes or {r['rank'] for r in records} != set(range(processes)):
        raise ValueError('Incomplete or duplicated physical process inventory')
    rows, owned_devices = [], []
    sources = set()
    kinds = set()
    outputs = set()
    for record in records:
        if (record['backend'] != 'tpu' or record['processes'] != processes
                or record['global_devices'] != devices or len(record['local_devices']) != devices // processes):
            raise ValueError('Physical topology differs from declaration')
        if len(record['rows']) != batch // processes:
            raise ValueError('Host does not own its declared global row slice')
        rows.extend(record['rows'])
        owned_devices.extend(record['local_devices'])
        sources.add((record['source_sha256'], record['worker_sha256'], record['runtime_sha256']))
        kinds.add(record['device_kind'])
        outputs.add(record['campaign_output'])
    if sorted(rows) != list(range(batch)) or len(set(owned_devices)) != devices:
        raise ValueError('Rows/devices overlap or global ownership is incomplete')
    if len(sources) != 1 or len(kinds) != 1 or len(outputs) != 1:
        raise ValueError('Hosts disagree on source or physical device kind')
    return {'passed': True, 'physical_processes': processes, 'global_devices': devices,
            'global_rows': batch, 'device_kind': next(iter(kinds)), 'source_sha256': next(iter(sources))[0],
            'worker_sha256': next(iter(sources))[1], 'runtime_sha256': next(iter(sources))[2]}


def compare_manifests(left, right):
    checks = {key: left.get(key) == right.get(key) and bool(left.get(key))
              for key in ('model_state', 'optimizer_state', 'training_state')}
    if not all(checks.values()):
        raise ValueError(f'Interrupted embedding stage differs from uninterrupted: {checks}')
    return checks


def _fixture(directory, *, batch, pair_only):
    """Use real public export and preparer APIs; no copied training loop."""
    from flax import nnx
    import tokenizers
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.public_encoder import export_state_safetensors
    from flaxchat.encoder_data import file_hash
    from scripts.prepare_yat_embedding_finetune import _tokenize
    parent = directory / 'parent'
    parent.mkdir()
    train_count = max(32, batch * 2)
    dev_start = train_count + 10
    vocab_size = max(128, dev_start + 24)
    config = EncoderConfig(vocab_size=vocab_size, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        local_attention=16, ffn_type='yat_glu', attention_score='yat_softmax',
        compute_dtype='bfloat16', residual_dtype='float32', use_remat=False)
    model = ModernBert(config, rngs=nnx.Rngs(29))
    export_state_safetensors(nnx.to_pure_dict(nnx.state(model)), parent / 'model.safetensors')
    (parent / 'config.json').write_text(json.dumps(asdict(config), sort_keys=True))
    vocab = {'[PAD]': 0, '[UNK]': 1, 'query': 2, 'document': 3, '[MASK]': 4, 'negative': 5}
    vocab.update({str(i): i + 6 for i in range(vocab_size - 6)})
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(parent / 'tokenizer.json'))
    public_manifest = parent / 'manifest.json'
    public_manifest.write_text(json.dumps({name: file_hash(parent / name)
        for name in ('config.json', 'tokenizer.json', 'model.safetensors')}, sort_keys=True))
    folder = directory / 'pairs'
    folder.mkdir()
    for split, span in [('train', range(train_count)), ('dev', range(dev_start, dev_start + 8))]:
        rows = [dict(query=f'query {i}', positive=f'document {i}',
                     negative=None if pair_only else f'negative {i}', group=str(i), language='en') for i in span]
        (folder / f'{split}.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    tokens = _tokenize(folder, parent / 'tokenizer.json', 16, 16, 0)
    manifest = dict(format='flaxchat-yat-embedding-triplets-v2', source='pairs',
        tokenizer_sha256=file_hash(parent / 'tokenizer.json'), rows={'train': train_count, 'dev': 8},
        raw_files={f'{split}.jsonl': file_hash(folder / f'{split}.jsonl') for split in ('train', 'dev')},
        query_length=16, document_length=16, pad_id=0, vocab_size=vocab_size, tokenization=tokens)
    (folder / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    # Return source-independent paths; all hashes must agree across hosts.
    del model
    return parent, public_manifest, folder


def _loss_case(batch, mesh, *, rank, processes):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax.experimental import multihost_utils
    from flaxchat.contrastive import hard_negative_infonce
    from flaxchat.training import place_host_batch
    rng = np.random.default_rng(29)
    full = [rng.normal(size=(batch, 128)).astype(np.float32) for _ in range(3)]
    ids = np.arange(batch, dtype=np.int32) + 1
    qid, pid, nid, gid = ids.copy(), ids + 10000, ids + 20000, ids.copy()
    # Alternative positive resides on the last remote rank; another known
    # positive never appears in the positive pool, but occurs as mined negative.
    qid[-1] = qid[0]
    gid[-1] = gid[0]
    nid[0], nid[1] = pid[-1], 999999
    known = np.zeros((batch, 3), dtype=np.int32)
    known[:, 0] = pid
    known[0] = known[-1] = [pid[0], pid[-1], 999999]
    valid = np.ones(batch, dtype=bool)
    local = batch // processes
    take = slice(rank * local, (rank + 1) * local)
    sharded = [place_host_batch(value[take], mesh) for value in full]
    replicated = [jax.device_put(value, NamedSharding(mesh, P())) for value in full]
    labels = [jax.device_put(value, NamedSharding(mesh, P())) for value in (qid, pid, nid, gid, valid, known)]
    def objective(q, p, n):
        return hard_negative_infonce(q, p, n, *labels[:5], known_positive_text_ids=labels[5],
                                     expected_global_batch=batch, temperature=.2)
    evaluate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))
    loss, gradients = evaluate(*sharded)
    reference, reference_gradients = evaluate(*replicated)
    np.testing.assert_allclose(float(loss), float(reference), rtol=2e-5, atol=2e-5)
    errors = []
    for actual, expected in zip(gradients, reference_gradients, strict=True):
        # Gathering global arrays replicates their global content, not local
        # rank losses. It is a physical TPU collective in this worker.
        actual = np.asarray(multihost_utils.process_allgather(actual, tiled=True))
        expected = np.asarray(multihost_utils.process_allgather(expected, tiled=True))
        if actual.shape != (batch, 128) or not np.isfinite(actual).all():
            raise ValueError('Global embedding gradients incomplete/nonfinite')
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-5)
        if np.linalg.norm(actual[take]) == 0:
            raise ValueError('Host-local encoder cotangents vanished')
        errors.append(float(np.max(np.abs(actual - expected))))
    _, masking = hard_negative_infonce(*sharded, *labels[:5], known_positive_text_ids=labels[5],
                                       expected_global_batch=batch, temperature=.2, return_metrics=True)
    if int(masking['masked_known_positive_negatives']) != 4:
        raise ValueError('Cross-host or out-of-batch known positives were not masked')
    return {'passed': True, 'masking': {key: int(value) for key, value in masking.items()}, 'loss': float(loss), 'replicated_reference_loss': float(reference),
            'gradient_max_absolute_errors': errors, 'global_pairs': batch,
            'cross_host_known_positive_fixture': True, 'reference_backend': 'tpu'}


def _encoder_gradient_case(parent, folder, batch, mesh, *, rank, processes, pair_only, cache_chunk):
    """Actual BF16 YAT parameters must receive the same global cotangents."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax.experimental import multihost_utils
    from flaxchat.common import replicate_on_mesh
    from flaxchat.public_encoder import load_public_encoder
    from flaxchat.embedding_gradient_cache import nnx_cached_pool
    from flaxchat.contrastive import hard_negative_infonce
    from flaxchat.training import place_host_batch
    full = [np.load(folder / 'train' / f'{field}_tokens.npy', allow_pickle=False)[:batch]
            for field in ('query', 'positive', 'negative')]
    local = batch // processes
    take = slice(rank * local, (rank + 1) * local)
    sharded = [place_host_batch(rows[take], mesh) for rows in full]
    replicated = [jax.device_put(rows, NamedSharding(mesh, P())) for rows in full]
    ids = jnp.arange(batch, dtype=jnp.int32) + 1
    def objective(model, inputs, cached):
        pool = (lambda rows: nnx_cached_pool(model, rows, cache_chunk)) if cached and cache_chunk else model.pool
        q, p = pool(inputs[0]), pool(inputs[1])
        n = jnp.zeros_like(p) if pair_only else pool(inputs[2])
        return hard_negative_infonce(q, p, n, ids, ids + 10000, ids + 20000, ids,
            jnp.full(batch, not pair_only), expected_global_batch=batch,
            known_positive_text_ids=(ids + 10000)[:, None])
    direct = load_public_encoder(parent)
    reference = load_public_encoder(parent)
    for model in (direct, reference):
        nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    @nnx.jit
    def global_grad(model):
        return nnx.value_and_grad(lambda inner: objective(inner, sharded, True))(model)
    @nnx.jit
    def replicated_grad(model):
        return nnx.value_and_grad(lambda inner: objective(inner, replicated, False))(model)
    loss, gradients = global_grad(direct)
    expected_loss, expected_gradients = replicated_grad(reference)
    np.testing.assert_allclose(float(loss), float(expected_loss), rtol=2e-3, atol=2e-3)
    if jax.tree.structure(gradients) != jax.tree.structure(expected_gradients):
        raise ValueError('YAT parameter gradient tree differs from replicated reference')
    differences, squared_error, squared_reference = [], 0., 0.
    for actual, expected in zip(jax.tree.leaves(gradients), jax.tree.leaves(expected_gradients), strict=True):
        actual = np.asarray(multihost_utils.process_allgather(actual, tiled=True)).astype(np.float64)
        expected = np.asarray(multihost_utils.process_allgather(expected, tiled=True)).astype(np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('YAT parameter gradients are incomplete/nonfinite')
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-3)
        differences.append(float(np.max(np.abs(actual - expected))))
        squared_error += float(np.square(actual - expected).sum())
        squared_reference += float(np.square(expected).sum())
    relative = np.sqrt(squared_error / max(squared_reference, 1e-30))
    if relative > .03:
        raise ValueError(f'Global YAT gradient relative L2 drift exceeds admission: {relative}')
    return {'passed': True, 'loss': float(loss), 'replicated_reference_loss': float(expected_loss),
            'parameter_leaves': len(differences), 'maximum_absolute_error': max(differences),
            'relative_l2_error': float(relative), 'loss_rtol': .002, 'loss_atol': .002,
            'gradient_rtol': .02, 'gradient_atol': .002, 'gradient_relative_l2_limit': .03,
            'reference_backend': 'tpu', 'compute_dtype': 'bfloat16', 'cache_chunk': cache_chunk}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Fresh GCS campaign prefix')
    parser.add_argument('--expected-devices', type=int, required=True)
    parser.add_argument('--expected-processes', type=int, required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--encoder-chunk-size', type=int, default=0)
    parser.add_argument('--pair-only', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=1200)
    args = parser.parse_args(argv)
    parsed = urlparse(args.output)
    if (parsed.scheme != 'gs' or not parsed.netloc or not parsed.path.strip('/')
            or args.expected_processes < 2 or not 60 <= args.timeout_seconds <= 1800):
        parser.error('Fresh GCS prefix, >=2 declared processes and 60–1800 second limit required')
    batch = args.batch_size if args.batch_size is not None else 2 * args.expected_devices
    if (args.expected_devices < args.expected_processes or args.expected_devices % args.expected_processes
            or batch < 2 or batch % args.expected_devices or args.encoder_chunk_size < 0
            or args.encoder_chunk_size and (batch % args.encoder_chunk_size or args.encoder_chunk_size % args.expected_devices)):
        parser.error('Declared global batch/topology must align')
    started = time.monotonic()
    def deadline(signum, frame):
        raise TimeoutError('Embedding multi-host worker deadline exceeded; outer timeout must bound blocked collectives')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    # Must precede backend discovery, imported model modules and every JAX op.
    import jax
    jax.distributed.initialize(initialization_timeout=min(120, args.timeout_seconds // 2))
    import numpy as np
    import fsspec
    from jax.sharding import Mesh
    from jax.experimental import multihost_utils
    from flaxchat.training import gather_process_metadata
    from flaxchat.embedding_contract import source_identity, canonical_hash
    from flaxchat.encoder_data import file_hash
    from flaxchat.runtime import runtime_identity
    from flaxchat.checkpoint import create_checkpoint_manager, load_checkpoint_metadata
    from flaxchat.embedding_stage import add_stage_arguments
    from scripts.train_yat_embedding_finetune import run
    rank, processes = jax.process_index(), jax.process_count()
    report = {'passed': False, 'rank': rank, 'scope': 'physical-embedding-multihost-admission-v1',
              'cases': {}, 'pair_only': args.pair_only, 'encoder_chunk_size': args.encoder_chunk_size,
              'runtime': runtime_identity(), 'worker_sha256': file_hash(Path(__file__)),
              'limitations': ['Four-update synthetic encoder admission; not sustained production quality or throughput.',
                             'Only this device family/chip/host count and selected pair/cache policy are qualified.',
                             'No transport-disconnect or abrupt process-kill claim; separate fault suite required.']}
    fs, prefix = fsspec.core.url_to_fs(args.output.rstrip('/'))
    owns_output = False
    try:
        if jax.default_backend() != 'tpu' or processes < 2 or processes != args.expected_processes or jax.device_count() != args.expected_devices:
            raise ValueError('Requires genuine physical TPU multi-host execution with declared topology')
        if fs.exists(prefix):
            raise ValueError('Campaign prefix exists; preserve evidence and use a new identity')
        multihost_utils.sync_global_devices('embedding-fresh-prefix')
        owns_output = True
        local = batch // processes
        source = source_identity(Path(__file__).resolve().parents[1])
        record = {'rank': rank, 'backend': jax.default_backend(), 'processes': processes,
                  'global_devices': jax.device_count(), 'local_devices': [device.id for device in jax.local_devices()],
                  'device_kind': jax.local_devices()[0].device_kind, 'rows': list(range(rank * local, (rank + 1) * local)),
                  'source_sha256': source['sha256'], 'worker_sha256': report['worker_sha256'],
                  'runtime_sha256': canonical_hash(report['runtime']), 'campaign_output': args.output.rstrip('/')}
        records = gather_process_metadata(record)
        report['topology'] = validate_ownership(records, devices=args.expected_devices, processes=args.expected_processes, batch=batch)
        report['ownership_records'] = records
        mesh = Mesh(np.array(jax.devices()), ('data',))
        report['cases']['global_loss_gradient_reference'] = _loss_case(batch, mesh, rank=rank, processes=processes)
        with tempfile.TemporaryDirectory(prefix='embedding-multihost-') as temporary:
            parent, manifest, folder = _fixture(Path(temporary), batch=batch, pair_only=args.pair_only)
            report['fixture'] = {'encoder': json.loads((parent / 'config.json').read_text()),
                'public_files_sha256': json.loads(manifest.read_text()),
                'prepared_manifest_sha256': file_hash(folder / 'manifest.json')}
            report['cases']['yat_global_parameter_gradients'] = _encoder_gradient_case(parent, folder, batch, mesh,
                rank=rank, processes=processes, pair_only=args.pair_only, cache_chunk=args.encoder_chunk_size)
            base = ['--parent-public', str(parent), '--parent-manifest', str(manifest), '--data', f'pairs={folder}',
                    '--source-weights', 'pairs=1', '--steps', '4', '--warmup', '1', '--batch-size', str(batch),
                    '--save-every', '1', '--eval-every', '4', '--keep-checkpoints', '1', '--dev-max-rows', '8',
                    '--max-dev-regression', '1', '--distributed', '--encoder-chunk-size', str(args.encoder_chunk_size)]
            stage_parser = add_stage_arguments(argparse.ArgumentParser())
            for name, extra in [('baseline', []), ('recovery', ['--stop-after', '2']), ('recovery', ['--resume'])]:
                run(stage_parser.parse_args(base + ['--output', args.output.rstrip('/') + '/' + name] + extra))
                multihost_utils.sync_global_devices('embedding-stage-' + name + ('-resume' if extra == ['--resume'] else '-segment'))
            manifests = []
            for name in ('baseline', 'recovery'):
                manager = create_checkpoint_manager(args.output.rstrip('/') + '/' + name, async_checkpointing=False)
                try:
                    import orbax.checkpoint as ocp
                    value = manager.restore(4, args=ocp.args.Composite(manifest=ocp.args.JsonRestore()))['manifest']
                    manifests.append(value)
                finally:
                    manager.close()
            comparisons = compare_manifests(*manifests)
            best = [load_checkpoint_metadata(args.output.rstrip('/') + '/' + name + '/best', include_receipt=True)
                    for name in ('baseline', 'recovery')]
            if best[0]['committed_receipt'] != best[1]['committed_receipt'] or best[0]['quality_selection'] != best[1]['quality_selection']:
                raise ValueError('Multi-host interrupted stage changed selected best artifact')
            report['cases']['trainer_exact_recovery'] = {'passed': True, 'completed_updates': 4,
                'off_cadence_stop': 2, 'evaluation_cadence': 4, 'exact_manifest_comparisons': comparisons,
                'best_step': best[0]['step'], 'best_manifest_sha256': best[0]['committed_receipt']['manifest_sha256']}
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        # Each physical rank owns its evidence path; no last-writer overwrite.
        if owns_output:
            with fs.open(f'{prefix}/receipts/rank-{rank}.json', 'w') as stream:
                json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        else:
            print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    gathered = gather_process_metadata({'rank': rank, 'passed': report['passed'], 'cases': report['cases']})
    if len(gathered) != processes or any(row['passed'] is not True for row in gathered):
        raise ValueError('Not every physical host passed embedding admission')
    if rank == 0:
        report['rank_results'] = gathered
        with fs.open(f'{prefix}/receipts/summary.json', 'w') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    signal.alarm(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
