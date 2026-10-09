"""Read-only physical-TPU diagnosis, external to frozen model source."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def child(args):
    sys.path.insert(0, str(args.root / 'source'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from flaxchat.encoder import ModernBert, EncoderConfig
    from flaxchat.checkpoint import load_checkpoint_metadata, restore_model_from_checkpoint
    from flaxchat.common import replicate_on_mesh
    from flaxchat.training import place_host_batch
    from flaxchat.encoder_data import load_prepared_rows, file_hash
    from flaxchat.mlm import mask_tokens
    from scripts.evaluate_encoder import evaluation_order
    from scripts.representation_run import source_snapshot, source_tree_digest
    if jax.default_backend() != 'tpu' or jax.device_count() != 8 or jax.process_count() != 1:
        raise RuntimeError('Diagnostic requires one physical eight-device TPU host')
    original = read(args.root / 'original-qualification.json')
    source_sha = source_tree_digest(source_snapshot(args.root / 'source'))
    if (source_sha != original['source_tree_sha256']
            or os.environ.get('FLAXCHAT_RUNTIME_LOCK_SHA256') != original['runtime_lock_sha256']):
        raise ValueError('Frozen source or runtime differs from original qualification')
    reference = read(args.root / 'original-evaluate2000.json')
    destination = args.evidence / f'diagnostic-{args.step}.json'
    result = dict(step=args.step, source_tree_sha256=source_sha,
        runtime_lock_sha256=os.environ.get('FLAXCHAT_RUNTIME_LOCK_SHA256'),
        manifest_sha256=os.environ.get('FLAXCHAT_RUN_MANIFEST_SHA256'),
        diagnostic_sha256=file_sha(__file__), backend='tpu', devices=8,
        status='running', weights=[], losses=[], layers=[])
    def emit():
        write(destination, result)
    emit()
    try:
        metadata = load_checkpoint_metadata(args.output, step=args.step)
        config = EncoderConfig(**metadata['resolved_config']['encoder'])
        if config.attention_score != 'goat_input':
            raise ValueError('Expected corrected GOAT-input checkpoint')
        identity = metadata['resolved_config']
        train_dir, validation_dir = args.root / 'corpus/train', args.root / 'corpus/validation'
        if (file_hash(train_dir / 'manifest.json') != metadata['data_manifest_identity']
                or file_hash(validation_dir / 'manifest.json') != reference['validation_manifest_sha256']):
            raise ValueError('Data identity differs')
        validation, manifest = load_prepared_rows(validation_dir, config)
        if (manifest['tokenizer_sha256'] != metadata['tokenizer_identity']
                or manifest['special_token_ids'] != identity['special_token_ids']):
            raise ValueError('Tokenizer/special-token policy differs')
        selected_path = args.evidence / 'selected-rows.json'
        if selected_path.exists():
            chosen = read(selected_path)
        else:
            train, _ = load_prepared_rows(train_dir, config)
            def rowhash(row):
                return hashlib.sha256(np.asarray(row[row != config.pad_token_id], dtype='<i4').tobytes()).digest()
            training_rows = {rowhash(row) for row in train}
            order, _ = evaluation_order(validation_dir, len(validation), seed=2026, balanced_languages=True)
            chosen, seen = [], set()
            for i in order:
                hashed = rowhash(validation[i])
                if hashed in training_rows or hashed in seen:
                    continue
                seen.add(hashed)
                chosen.append(int(i))
                if len(chosen) == 512:
                    break
            write(selected_path, chosen)
            del training_rows, train
        selected_hash = hashlib.sha256(np.asarray(chosen, dtype='<i8').tobytes()).hexdigest()
        if selected_hash != reference['selected_rows_sha256']:
            raise ValueError('Heldout row selection differs')
        result.update(selected_rows_sha256=selected_hash, reference_evaluation_sha256=file_sha(args.root / 'original-evaluate2000.json'))
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('data',))
        model = ModernBert(config, rngs=nnx.Rngs(0))
        nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
        restore_model_from_checkpoint(model, args.output, step=args.step,
            expected_identity={'resolved_config': identity})
        @jax.jit
        def summary(value):
            value = value.astype(jnp.float32)
            return jnp.stack([jnp.isfinite(value).all().astype(jnp.float32),
                jnp.max(jnp.abs(value)), jnp.min(value), jnp.max(value)])
        def statistics(value):
            finite, absmax, low, high = np.asarray(summary(value))
            return dict(finite=bool(finite), absmax=finite_number(absmax),
                        minimum=finite_number(low), maximum=finite_number(high))
        for path, variable in nnx.state(model, nnx.Param).flat_state():
            stats = statistics(variable[...])
            result['weights'].append(dict(path='/'.join(map(str, path)), **stats))
        result['parameters_finite'] = all(x['finite'] for x in result['weights'])
        result['alphas'] = [x for x in result['weights'] if 'alpha' in x['path']]
        emit()
        if not result['parameters_finite']:
            result['status'] = 'nonfinite_parameters'
            return
        loss_fn = nnx.jit(lambda m, x, y: m(x, y))
        first_bad = None
        # Scan the SAME heldout rows with identical masks until first failing
        # batch, rather than assuming the earlier evaluator failed on batch0.
        numerator, denominator = 0., 0
        for start in range(0, len(chosen), 8):
            indices = np.asarray(chosen[start:start + 8])
            x, y = mask_tokens(validation[indices], seed=2026, step=0, example_ids=indices,
                vocab_size=config.vocab_size, mask_token_id=config.mask_token_id,
                special_token_ids=identity['special_token_ids'], probability=identity['mask_probability'])
            count = int((y >= 0).sum())
            value = float(loss_fn(model, place_host_batch(x, mesh), place_host_batch(y, mesh)))
            result['losses'].append(dict(batch_start=start, batch_size=8, loss=finite_number(value), masked_tokens=count))
            numerator += value * count
            denominator += count
            emit()
            if not math.isfinite(value):
                first_bad = (x, y, indices)
                break
        result['evaluation_loss'] = finite_number(numerator / max(denominator, 1))
        if first_bad is None:
            result.update(status='heldout_finite', masked_tokens=denominator)
            if denominator != reference['masked_tokens']:
                raise ValueError('Heldout mask identity differs')
            return
        x, y, indices = first_bad
        result['failing_example_ids'] = indices.tolist()
        np.savez(args.evidence / f'failing-batch-{args.step}.npz', inputs=x, labels=y, indices=indices)
        # Same examples/masks repeated: the ideal mean loss is invariant.
        for batch_size in (16, 64):
            copies = batch_size // len(x)
            xx, yy = np.tile(x, (copies, 1)), np.tile(y, (copies, 1))
            value = float(loss_fn(model, place_host_batch(xx, mesh), place_host_batch(yy, mesh)))
            result['losses'].append(dict(batch_size=batch_size, repeated_failing_batch=True, loss=finite_number(value)))
            emit()
        ids = place_host_batch(x, mesh)
        segments = jnp.where(ids == config.pad_token_id, -1, 0)
        positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
        h = nnx.jit(lambda m, ids: m.embedding_norm(m.embedding(ids).astype(jnp.float32)))(model, ids)
        result['layers'].append(dict(layer='embedding_norm', **statistics(h)))
        layer_fn = nnx.jit(lambda block, h, s, p: block(h, s, p, packed=False))
        for index, block in enumerate(model.layers):
            h = layer_fn(block, h, segments, positions)
            stats = statistics(h)
            result['layers'].append(dict(layer=index, **stats))
            emit()
            if not stats['finite']:
                break
        if result['layers'][-1]['finite']:
            h = nnx.jit(lambda m, h: m.final_norm(h))(model, h)
            result['layers'].append(dict(layer='final_norm', **statistics(h)))
            features = nnx.jit(lambda m, h: m.prediction_features(h))(model, h)
            result['layers'].append(dict(layer='mlm_head', **statistics(features)))
        result['status'] = 'nonfinite_heldout_localized'
    except BaseException as error:
        result.update(status='diagnostic_error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        emit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', default='gs://azettaai-yat-eval-0929/goat-input-1007b/yat-embed-torch-goat-input-1007b/input-goat-scratch-base-mlm/checkpoints')
    p.add_argument('--evidence', type=Path)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--evidence-prefix')
    p.add_argument('--step', type=int, choices=[2500, 2750, 3000])
    p.add_argument('--max-seconds', type=int, default=900)
    args = p.parse_args()
    if args.evidence is None:
        args.evidence = args.root / 'diagnostic-evidence'
    if args.preflight_only:
        if not (args.root / 'source/flaxchat/encoder.py').is_file():
            raise ValueError('Frozen source missing')
        original = read(args.root / 'original-qualification.json')
        reference = read(args.root / 'original-evaluate2000.json')
        if (original.get('manifest_sha256') != '3e87cbe501a2b568f7028dc69661f1e7c013d4e50f4ef144df79abdae4b3dcfc'
                or reference.get('checkpoint_step') != 2000
                or reference.get('masked_token_loss') != 3.869532731642492
                or not (args.root / 'corpus/validation/manifest.json').is_file()):
            raise ValueError('Diagnostic reference metadata missing or changed')
        return
    args.evidence.mkdir(parents=True, exist_ok=True)
    if args.step:
        child(args)
        return
    if not 120 < args.max_seconds <= 3300:
        raise ValueError('Invalid bounded diagnostic duration')
    deadline = time.monotonic() + args.max_seconds
    status = {'status': 'running', 'steps': [], 'diagnostic_sha256': file_sha(__file__)}
    def stopped(*_):
        raise TimeoutError('Diagnostic supervisor terminated')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        for step in (3000, 2750, 2500):
            command = [sys.executable, str(Path(__file__).resolve()), '--root', str(args.root),
                '--output', args.output, '--evidence', str(args.evidence), '--step', str(step)]
            remaining = deadline - time.monotonic() - 90
            if remaining <= 0:
                raise TimeoutError('Diagnostic deadline')
            with (args.evidence / f'diagnostic-{step}.log').open('wb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=remaining)
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            status['steps'].append(dict(step=step, returncode=code))
            write(args.evidence / 'status.json', status)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(args.evidence / 'status.json', status)
        if args.evidence_prefix:
            subprocess.run(['gcloud', 'storage', 'rsync', str(args.evidence), args.evidence_prefix,
                '--recursive'], timeout=80, check=True)


if __name__ == '__main__':
    main()
