"""Host-only admission contracts for embedding stages; never allocate a model."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_identity(root):
    """Bind every imported project module, including optional kernel choices."""
    root = Path(root)
    paths = sorted(root.glob('flaxchat/**/*.py')) + [root / 'scripts/train_yat_embedding_finetune.py']
    entries = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return {'files': entries, 'sha256': canonical_hash(entries)}


def validate_parent_metadata(metadata, encoder, tokenizer):
    parent = metadata.get('resolved_config', {}).get('encoder')
    if not isinstance(parent, dict) or parent != encoder:
        raise ValueError('Parent checkpoint encoder semantics differ; qualify an explicit migration before changing them')
    if metadata.get('tokenizer_identity') != tokenizer:
        raise ValueError('Parent checkpoint tokenizer differs')
    return {'metadata_sha256': canonical_hash(metadata), 'step': metadata['step'],
            'policy': 'model_weights_only; fresh_optimizer_schedule_and_data_cursor',
            'execution_overrides': {}}


def validate_cursor(state, *, horizon, stop, committed_step):
    values = []
    for name in ('completed_steps', 'optimizer_updates'):
        value = np.asarray(state.get(name))
        if value.shape != (1,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f'Invalid checkpoint {name}')
        values.append(int(value[0]))
    completed, updates = values
    if completed != updates or completed != committed_step or not 0 <= completed <= horizon:
        raise ValueError('Checkpoint step, cursor and optimizer updates disagree')
    if stop < completed:
        raise ValueError('Requested stop precedes checkpoint cursor')
    return completed


def validate_optimizer_cursor(step, committed_cursor):
    """Authenticate the restored scalar optimizer count, without model execution."""
    value = np.asarray(step)
    if (value.shape != () or not np.issubdtype(value.dtype, np.integer)
            or int(value) != committed_cursor):
        raise ValueError('Restored optimizer counter differs from committed recovery cursor')


def emit_completion_status(start, stop):
    """Emit the actual trainer's explicit no-work status after cursor admission."""
    if start != stop:
        return False
    print(json.dumps({'event': 'embedding_already_completed', 'step': start,
                      'requested_stop': stop}), flush=True)
    return True


def validate_arrays(arrays, manifest, encoder):
    if manifest.get('pad_id') != encoder['pad_token_id']:
        raise ValueError('Prepared padding differs from encoder')
    vocabulary = encoder['vocab_size']
    if manifest.get('vocab_size', manifest.get('tokenization', {}).get('vocab_size')) != vocabulary:
        raise ValueError('Prepared vocabulary differs or is missing; regenerate manifest')
    for split, rows in arrays.items():
        count = manifest['rows'][split]
        if type(count) is not int or count < 2:
            raise ValueError('At least two rows per split required')
        for name, value in rows.items():
            if name.endswith('_tokens'):
                length = manifest['query_length' if name == 'query_tokens' else 'document_length']
                if value.shape != (count, length) or value.dtype != np.int32 or not 2 <= length <= encoder['max_position_embeddings']:
                    raise ValueError(f'Invalid token schema: {split}/{name}')
                for start in range(0, count, 1024):
                    chunk = value[start:start + 1024]
                    if np.any((chunk < 0) | (chunk >= vocabulary)) or np.any(np.all(chunk == encoder['pad_token_id'], axis=1)):
                        raise ValueError(f'Invalid token values: {split}/{name}')
            elif name == 'negative_valid':
                if value.shape != (count,) or value.dtype != np.bool_:
                    raise ValueError('Invalid negative-valid flags')
            elif name.endswith('_ids'):
                if value.shape != (count,) or value.dtype != np.int32:
                    raise ValueError(f'Invalid identity schema: {name}')
                for start in range(0, count, 4096):
                    if np.any(value[start:start + 4096] < 0):
                        raise ValueError(f'Negative identity: {name}')
        policy = manifest.get('tokenization', {}).get('special_token_ids')
        if not isinstance(policy, list) or encoder['pad_token_id'] not in policy or any(type(v) is not int or not 0 <= v < vocabulary for v in policy):
            raise ValueError('Missing or invalid prepared special-token policy')
