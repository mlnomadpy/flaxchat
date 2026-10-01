"""Prepare an immutable held-out STS development probe (never public test data).

Input JSONL fields: sentence1, sentence2, score, language (optional). Supply its
SHA256 and declare dev/validation provenance. HF provenance can additionally
be bound with --source-repo/--source-revision. No model inference is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from flaxchat.encoder_data import file_hash


def prepare(raw, raw_sha, tokenizer, encoder_config, output, *, source_split, length=128,
            source_repo=None, source_revision=None):
    import re
    from tokenizers import Tokenizer
    if (source_split not in ('dev', 'validation') or output.exists()
            or file_hash(raw) != raw_sha or not re.fullmatch('[0-9a-f]{64}', raw_sha)):
        raise ValueError('Invalid held-out source identity, split, or output')
    if bool(source_repo) != bool(source_revision) or source_revision and not re.fullmatch('[0-9a-f]{40}', source_revision):
        raise ValueError('HF STS provenance requires repository and commit revision')
    config = json.loads(encoder_config.read_text())
    if not 2 <= length <= config['max_position_embeddings']:
        raise ValueError('Invalid STS context length')
    tok = Tokenizer.from_file(str(tokenizer))
    if tok.get_vocab_size() != config['vocab_size'] or config['pad_token_id'] not in tok.get_vocab().values():
        raise ValueError('STS tokenizer vocabulary/padding differs from model')
    tok.no_padding(); tok.no_truncation()
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    scores = np.array([row['score'] for row in rows], np.float64)
    if len(rows) < 3 or scores.shape != (len(rows),) or not np.all(np.isfinite(scores)):
        raise ValueError('STS requires at least three finite scalar scores')
    if any(not isinstance(row[field], str) or not row[field].strip() for row in rows for field in ('sentence1', 'sentence2')):
        raise ValueError('STS text must be nonempty')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.sts-', dir=output.parent))
    truncated = {}
    try:
        shutil.copyfile(raw, staging / 'raw.jsonl')
        np.save(staging / 'scores.npy', scores, allow_pickle=False)
        for field, name in [('sentence1', 'query_tokens'), ('sentence2', 'positive_tokens')]:
            tokens = np.lib.format.open_memmap(staging / f'{name}.npy', mode='w+', dtype=np.int32, shape=(len(rows), length))
            tokens[:] = config['pad_token_id']
            truncated[name] = 0
            for start in range(0, len(rows), 1024):
                for offset, item in enumerate(tok.encode_batch([row[field] for row in rows[start:start + 1024]])):
                    ids = item.ids
                    if not ids:
                        raise ValueError('STS tokenizer produced empty sequence')
                    if len(ids) > length:
                        ids = ids[:length - 1] + ids[-1:]
                        truncated[name] += 1
                    tokens[start + offset, :len(ids)] = ids
            tokens.flush()
            del tokens
        files = ('scores.npy', 'query_tokens.npy', 'positive_tokens.npy', 'raw.jsonl')
        manifest = {'format': 'flaxchat-embedding-sts-dev-v1', 'source_split': source_split,
            'source_identity': {'sha256': raw_sha, **({'repo': source_repo, 'revision': source_revision} if source_repo else {})},
            'tokenizer_sha256': file_hash(tokenizer), 'vocab_size': config['vocab_size'], 'pad_id': config['pad_token_id'],
            'rows': len(rows), 'length': length, 'truncated': truncated,
            'truncation_policy': 'disable-inherited; terminal-token-preserving-v1',
            'preparation_sha256': file_hash(Path(__file__)),
            'files': {name: file_hash(staging / name) for name in files}}
        (staging / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--encoder-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-split', choices=('dev', 'validation'), required=True)
    parser.add_argument('--source-repo')
    parser.add_argument('--source-revision')
    parser.add_argument('--length', type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.input_sha256, args.tokenizer, args.encoder_config,
        args.output, source_split=args.source_split, length=args.length,
        source_repo=args.source_repo, source_revision=args.source_revision), sort_keys=True))


if __name__ == '__main__':
    main()
