"""Prepare bounded JSONL text into encoder rows using a local HF tokenizer.

One document per row; long documents are truncated, never joined across rows.
Use a pinned dataset export. This command does not download a corpus or start TPUs.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from tokenizers import Tokenizer

from scripts.train_encoder import file_hash


def prepare(source, tokenizer_path, output, *, sequence_length, max_rows, pad_token_id, mask_token_id=4):
    if sequence_length <= 0 or max_rows <= 0:
        raise ValueError('sequence-length and max-rows must be positive')
    source, tokenizer_path, output = Path(source), Path(tokenizer_path), Path(output)
    if output.exists():
        raise ValueError('Output directory already exists')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if not 0 <= pad_token_id < tokenizer.get_vocab_size():
        raise ValueError('Invalid pad token ID')
    if not 0 <= mask_token_id < tokenizer.get_vocab_size() or mask_token_id == pad_token_id:
        raise ValueError('Invalid or identical mask token ID')
    tokenizer.enable_truncation(max_length=sequence_length)
    tokenizer.no_padding()
    raw = json.loads(tokenizer_path.read_text())
    special_ids = sorted({pad_token_id, mask_token_id} | {t['id'] for t in raw.get('added_tokens', []) if t.get('special')})
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.encoder-', dir=output.parent))
    try:
        staging = np.lib.format.open_memmap(temp / 'staging.npy', mode='w+', dtype=np.int32,
                                           shape=(max_rows, sequence_length))
        count = 0
        with source.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                text = json.loads(line)['text']
                if not isinstance(text, str):
                    raise ValueError('JSONL text must be a string')
                ids = tokenizer.encode(text).ids
                if not ids or all(t in special_ids for t in ids):
                    continue
                staging[count] = pad_token_id
                staging[count, :len(ids)] = ids
                count += 1
                if count == max_rows:
                    break
        if not count:
            raise ValueError('No eligible documents')
        final = np.lib.format.open_memmap(temp / 'tokens.npy', mode='w+', dtype=np.int32,
                                         shape=(count, sequence_length))
        for start in range(0, count, 1024):
            final[start:start + 1024] = staging[start:min(start + 1024, count)]
        final.flush()
        del final, staging
        (temp / 'staging.npy').unlink()
        manifest = dict(format='flaxchat-encoder-rows-v1', rows=count, sequence_length=sequence_length,
                        vocab_size=tokenizer.get_vocab_size(), pad_token_id=pad_token_id, mask_token_id=mask_token_id,
                        special_token_ids=special_ids, tokenizer_sha256=file_hash(tokenizer_path),
                        source_sha256=file_hash(source), tokens_sha256=file_hash(temp / 'tokens.npy'),
                        policy='one-document-per-row; truncate; no shifting')
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        temp.rename(output)
        return manifest
    finally:
        if temp.exists():
            shutil.rmtree(temp)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='JSONL with text fields')
    p.add_argument('--tokenizer', required=True, help='Local tokenizer.json')
    p.add_argument('--output', required=True)
    p.add_argument('--sequence-length', type=int, default=512)
    p.add_argument('--max-rows', type=int, default=10000)
    p.add_argument('--pad-token-id', type=int, default=0)
    p.add_argument('--mask-token-id', type=int, default=4, help='Model mask token, even if tokenizer marks it non-special')
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.tokenizer, a.output, sequence_length=a.sequence_length,
                            max_rows=a.max_rows, pad_token_id=a.pad_token_id, mask_token_id=a.mask_token_id), indent=2))
