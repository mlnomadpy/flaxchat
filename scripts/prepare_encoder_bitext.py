"""Prepare a checksum-pinned bitext shard without dropping or truncating pairs."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from tokenizers import Tokenizer
from flaxchat.encoder_data import file_hash


def prepare(source, tokenizer_path, output, *, source_sha256, dataset, revision,
            subset, sequence_length=512, max_pairs=10000, pad_token_id=0, mask_token_id=4):
    source, tokenizer_path, output = map(Path, (source, tokenizer_path, output))
    if (output.exists() or not all(isinstance(s, str) and s for s in (dataset, revision, subset))
            or type(sequence_length) is not int or sequence_length < 1
            or type(max_pairs) is not int or max_pairs < 1):
        raise ValueError('Fresh output, pinned identity and positive dimensions required')
    if file_hash(source) != source_sha256:
        raise ValueError('Source checksum mismatch')
    tokenizer_hash = file_hash(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    raw = json.loads(tokenizer_path.read_text())
    special = sorted({pad_token_id, mask_token_id} | {
        t['id'] for t in raw.get('added_tokens', []) if t.get('special')})
    if pad_token_id == mask_token_id or any(not 0 <= t < tokenizer.get_vocab_size() for t in special):
        raise ValueError('Invalid special token IDs')
    pairs = []
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if len(pairs) >= max_pairs:
                raise ValueError('Too many pairs; refusing silent sampling')
            if any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ('sentence1', 'sentence2', 'lang')):
                raise ValueError('Nonempty aligned texts and language required')
            pairs.append(row)
    if not pairs or len({p['lang'] for p in pairs}) != 1:
        raise ValueError('Exactly one nonempty language-pair shard required')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.bitext-', dir=output.parent))
    policy = 'one aligned sentence per row; special tokens; no truncation, filtering or deduplication'
    try:
        lengths = {}
        for name, column in [('queries', 'sentence1'), ('corpus', 'sentence2')]:
            directory = staging / name
            directory.mkdir()
            values = np.lib.format.open_memmap(directory/'tokens.npy', mode='w+', dtype=np.int32,
                                              shape=(len(pairs), sequence_length))
            values[:] = pad_token_id
            maximum = 0
            for i, pair in enumerate(pairs):
                ids = tokenizer.encode(pair[column]).ids
                if not ids or all(t in special for t in ids):
                    raise ValueError('Tokenization produced empty content')
                if len(ids) > sequence_length:
                    raise ValueError('Sentence exceeds sequence length; refusing truncation')
                values[i, :len(ids)] = ids
                maximum = max(maximum, len(ids))
            values.flush()
            del values
            lengths[name] = maximum
            manifest = dict(format='flaxchat-encoder-rows-v1', task='bitext_mining',
                dataset=dataset, revision=revision, split='test', subset=subset,
                rows=len(pairs), sequence_length=sequence_length, vocab_size=tokenizer.get_vocab_size(),
                pad_token_id=pad_token_id, mask_token_id=mask_token_id, special_token_ids=special,
                tokenizer_sha256=tokenizer_hash, source_sha256=source_sha256,
                tokens_sha256=file_hash(directory/'tokens.npy'), policy=policy)
            (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        query_ids = [f'q{i:06d}' for i in range(len(pairs))]
        document_ids = [f'd{i:06d}' for i in range(len(pairs))]
        task = dict(dataset=dataset, revision=revision, subset=subset, split='test',
                    tokenization_policy=policy, source_sha256=source_sha256,
                    query_ids=query_ids, document_ids=document_ids,
                    qrels={q: {d: 1} for q, d in zip(query_ids, document_ids, strict=True)},
                    languages={q: pairs[i]['lang'] for i, q in enumerate(query_ids)},
                    maximum_token_lengths=lengths, quality_qualified=False,
                    official_mteb_search_parity=False)
        (staging/'judgments.json').write_text(json.dumps(task, indent=2)+'\n')
        if file_hash(source) != source_sha256 or file_hash(tokenizer_path) != tokenizer_hash:
            raise ValueError('Inputs changed during preparation')
        staging.rename(output)
        return dict(pairs=len(pairs), maximum_token_lengths=lengths, subset=subset)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'tokenizer', 'output', 'source-sha256', 'dataset', 'revision', 'subset'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--sequence-length', type=int, default=512)
    args = vars(parser.parse_args())
    args['tokenizer_path'] = args.pop('tokenizer')
    print(json.dumps(prepare(**args)))


if __name__ == '__main__':
    main()
