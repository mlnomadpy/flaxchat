"""Reprepare an authenticated mixture with global held-out text quarantine.

Runs data work only; use the resulting per-source folders with the trainer.
Existing inputs are immutable. All output is atomically committed together.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from flaxchat.embedding_data import IDENTITY_POLICY, row_identities
from flaxchat.encoder_data import file_hash
from scripts.prepare_yat_embedding_finetune import _raw_rows, _tokenize


def prepare_mixture(sources, output, tokenizer):
    if output.exists() or not sources:
        raise ValueError('New output and nonempty source registry required')
    manifests = {}
    parents, query_group = {}, {}
    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    dev_hashes, dev_queries, dev_groups = set(), set(), set()
    for name, folder in sources.items():
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['source'] != name or manifest['tokenizer_sha256'] != file_hash(tokenizer):
            raise ValueError('Source or tokenizer mismatch')
        for split in ('train', 'dev'):
            path = folder / f'{split}.jsonl'
            if file_hash(path) != manifest['raw_files'][f'{split}.jsonl']:
                raise ValueError('Raw input checksum mismatch')
        manifests[name] = manifest
        for split in ('train', 'dev'):
            for row in _raw_rows(folder / f'{split}.jsonl'):
                group = (name, row['group'])
                parents.setdefault(group, group)
                query = row_identities(row, name)['query']
                if query in query_group:
                    parents[root(group)] = root(query_group[query])
                else:
                    query_group[query] = group
        for row in _raw_rows(folder / 'dev.jsonl'):
            ids = row_identities(row, name)
            dev_hashes.update(ids.values())
            dev_queries.add(ids['query'])
            dev_groups.add((name, row['group']))
    held_components = {root(group) for group in dev_groups}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.mixture-', dir=output.parent))
    receipt = {'policy': IDENTITY_POLICY, 'relevance_policy': 'identical-query-and-positive-group-equivalence-closure-v1',
               'quarantined': {}, 'sources': {}}
    try:
        for name, folder in sources.items():
            target = staging / name
            target.mkdir()
            counts, rejected = Counter(), Counter()
            manifest = dict(manifests[name])
            for split in ('train', 'dev'):
                with (target / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
                    for row in _raw_rows(folder / f'{split}.jsonl'):
                        ids = row_identities(row, name)
                        if split == 'train' and (dev_hashes.intersection(ids.values()) or ids['query'] in dev_queries
                                                  or root((name, row['group'])) in held_components):
                            rejected['global_dev_overlap'] += 1
                            continue
                        row['modalities'] = row.get('modalities', {'query': 'text',
                            'positive': 'code' if name == 'code' else 'text',
                            'negative': 'code' if name == 'code' else 'text'})
                        row['text_hashes'] = [ids[field] for field in ('query', 'positive', 'negative') if row[field] is not None]
                        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                        counts[split] += 1
            if counts['train'] < 2 or counts['dev'] < 2:
                raise ValueError('Global quarantine leaves insufficient training or dev rows')
            tokens = _tokenize(target, tokenizer, manifest['query_length'], manifest['document_length'], manifest['pad_id'])
            manifest.update(format='flaxchat-yat-embedding-triplets-v2', identity_policy=IDENTITY_POLICY,
                            rows=dict(counts), tokenization=tokens, vocab_size=tokens['vocab_size'],
                            mixture_quarantine=dict(rejected), parent_manifest_sha256=file_hash(folder / 'manifest.json'),
                            preparation_sha256=file_hash(Path(__file__)),
                            identity_implementation_sha256=file_hash(Path(__file__).parents[1] / 'flaxchat/embedding_data.py'),
                            raw_files={f'{split}.jsonl': file_hash(target / f'{split}.jsonl') for split in ('train', 'dev')})
            (target / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
            receipt['quarantined'][name] = dict(rejected)
            receipt['sources'][name] = file_hash(target / 'manifest.json')
        receipt['global_train_dev_overlap'] = 0
        (staging / 'mixture.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        staging.rename(output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='append', required=True, help='NAME=DIR')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for value in args.data:
        name, separator, path = value.partition('=')
        if not separator or name in sources or not name or Path(name).name != name:
            raise ValueError('Invalid or duplicate source name')
        sources[name] = Path(path)
    print(json.dumps(prepare_mixture(sources, args.output, args.tokenizer), sort_keys=True))


if __name__ == '__main__':
    main()
