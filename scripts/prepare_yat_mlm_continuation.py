"""Prepare pinned, fresh multilingual/code MLM windows on a cloud data VM.

No models are loaded. One tokenizer pass per selected shard, complete documents,
exact normalized document deduplication, and a hash-assigned held-out split.
Benchmark/semantic decontamination is NOT established by this preparation.
"""
import argparse
import hashlib
import json
import random
import shutil
import sqlite3
import time
import unicodedata
from pathlib import Path

from flaxchat.encoder_data import file_hash
from scripts.prepare_encoder_corpus import prepare

SOURCES = {
    'hq': ('epfml/FineWeb2-HQ', 'c0c06e94fd3a44ae9e802b2b0fc533817601eb5e'),
    'english': ('HuggingFaceFW/fineweb-edu', '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'),
    'missing': ('HuggingFaceFW/fineweb-2', 'af9c13333eb981300149d5ca60a8e9d659b276b9'),
    'code': ('codeparrot/github-code-clean', 'c48d40f9e70f0196f8236901ee35807f7d6c44c0'),
}
SHARES = {'hq': .45, 'english': .2, 'missing': .15, 'code': .2}
MISSING_LANGUAGES = ('hin_Deva', 'urd_Arab', 'swh_Latn', 'tha_Thai', 'kor_Hang', 'ben_Beng')
DATASET = 'flaxchat/yat-mlm-continuation'


def normalized_digest(text):
    value = ' '.join(unicodedata.normalize('NFKC', text).split())
    return hashlib.sha256(value.encode()).hexdigest()


def prior_coordinates(paths):
    excluded = set()
    for path in paths:
        obj = json.loads(Path(path).read_text())
        chunks = obj.get('chunks', [])
        if not isinstance(chunks, list):
            raise ValueError('Invalid prior recipe chunks')
        for item in chunks:
            excluded.add((item['dataset'], item['revision'], item['path'], item['row_group']))
    return excluded


def split_for_digest(digest):
    return 'validation' if int(digest[:16], 16) % 1000 == 0 else 'train'


def merge_shards(shards, output, recipe_revision, tokenizer):
    """Join prepared shards without re-tokenizing or holding token arrays in RAM."""
    import numpy as np
    if not shards:
        raise ValueError('No prepared shards for split')
    output = Path(output)
    output.mkdir()
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in shards]
    rows = sum(m['rows'] for m in manifests)
    length = manifests[0]['sequence_length']
    final = np.lib.format.open_memmap(output / 'tokens.npy', mode='w+', dtype=np.int32, shape=(rows, length))
    cursor, counts = 0, {}
    with (output / 'documents.jsonl').open('w') as stream:
        for path, manifest in zip(shards, manifests, strict=True):
            values = np.load(path / 'tokens.npy', mmap_mode='r')
            for start in range(0, len(values), 4096):
                final[cursor + start:cursor + start + len(values[start:start + 4096])] = values[start:start + 4096]
            with (path / 'documents.jsonl').open() as docs:
                for line in docs:
                    doc = json.loads(line)
                    doc['first_row'] += cursor
                    doc['end_row'] += cursor
                    stream.write(json.dumps(doc, ensure_ascii=False) + '\n')
            for language, stat in manifest['language_counts'].items():
                target = counts.setdefault(language, dict(documents=0, rows=0, nonpadding_tokens=0))
                for key in target:
                    target[key] += stat[key]
            cursor += len(values)
    final.flush()
    del final
    manifest = dict(manifests[0])
    manifest.update(revision=recipe_revision, rows=rows, documents=sum(m['documents'] for m in manifests),
                    nonpadding_tokens=sum(m['nonpadding_tokens'] for m in manifests), language_counts=counts,
                    source_sha256=recipe_revision, source_token_shares=SHARES,
                    tokenizer_sha256=file_hash(tokenizer), input_positions=rows * length,
                    tokens_sha256=file_hash(output / 'tokens.npy'), documents_sha256=file_hash(output / 'documents.jsonl'))
    manifest['nonpadding_fraction'] = manifest['nonpadding_tokens'] / manifest['input_positions']
    manifest['sampling'] = 'CoverageMixtureRows; expected source token shares; temperature 0.5 within each source'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def run(args):
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq
    root = Path(args.root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use an empty output root; partial preparation must not be mistaken for complete data')
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rng = random.Random(args.seed)
    excluded = prior_coordinates(args.prior_recipe)
    api, fs = HfApi(), HfFileSystem()
    pools = {}
    for source, (dataset, revision) in SOURCES.items():
        if source == 'missing':
            files = [entry.path for language in MISSING_LANGUAGES
                     for entry in api.list_repo_tree(dataset, path_in_repo=f'data/{language}/train',
                         repo_type='dataset', revision=revision, recursive=True)
                     if entry.path.endswith('.parquet')]
        else:
            files = api.list_repo_files(dataset, repo_type='dataset', revision=revision)
        files = sorted(p for p in files if p.endswith('.parquet') and
                       (source in ('hq', 'missing') or (source == 'english' and p.startswith('data/')) or
                        (source == 'code' and p.startswith('data/train-'))))
        if not files:
            raise ValueError(f'No pinned training files for {dataset}')
        for path in files:
            language = (path.split('/')[0] if source == 'hq' else
                        path.split('/')[1] if source == 'missing' else
                        'eng_Latn' if source == 'english' else 'code')
            pools.setdefault((source, language), []).append(path)
    for files in pools.values():
        rng.shuffle(files)
    counters = {key: 0 for key in pools}
    shards = {'train': [], 'validation': []}
    seen = sqlite3.connect(root / 'dedup.sqlite')
    seen.execute('CREATE TABLE seen (digest TEXT PRIMARY KEY)')
    recipe = dict(format='flaxchat-mlm-continuation-data-v1', source_token_shares=SHARES,
                  language_exponent=.5, seed=args.seed, sequence_length=args.sequence_length,
                  target_nonpadding_tokens=args.target_tokens, prior_recipe_sha256=[file_hash(p) for p in args.prior_recipe],
                  tokenizer_sha256=file_hash(args.tokenizer), sources=SOURCES, chunks=[],
                  split_policy='SHA256 normalized document modulo 1000; 0 held out; source train only',
                  decontamination='Normalized exact-document dedup only; no benchmark or semantic decontamination claim')
    revision = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    iteration = 0
    remaining_groups = {}
    while any(counters[k] < args.target_tokens * SHARES[k[0]] / sum(x[0] == k[0] for x in pools) for k in pools):
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Preparation deadline reached; partial output is not qualified')
        if shutil.disk_usage(root).free < args.min_free_gb * 1024 ** 3:
            raise RuntimeError('Insufficient disk reserve; partial output is not qualified')
        eligible = [k for k in pools if pools[k] and counters[k] < args.target_tokens * SHARES[k[0]] / sum(x[0] == k[0] for x in pools)]
        if not eligible:
            raise RuntimeError('Pinned corpus exhausted before requested token target')
        key = min(eligible, key=lambda k: counters[k] / (SHARES[k[0]] / sum(x[0] == k[0] for x in pools)))
        source, language = key
        dataset, pin = SOURCES[source]
        path = pools[key].pop()
        with fs.open(f'datasets/{dataset}@{pin}/{path}', 'rb', block_size=8 * 1024 * 1024) as remote:
            parquet = pq.ParquetFile(remote)
            group_key = (dataset, pin, path)
            if group_key not in remaining_groups:
                groups = [g for g in range(parquet.num_row_groups) if (*group_key, g) not in excluded]
                rng.shuffle(groups)
                remaining_groups[group_key] = groups
            groups = remaining_groups[group_key]
            wanted = {'text', 'code', 'content', 'language', 'license', 'licenses', 'repo_name', 'path', 'language_score'}
            columns = [name for name in parquet.schema_arrow.names if name in wanted]
            if not {'text', 'code', 'content'} & set(columns):
                raise ValueError(f'No supported text column in {dataset}/{path}')
            group = None
            while groups:
                candidate = groups.pop()
                metadata = parquet.metadata.row_group(candidate)
                decoded_bytes = sum(metadata.column(c).total_uncompressed_size for c in range(metadata.num_columns)
                                    if metadata.column(c).path_in_schema.split('.')[0] in columns)
                if decoded_bytes <= 128 * 1024 ** 2:
                    group = candidate
                    break
            if groups:
                # Revisit other groups after other shards, without exhausting small languages.
                pools[key].insert(0, path)
            if group is None:
                continue
            records = parquet.read_row_group(group, columns=columns).to_pylist()
        raw_paths = {split: root / f'raw-{iteration}-{split}.jsonl' for split in shards}
        streams = {split: path.open('w') for split, path in raw_paths.items()}
        digest, accepted = hashlib.sha256(), {'train': 0, 'validation': 0}
        try:
            for row_index, row in enumerate(records):
                text = row.get('text', row.get('code', row.get('content')))
                if not isinstance(text, str) or not 100 <= len(text) <= 200000 or not text.strip():
                    continue
                if source != 'code' and row.get('language_score') is not None and float(row['language_score']) < .8:
                    continue
                identity = normalized_digest(text)
                try:
                    seen.execute('INSERT INTO seen VALUES (?)', (identity,))
                except sqlite3.IntegrityError:
                    continue
                split = split_for_digest(identity)
                code_language = row.get('language') or 'unknown'
                doc = dict(dataset=DATASET, revision=revision, split=split, id=identity, text=text,
                           language_script=f'code:{code_language}' if source == 'code' else language,
                           source_group=source, source_dataset=dataset, source_revision=pin,
                           source_split='train', source_path=path, source_row_group=group, source_row=row_index,
                           license=row.get('license', row.get('licenses', 'upstream-dataset-card')),
                           repository=row.get('repo_name'), code_path=row.get('path'), text_sha256=identity)
                encoded = json.dumps(doc, ensure_ascii=False) + '\n'
                streams[split].write(encoded)
                digest.update(encoded.encode())
                accepted[split] += 1
        finally:
            for stream in streams.values():
                stream.close()
        seen.commit()
        receipt = dict(dataset=dataset, revision=pin, path=path, row_group=group, source_group=source,
                       language=language, selected_documents=accepted, selected_jsonl_sha256=digest.hexdigest())
        for split, raw_path in raw_paths.items():
            if accepted[split]:
                output = root / f'shard-{iteration}-{split}'
                manifest = prepare(raw_path, args.tokenizer, output, dataset=DATASET, revision=revision,
                                   split=split, sequence_length=args.sequence_length, max_rows=10000000)
                shards[split].append(output)
                if split == 'train':
                    counters[key] += manifest['nonpadding_tokens']
                receipt[split + '_nonpadding_tokens'] = manifest['nonpadding_tokens']
            raw_path.unlink()
        recipe['chunks'].append(receipt)
        (root / 'recipe.partial.json').write_text(json.dumps(recipe, indent=2) + '\n')
        print(json.dumps(dict(chunk=iteration, prepared_train_tokens=sum(counters.values()), source=source, language=language)), flush=True)
        iteration += 1
    seen.close()
    for split, paths in shards.items():
        merge_shards(paths, root / split, revision, args.tokenizer)
    recipe['actual_train_nonpadding_tokens'] = sum(counters.values())
    recipe['elapsed_seconds'] = time.monotonic() - started
    (root / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    for paths in shards.values():
        for path in paths:
            shutil.rmtree(path)
    (root / 'recipe.partial.json').unlink()
    (root / 'COMPLETE.json').write_text(json.dumps(dict(recipe_sha256=file_hash(root / 'recipe.json'),
        train_manifest_sha256=file_hash(root / 'train/manifest.json'), validation_manifest_sha256=file_hash(root / 'validation/manifest.json')), indent=2) + '\n')
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--target-tokens', type=int, default=1000000000)
    parser.add_argument('--max-seconds', type=int, default=10800)
    parser.add_argument('--min-free-gb', type=float, default=20)
    parser.add_argument('--sequence-length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=106)
    parser.add_argument('--prior-recipe', action='append', default=[])
    args = parser.parse_args()
    if args.target_tokens < 1 or args.max_seconds < 1 or args.min_free_gb <= 0 or args.seed < 0:
        parser.error('Require positive target/time/disk limits and nonnegative seed')
    run(args)


if __name__ == '__main__':
    main()
