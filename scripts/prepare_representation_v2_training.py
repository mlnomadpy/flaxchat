"""Bounded, model-free cloud preparation of the representation-v2 mixture.

Run under a provider-enforced VM lease and a cloud-side durable controller.
This runner does not allocate resources, launch a model, or upload artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

PORTABLE_SHA = '4447de1376a17cd0c5c98cf082d16b37391469d34fa5a6225e317ebff8df2559'
HELDOUT_SHA = 'fe7a75ee13ece1880cfd833c69a2aaaaa5003fa295895ffac6855b45a809b20d'
PAIR_SOURCE = 'contrastive_0eb6c456c95f51feb8ac5d1bf6d99841c792d8ebe8c037eddf65e9b52372c923'
SOURCE_SHUFFLE_SEED = 29
DEVELOPMENT_SPLIT_SEED = 41
LANGUAGES = ('ar', 'bn', 'fr', 'hi', 'ja', 'ru', 'sw', 'zh')
WEIGHTS = {'marco_replay': .45, 'code_replay': .15, 'bitext_replay': .20,
           **{'native_' + language: .025 for language in LANGUAGES}}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def extract(archive, expected, destination):
    if sha(archive) != expected:
        raise ValueError(f'Full archive hash mismatch: {archive}')
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not member.isfile() and not member.isdir():
                raise ValueError('Archive links and special files are not allowed')
            target = destination / member.name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive member outside destination')
        handle.extractall(destination, filter='data')


def historical_heldouts(history, supplement):
    """Resolve all six authenticated heldouts before consuming training rows.

    The portable exposure bundle contains training rows only. Supplemental
    heldouts therefore include retrieval/code sources as well as pair sources.
    Never silently omit a source when an input bundle is incomplete.
    """
    inventory, missing = [], []
    for folder in sorted(path for path in history.iterdir() if path.is_dir()):
        original = json.loads((folder / 'manifest.json').read_text())
        expected = (original['files']['dev.jsonl']['sha256']
                    if folder.name.startswith('contrastive_') else original['raw_files']['dev.jsonl'])
        candidates = [supplement / folder.name / 'dev.jsonl', folder / 'dev.jsonl']
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            missing.append(folder.name + '/dev.jsonl')
            continue
        for path in existing:
            if sha(path) != expected:
                raise ValueError(f'Historical held-out bytes differ from original manifest: {path}')
        inventory.append((existing[0], expected))
    if missing:
        raise ValueError('Incomplete historical heldout bundle; missing: ' + ', '.join(missing))
    return inventory


def replay_row(row, source):
    """Materialize the registry schema without inferring missing provenance.

    Historical CodeSearchNet rows do not declare a natural language. Missing
    language means undetermined, not English. Preserve original text/group
    bytes and any declared programming language; never mutate source artifacts.
    """
    result = dict(row)
    for field in ('query', 'positive', 'group'):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f'{source}: missing or invalid required replay field {field}')
    result.setdefault('negative', None)
    if result['negative'] is not None and (not isinstance(result['negative'], str) or not result['negative'].strip()):
        raise ValueError(f'{source}: invalid optional replay field negative')
    for field, missing in (('language', 'und'), ('programming_language', 'unknown')):
        value = result.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            result[field] = missing
        elif not isinstance(value, str):
            raise ValueError(f'{source}: invalid replay metadata {field}')
    if not result.get('programming_language_provenance'):
        result['programming_language_provenance'] = 'unavailable' if result['programming_language'] == 'unknown' else 'historical-row'
    if not isinstance(result['programming_language_provenance'], str):
        raise ValueError(f'{source}: invalid replay metadata programming_language_provenance')
    return result


def raw(args):
    from datasets import load_dataset
    from flaxchat.embedding_data import text_identity
    from flaxchat.embedding_development_quarantine import load_candidate_exclusions
    from flaxchat.embedding_historical_rows import historical_manifest_view, historical_row_view

    root = args.output
    portable, supplement = root / 'development', root / 'original-heldout'
    extract(args.portable_archive, PORTABLE_SHA, portable)
    extract(args.heldout_archive, args.heldout_sha256, supplement)
    history = portable / 'historical'
    tokenizer = args.tokenizer.resolve()
    if sha(tokenizer) != json.loads((portable / 'parent-files.json').read_text())['tokenizer.json']:
        raise ValueError('Tokenizer differs from authenticated parent')
    sources = sorted(path for path in history.iterdir() if path.is_dir())
    if len(sources) != 6 or not {PAIR_SOURCE, 'msmarco', 'code', 'miracl'} <= {p.name for p in sources}:
        raise ValueError('All six historical sources required')
    held = set()
    held_inventory = []
    for path, expected in historical_heldouts(history, supplement):
        count = 0
        for row in rows(path):
            count += 1
            for key in ('sentence1', 'sentence2', 'query', 'positive', 'negative'):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    held.update(text_identity(value, mode) for mode in ('text', 'code'))
        held_inventory.append({'path': str(path.relative_to(root)), 'sha256': expected, 'rows': count})
    write(root / 'heldout-inventory.json', {'files': held_inventory, 'never_training': True})
    exclusions = load_candidate_exclusions(portable / 'candidate')
    policies = json.loads((portable / 'metadata-inputs/producer-policies.json').read_text())
    rawdir = root / 'raw'
    rawdir.mkdir(exist_ok=False)
    registry = {}
    requests = [('marco_replay', 'msmarco'), ('code_replay', 'code'), ('bitext_replay', PAIR_SOURCE)]
    requests += [('native_' + language, None) for language in LANGUAGES]
    for name, historical in requests:
        print(json.dumps({'event': 'raw_source_started', 'source': name}), flush=True)
        if historical:
            folder = history / historical
            manifest_path = folder / 'manifest.json'
            manifest = historical_manifest_view(json.loads(manifest_path.read_text()), sha(manifest_path))
            identity = manifest['source_identity']
            source = folder / 'train.jsonl'
            if sha(source) != manifest['raw_files']['train.jsonl']:
                raise ValueError('Original training bytes differ')
            iterable = (historical_row_view(row, identity, historical, manifest,
                        producer_policy=policies.get(historical)) for row in rows(source))
        else:
            language = name.removeprefix('native_')
            identity = {'repo': 'sentence-transformers/miracl',
                        'revision': '07e2b629250bf4185f4c87f640fac15949b8aa73',
                        'config': language + '-triplet'}
            dataset = load_dataset(identity['repo'], identity['config'],
                                   revision=identity['revision'], split='train').shuffle(seed=SOURCE_SHUFFLE_SEED)
            iterable = (dict(query=row['anchor'], positive=row['positive'], negative=row['negative'],
                             group=row['anchor'], language=language) for row in dataset)
        target = rawdir / (name + '.jsonl')
        counts, languages = Counter(), Counter()
        with target.open('x') as stream:
            for row in iterable:
                counts['scanned'] += 1
                row = replay_row(row, name)
                values = [row.get(key) for key in ('query', 'positive', 'negative') if row.get(key)]
                if (any(text_identity(value, mode) in held for value in values for mode in ('text', 'code'))
                        or exclusions.matches(row, identity, name)):
                    counts['quarantined'] += 1
                    continue
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts['accepted'] += 1
                languages[str(row.get('language', 'und'))] += 1
        if counts['accepted'] < 256:
            raise ValueError(f'Insufficient source rows after quarantine: {name} {dict(counts)}')
        registry[name] = {**identity, 'jsonl': str(target.resolve()), 'sha256': sha(target),
            'split': 'train', 'fields': {'query': 'query', 'positive': 'positive', 'negative': 'negative',
                                      'group': 'group', 'language': 'language',
                                      'programming_language': 'programming_language',
                                      'programming_language_provenance': 'programming_language_provenance'},
            'modalities': {'query': 'text', 'positive': 'code' if name == 'code_replay' else 'text',
                          'negative': 'code' if name == 'code_replay' else 'text'},
            'programming_language': 'unknown', 'candidate_limit': None, 'train_limit': None,
            'candidate_identity': exclusions.identity, 'historical_heldout_sha256': sha(root / 'heldout-inventory.json'),
            'row_counts': dict(counts), 'human_language_counts': dict(languages)}
        write(root / 'registry-progress.json', registry)
        print(json.dumps({'event': 'raw_source_finished', 'source': name, 'counts': dict(counts)}), flush=True)
    write(root / 'registry.json', registry)
    write(root / 'weights.json', WEIGHTS)


def audit(args):
    from flaxchat.embedding_data import text_identity
    from flaxchat.embedding_development_quarantine import load_candidate_exclusions
    root = args.output
    held = set()
    inventory = json.loads((root / 'heldout-inventory.json').read_text())
    for item in inventory['files']:
        path = root / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Historical heldout changed')
        for row in rows(path):
            for key in ('sentence1', 'sentence2', 'query', 'positive', 'negative'):
                if isinstance(row.get(key), str) and row[key].strip():
                    held.update(text_identity(row[key], mode) for mode in ('text', 'code'))
    exclusions = load_candidate_exclusions(root / 'development/candidate')
    registry = json.loads((root / 'registry.json').read_text())
    receipt = {'model_execution': False, 'scope': 'all final training rows versus six original heldouts and independent candidate', 'sources': {}}
    for name, source in registry.items():
        folder = root / 'mixture' / name
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['source_identity'] != source:
            raise ValueError('Final mixture source identity changed')
        count = 0
        for split in ('train', 'dev'):
            path = folder / (split + '.jsonl')
            if sha(path) != manifest['raw_files'][path.name]:
                raise ValueError('Final mixture raw bytes changed')
            for row in rows(path):
                values = [row.get(key) for key in ('query', 'positive', 'negative') if row.get(key)]
                if (any(text_identity(value, mode) in held for value in values for mode in ('text', 'code'))
                        or exclusions.matches(row, source, name)):
                    raise ValueError(f'Final mixture retained historical/development overlap: {name}')
                if split == 'train':
                    count += 1
        receipt['sources'][name] = {'train_rows': count, 'manifest_sha256': sha(folder / 'manifest.json')}
    receipt.update(status='passed', exact_overlaps=0, candidate_identity=exclusions.identity)
    write(root / 'final-quarantine.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portable-archive', type=Path, required=True)
    parser.add_argument('--heldout-archive', type=Path, required=True)
    parser.add_argument('--heldout-sha256', default=HELDOUT_SHA, help='Authenticated full heldout archive SHA256')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--total-seconds', type=int, default=14400)
    parser.add_argument('--worker', choices=('raw', 'audit'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 60 <= args.total_seconds <= 21600:
        raise ValueError('A 60–21600 second finite data preparation deadline is required')
    if args.worker:
        (raw if args.worker == 'raw' else audit)(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.total_seconds
    report = {'status': 'running', 'model_execution': False, 'commands': [],
              'portable_sha256': PORTABLE_SHA, 'original_heldout_sha256': args.heldout_sha256,
              'source_sha256': sha(Path(__file__)), 'total_seconds': args.total_seconds,
              'source_shuffle_seed': SOURCE_SHUFFLE_SEED, 'development_split_seed': DEVELOPMENT_SPLIT_SEED}

    def run(command):
        remaining = deadline - time.monotonic() - 15
        if remaining <= 0:
            raise TimeoutError('Preparation deadline exhausted')
        item = {'argv': command, 'timeout_seconds': remaining}
        report['commands'].append(item)
        write(args.output / 'terminal-data.json', report)
        with (args.output / f'command-{len(report["commands"])}.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                item['returncode'] = process.wait(timeout=remaining)
                if item['returncode']:
                    raise RuntimeError(f"Data preparation command exited {item['returncode']}")
            except BaseException as error:
                item['error'] = f'{type(error).__name__}: {error}'
                raise
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                log.flush()
                item['returncode'] = process.returncode
                item['worker_log'] = Path(log.name).name
                if item.get('error'):
                    # Retain diagnostics for timeouts/signals as well as exits.
                    with open(log.name, 'rb') as failed_log:
                        failed_log.seek(0, os.SEEK_END)
                        failed_log.seek(max(0, failed_log.tell() - 16384))
                        diagnostic = failed_log.read().decode('utf-8', errors='replace')
                    item['worker_error_tail'] = diagnostic
                    print(diagnostic, file=sys.stderr, flush=True)
                write(args.output / 'terminal-data.json', report)

    worker = [sys.executable, '-m', 'scripts.prepare_representation_v2_training',
              '--portable-archive', str(args.portable_archive), '--heldout-archive', str(args.heldout_archive),
              '--heldout-sha256', args.heldout_sha256,
              '--tokenizer', str(args.tokenizer), '--output', str(args.output),
              '--total-seconds', str(args.total_seconds)]
    try:
        run([*worker, '--worker', 'raw'])
        registry = json.loads((args.output / 'registry.json').read_text())
        for name in registry:
            run([sys.executable, '-m', 'scripts.prepare_yat_embedding_finetune', '--source', name,
                 '--registry', str(args.output / 'registry.json'), '--output', str(args.output / 'prepared' / name),
                 '--tokenizer', str(args.tokenizer), '--development-exclusions', str(args.output / 'development/candidate'),
                 '--query-length', '128', '--document-length', '256', '--seed', str(DEVELOPMENT_SPLIT_SEED)])
        command = [sys.executable, '-m', 'scripts.qualify_yat_embedding_mixture',
                   '--output', str(args.output / 'mixture'), '--tokenizer', str(args.tokenizer)]
        for name in registry:
            command += ['--data', name + '=' + str(args.output / 'prepared' / name)]
        run(command)
        run([*worker, '--worker', 'audit'])
        report.update(status='passed', final_quarantine_sha256=sha(args.output / 'final-quarantine.json'),
                      training_launched=False, physical_tpu_qualified=False)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(args.output / 'terminal-data.json', report)


if __name__ == '__main__':
    main()
