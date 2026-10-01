"""Bounded streaming export of pinned candidate development rows, without models.

Exports raw probes and an exclusion receipt, not training arrays. Quarantine
these probes across future training and check parent exposure before tuning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import itertools
import time
from flaxchat.embedding_data import text_identity
from flaxchat.encoder_data import file_hash

MAX_SPEC_BYTES = 1024 * 1024
MAX_CONFIGS = 256
MAX_SCAN_ROWS = 2000000
MAX_ROW_BYTES = 4 * 1024 * 1024


def read_spec(path):
    with path.open('rb') as handle:
        raw = handle.read(MAX_SPEC_BYTES + 1)
    if len(raw) > MAX_SPEC_BYTES:
        raise ValueError('Development specification exceeds size bound')
    return raw


def validate_spec(spec):
    if spec.get('format') != 'flaxchat-development-candidate-v1' or not spec.get('sources'):
        raise ValueError('Invalid development specification')
    for key in ('max_scanned_rows_per_config', 'max_selected_rows_per_config', 'heldout_hash_modulus'):
        if type(spec.get(key)) is not int or spec[key] < 2:
            raise ValueError('Finite positive development scan/selection bounds required')
    if (spec['max_scanned_rows_per_config'] > MAX_SCAN_ROWS or
            spec['max_selected_rows_per_config'] > 10000 or
            spec['max_selected_rows_per_config'] > spec['max_scanned_rows_per_config']):
        raise ValueError('Development scan/selection exceeds supported bounds')
    if type(spec.get('seed')) is not int or spec['seed'] < 0:
        raise ValueError('Invalid development seed')
    names = set()
    total_configs = 0
    for source in spec['sources']:
        if source['name'] in names or not re.fullmatch('[A-Za-z0-9_-]+', source['name']):
            raise ValueError('Duplicate or invalid development source name')
        names.add(source['name'])
        if not re.fullmatch('[0-9a-f]{40}', source.get('revision', '')):
            raise ValueError('Immutable source revision required')
        if source.get('task') not in ('retrieval', 'bitext', 'code', 'sts'):
            raise ValueError('Unsupported development task')
        if source.get('split') != ('validation' if source['task'] == 'sts' else 'train'):
            raise ValueError('Final benchmark splits cannot become tuning probes')
        if not source.get('configs') or source.get('group_policy') not in (
                'aligned-id-across-configs', 'english-text-across-configs', 'query-text', 'pair-text'):
            raise ValueError('Explicit configs and quarantine group policy required')
        configs = source['configs']
        if (not isinstance(configs, list) or any(not isinstance(config, str) or not config or
                len(config) > 128 for config in configs) or len(set(configs)) != len(configs)):
            raise ValueError('Unique bounded development configurations required')
        if not isinstance(source.get('repo'), str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', source['repo']):
            raise ValueError('Explicit dataset repository required')
        language_map = source.get('config_language_map')
        if (source['task'] == 'retrieval' and language_map is None and
                any(not re.fullmatch(r'[a-z]{2,3}', config) for config in configs)):
            raise ValueError('Retrieval configuration requires explicit config-to-human-language map')
        if language_map is not None:
            if (not isinstance(language_map, dict) or set(language_map) != set(configs) or
                    any(not isinstance(language, str) or
                        not re.fullmatch(r'[a-z]{2,3}(?:-[a-z0-9]{2,8})*', language) or
                        language == 'und' for language in language_map.values())):
                raise ValueError('Complete explicit config-to-human-language map required')
        total_configs += len(configs)
    if total_configs > MAX_CONFIGS or total_configs * spec['max_scanned_rows_per_config'] > MAX_SCAN_ROWS:
        raise ValueError('Aggregate development scan exceeds supported bounds')


def convert(source, config, original):
    fields = source['fields']
    query, positive = original[fields['query']], original[fields['positive']]
    if any(not isinstance(value, str) or not value.strip() for value in (query, positive)):
        raise ValueError('Empty development pair')
    policy = source['group_policy']
    if policy == 'aligned-id-across-configs':
        group = str(original[fields['group']])
    elif policy == 'pair-text':
        group = text_identity(query) + ':' + text_identity(positive)
    else:
        group = text_identity(query)
    row = {'query': query, 'positive': positive,
           'negative': original[fields['negative']] if fields.get('negative') else None,
           'group': f"{source['name']}:{group}",
           'language': source['config_language_map'][config] if 'config_language_map' in source else (
               config if source['task'] == 'retrieval' else (config[3:] if source['task'] == 'bitext' else 'en')),
           'modalities': {'query': 'text', 'positive': 'code' if source['task'] == 'code' else 'text'}}
    if source['task'] == 'code':
        row.update(programming_language=source.get('programming_language', 'unknown'),
                   programming_language_provenance='upstream-column-unavailable')
    if source['task'] == 'sts':
        score = float(original[fields['score']])
        if not source['score_range'][0] <= score <= source['score_range'][1]:
            raise ValueError('STS score outside pinned source range')
        row.update(sentence1=query, sentence2=positive, score=score)
    return row


def heldout(row, seed, modulus):
    digest = hashlib.sha256(f"{seed}:{row['group']}".encode()).hexdigest()
    return int(digest, 16) % modulus == 0


def export(spec_path, output, *, loader=None, timeout_seconds=1800, max_input_bytes=512 * 1024 * 1024):
    if (type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 3600 or
            type(max_input_bytes) is not int or not 1 <= max_input_bytes <= 4 * 1024 * 1024 * 1024):
        raise ValueError('Finite development deadline and input-byte budget required')
    deadline = time.monotonic() + timeout_seconds
    spec_raw = read_spec(spec_path)
    spec = json.loads(spec_raw)
    validate_spec(spec)
    if output.exists():
        raise ValueError('Development output must be new')
    if loader is None:
        from datasets import load_dataset
        loader = load_dataset
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.candidate-dev-', dir=output.parent))
    receipt = {'format': 'flaxchat-development-exclusion-v1', 'spec_sha256': hashlib.sha256(spec_raw).hexdigest(),
               'prepared_arrays': False, 'quarantine_applied': False, 'parent_exposure_checked': False,
               'sources': {}, 'excluded_groups': [], 'excluded_text_sha256': [],
               'preparation_limits': {'timeout_seconds': timeout_seconds, 'max_input_bytes': max_input_bytes,
                                      'max_row_bytes': MAX_ROW_BYTES}}
    groups, texts = set(), set()
    total_input_bytes = 0
    try:
        for source in spec['sources']:
            for config in source['configs']:
                records = loader(source['repo'], config, split=source['split'], revision=source['revision'], streaming=True)
                selected, scanned = [], 0
                for original in itertools.islice(records, spec['max_scanned_rows_per_config']):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Development preparation deadline exhausted')
                    original_bytes = len(json.dumps(original, ensure_ascii=False).encode())
                    total_input_bytes += original_bytes
                    if original_bytes > MAX_ROW_BYTES or total_input_bytes > max_input_bytes:
                        raise ValueError('Development input-byte budget exhausted')
                    scanned += 1
                    row = convert(source, config, original)
                    if source['task'] != 'sts' and not heldout(row, spec['seed'], spec['heldout_hash_modulus']):
                        continue
                    if len(selected) < spec['max_selected_rows_per_config']:
                        selected.append(row)
                    # Reserve every matching scanned group, including capped-out
                    # ones, so translated/aligned groups cannot enter training.
                    groups.add(row['group'])
                    for field in ('query', 'positive', 'negative'):
                        if row.get(field) is not None:
                            texts.add(text_identity(row[field], row['modalities'].get(field, 'text')))
                if len(selected) < (3 if source['task'] == 'sts' else 2):
                    raise ValueError(f'Insufficient candidate probe rows: {source["name"]}/{config}')
                name = source['name'] + '__' + config.replace('/', '_')
                path = staging / f'{name}.jsonl'
                raw = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in selected).encode()
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError('Development raw evidence exceeds admission size bound')
                path.write_bytes(raw)
                receipt['sources'][name] = {'repo': source['repo'], 'revision': source['revision'],
                    'config': config, 'source_split': source['split'], 'task': source['task'],
                    'scanned_rows': scanned, 'selected_rows': len(selected), 'raw_sha256': file_hash(path)}
                if 'config_language_map' in source:
                    receipt['sources'][name].update(
                        human_language=source['config_language_map'][config],
                        human_language_provenance='pinned-spec-config-language-map',
                        upstream_text_fields={key: source['fields'][key] for key in (
                            'query', 'positive', 'negative') if key in source['fields']})
        if time.monotonic() >= deadline:
            raise TimeoutError('Development preparation deadline exhausted')
        if read_spec(spec_path) != spec_raw:
            raise ValueError('Development specification changed during preparation')
        receipt.update(excluded_groups=sorted(groups), excluded_text_sha256=sorted(texts),
                       serialized_input_bytes=total_input_bytes)
        exclusion_raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
        if len(exclusion_raw) > 16 * 1024 * 1024:
            raise ValueError('Development exclusion evidence exceeds admission size bound')
        (staging / 'exclusions.json').write_bytes(exclusion_raw)
        (staging / 'spec.json').write_bytes(spec_raw)
        if time.monotonic() >= deadline:
            raise TimeoutError('Development preparation deadline exhausted')
        staging.rename(output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    parser.add_argument('--max-input-bytes', type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(export(args.spec, args.output, timeout_seconds=args.timeout_seconds,
                            max_input_bytes=args.max_input_bytes), sort_keys=True))


if __name__ == '__main__':
    main()
