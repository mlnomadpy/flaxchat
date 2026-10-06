"""Explicit, bounded new candidate identity after complete exact collision evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

from flaxchat.embedding_data import IDENTITY_POLICY, row_identities
from flaxchat.embedding_development_quarantine import load_candidate_exclusions

MAX_BYTES = 256 * 1024**2
MAX_FILE = 16 * 1024**2


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_FILE + 1)
    if len(raw) > MAX_FILE:
        raise ValueError('Candidate filtering evidence exceeds file bound')
    return raw


def encode(record):
    return (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()


def plan(candidate, scan_raw, details_raw, *, min_rows_per_config, deadline=None):
    deadline = deadline or time.monotonic()+120

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Candidate filter deadline exhausted')

    check()
    original_spec_raw = read(candidate / 'spec.json')
    original_receipt_raw = read(candidate / 'exclusions.json')
    spec = json.loads(original_spec_raw)
    if 'candidate_filter' in spec:
        raise ValueError('Nested candidate filtering is unsupported; prepare a new original pool')
    from scripts.prepare_representation_development import validate_spec
    validate_spec(spec)
    paths = [candidate / (source['name']+'__'+config.replace('/', '_')+'.jsonl')
             for source in spec['sources'] for config in source['configs']]
    metadata_bytes = len(scan_raw)+len(details_raw)+len(original_spec_raw)+len(original_receipt_raw)
    if sum(path.stat().st_size for path in paths) + metadata_bytes > MAX_BYTES:
        raise ValueError('Aggregate candidate filtering evidence exceeds byte bound')
    index = load_candidate_exclusions(candidate)
    receipt = json.loads(read(candidate / 'exclusions.json'))
    scan, details = json.loads(scan_raw), json.loads(details_raw)
    if (scan.get('format') != 'flaxchat-gcs-parent-raw-scan-v1' or
            details.get('format') != 'flaxchat-gcs-candidate-overlap-details-v1' or
            any(record.get('status') != 'complete' or record.get('coverage_complete') is not True or
                record.get('candidate_identity') != index.identity or
                record.get('scope') != 'supplied-known-contrastive-stage-raw-inputs-only' or
                record.get('identity_policy') != IDENTITY_POLICY for record in (scan, details)) or
            scan.get('candidate_exposure_checked') is not True or
            details.get('exact_text_identity_details_complete') is not True or
            details.get('details_truncated') is not False or
            details.get('matched_identity_kind') != 'exact-text-only' or
            scan.get('overlap_details', {}).get('sha256') != sha(details_raw)):
        raise ValueError('Complete authenticated candidate-bound collision evidence required')
    for key in ('sources', 'stage_metadata_sha256', 'object_receipt_sha256',
                'producer_policies_sha256', 'exact_aligned_overlap_rows'):
        if scan.get(key) != details.get(key):
            raise ValueError('Raw scan and collision details disagree')
    stages = scan.get('stage_metadata_sha256')
    if (not isinstance(stages, list) or not 1 <= len(stages) <= 64 or
            any(not isinstance(item, str) or not re.fullmatch('[0-9a-f]{64}', item) for item in stages) or
            len(stages) != len(set(stages)) or
            not isinstance(scan.get('object_receipt_sha256'), str) or
            not re.fullmatch('[0-9a-f]{64}', scan['object_receipt_sha256'])):
        raise ValueError('Authenticated historical stage and object identities required')
    sources = scan.get('sources')
    if not isinstance(sources, list) or not 1 <= len(sources) <= 256:
        raise ValueError('Complete bounded historical source inventory required')
    names = set()
    overlap_rows = 0
    for source in sources:
        if (not isinstance(source, dict) or not isinstance(source.get('name'), str) or
                not re.fullmatch(r'[A-Za-z0-9_-]+', source['name']) or source['name'] in names or
                type(source.get('checked_rows')) is not int or source['checked_rows'] < 1 or
                type(source.get('bytes')) is not int or source['bytes'] < 1 or
                any(not isinstance(source.get(key), str) or not re.fullmatch('[0-9a-f]{64}', source[key])
                    for key in ('manifest_sha256', 'train_raw_sha256')) or
                not re.fullmatch(r'gs://[^\s#]+/train\.jsonl#[1-9][0-9]*', source.get('uri', ''))):
            raise ValueError('Authenticated historical source coverage required')
        names.add(source['name'])
        kinds = source.get('overlap_kinds', {})
        if (not isinstance(kinds, dict) or set(kinds) - {'exact_text', 'aligned_group'} or
                any(type(value) is not int or value < 0 for value in kinds.values()) or
                type(source.get('overlap_rows')) is not int or
                not 0 <= source['overlap_rows'] <= source['checked_rows'] or
                source['overlap_rows'] != sum(kinds.values())):
            raise ValueError('Invalid historical collision counts')
        if kinds.get('aligned_group', 0):
            raise ValueError('Aligned group collisions cannot be resolved by exact-text-only evidence')
        overlap_rows += source['overlap_rows']
    if type(scan.get('exact_aligned_overlap_rows')) is not int or scan['exact_aligned_overlap_rows'] != overlap_rows:
        raise ValueError('Incomplete aggregate historical collision counts')
    matches = details.get('matched_text_sha256')
    cap = details.get('max_overlap_text_identities')
    if (type(cap) is not int or not 1 <= cap <= 100000 or not isinstance(matches, list) or
            len(matches) > cap or len(matches) != len(set(matches)) or
            any(not isinstance(item, str) or not re.fullmatch('[0-9a-f]{64}', item) or
                item not in index.text_hashes for item in matches) or
            bool(matches) != bool(overlap_rows)):
        raise ValueError('Complete collision SHA inventory required')
    matched = set(matches)
    original_rows = {}
    total_bytes = metadata_bytes
    rejected_groups = set()
    original_files = {}
    for name in receipt['sources']:
        check()
        raw = read(candidate / (name + '.jsonl'))
        total_bytes += len(raw)
        if total_bytes > MAX_BYTES:
            raise ValueError('Aggregate candidate filtering evidence exceeds byte bound')
        rows = [json.loads(line) for line in raw.splitlines()]
        original_rows[name] = rows
        original_files[name + '.jsonl'] = raw
        for row in rows:
            check()
            if matched & set(row_identities(row, name).values()):
                rejected_groups.add(row['group'])
    groups, texts, output_files = set(), set(), {}
    selected_receipt = json.loads(json.dumps(receipt))
    counts = {}
    retained_by_source = {
        name: [row for row in rows if row['group'] not in rejected_groups]
        for name, rows in original_rows.items()
    }
    deficits = {
        name: {'retained': len(rows), 'minimum': max(
            min_rows_per_config, 3 if receipt['sources'][name]['task'] == 'sts' else 2)}
        for name, rows in retained_by_source.items()
        if len(rows) < max(min_rows_per_config, 3 if receipt['sources'][name]['task'] == 'sts' else 2)
    }
    if deficits:
        raise ValueError('Filtered candidate lacks required rows per configuration: '
                         + json.dumps(deficits, sort_keys=True))
    for name, rows in original_rows.items():
        check()
        retained = retained_by_source[name]
        output_files[name + '.jsonl'] = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                               for row in retained).encode()
        selected_receipt['sources'][name].update(selected_rows=len(retained),
                                                raw_sha256=sha(output_files[name + '.jsonl']))
        counts[name] = {'original_rows': len(rows), 'rejected_rows': len(rows)-len(retained),
                        'retained_rows': len(retained)}
        for row in retained:
            check()
            groups.add(row['group'])
            texts.update(identity for field, identity in row_identities(row, name).items()
                         if row.get(field) is not None)
    if texts & matched:
        raise ValueError('Filtered candidate still contains exposed text')
    provenance = {'policy': 'exact-collision-rejection/new-candidate-v1',
                  'original_candidate_identity': index.identity, 'scan_sha256': sha(scan_raw),
                  'overlap_details_sha256': sha(details_raw), 'min_rows_per_config': min_rows_per_config,
                  'reservation_policy': 'retained-selected-rows-only',
                  'known_parent_scope': scan['scope'], 'exposure_clean_claim': False}
    spec['candidate_filter'] = provenance
    spec_raw = encode(spec)
    selected_receipt.update(spec_sha256=sha(spec_raw), excluded_groups=sorted(groups),
                            excluded_text_sha256=sorted(texts), parent_exposure_checked=False,
                            quarantine_applied=False, candidate_filter=provenance,
                            candidate_filter_counts=counts)
    output_files.update({'spec.json': spec_raw, 'exclusions.json': encode(selected_receipt)})
    if any(len(raw) > MAX_FILE for raw in output_files.values()):
        raise ValueError('Filtered candidate output exceeds file bound')
    original_files.update({name: read(candidate / name) for name in ('spec.json', 'exclusions.json')})
    if load_candidate_exclusions(candidate).identity != index.identity:
        raise ValueError('Original candidate changed during filtering')
    return output_files, original_files, provenance


def validate_filtered_candidate(directory, spec, receipt):
    provenance = spec.get('candidate_filter')
    if not isinstance(provenance, dict) or receipt.get('candidate_filter') != provenance:
        raise ValueError('Filtered candidate provenance mismatch')
    minimum = provenance.get('min_rows_per_config')
    if type(minimum) is not int or not 2 <= minimum <= 10000:
        raise ValueError('Finite filtered candidate coverage required')
    evidence = directory / 'filter-evidence'
    scan_raw, details_raw = read(evidence / 'scan.json'), read(evidence / 'overlap-details.json')
    if sha(scan_raw) != provenance.get('scan_sha256') or sha(details_raw) != provenance.get('overlap_details_sha256'):
        raise ValueError('Filtered candidate retained evidence hash mismatch')
    files, _, expected_provenance = plan(evidence / 'source-candidate', scan_raw, details_raw,
                                       min_rows_per_config=minimum)
    if expected_provenance != provenance or any(read(directory / name) != raw for name, raw in files.items()):
        raise ValueError('Filtered candidate does not replay from retained evidence')


def filter_candidate(candidate, scan_path, details_path, output, *, scan_sha256,
                     overlap_details_sha256, min_rows_per_config=64, timeout_seconds=120):
    candidate, scan_path, details_path, output = map(Path, (candidate, scan_path, details_path, output))
    if (output.exists() or output.is_symlink() or candidate.resolve() in output.resolve().parents or
            output.resolve() in candidate.resolve().parents or output.resolve() == candidate.resolve() or
            type(min_rows_per_config) is not int or not 2 <= min_rows_per_config <= 10000 or
            type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 600):
        raise ValueError('Fresh separate filtered output and finite coverage/deadline required')
    deadline = time.monotonic()+timeout_seconds
    scan_raw, details_raw = read(scan_path), read(details_path)
    if (sha(scan_raw) != scan_sha256 or sha(details_raw) != overlap_details_sha256):
        raise ValueError('Expected authenticated scan/detail SHA mismatch')
    files, original_files, provenance = plan(candidate, scan_raw, details_raw,
                                            min_rows_per_config=min_rows_per_config, deadline=deadline)
    if time.monotonic() >= deadline:
        raise TimeoutError('Candidate filter deadline exhausted')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.filtered-candidate-', dir=output.parent))
    try:
        for name, raw in files.items():
            (staging / name).write_bytes(raw)
        evidence = staging / 'filter-evidence'
        original = evidence / 'source-candidate'
        original.mkdir(parents=True)
        for name, raw in original_files.items():
            (original / name).write_bytes(raw)
        (evidence / 'scan.json').write_bytes(scan_raw)
        (evidence / 'overlap-details.json').write_bytes(details_raw)
        identity = load_candidate_exclusions(staging).identity
        if read(scan_path) != scan_raw or read(details_path) != details_raw:
            raise ValueError('Collision evidence changed during filtering')
        if load_candidate_exclusions(candidate).identity != provenance['original_candidate_identity']:
            raise ValueError('Original candidate changed during filtering')
        if time.monotonic() >= deadline:
            raise TimeoutError('Candidate filter deadline exhausted')
        staging.rename(output)
        return {'format': 'flaxchat-filtered-candidate-result-v1', 'candidate_identity': identity,
                'original_candidate_identity': provenance['original_candidate_identity'],
                'exposure_clean_claim': False, 'parent_exposure_checked': False}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'scan', 'overlap-details', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--scan-sha256', required=True)
    parser.add_argument('--overlap-details-sha256', required=True)
    parser.add_argument('--min-rows-per-config', type=int, default=64)
    parser.add_argument('--timeout-seconds', type=float, default=120)
    args = parser.parse_args()
    print(json.dumps(filter_candidate(args.candidate, args.scan, args.overlap_details, args.output,
        scan_sha256=args.scan_sha256, overlap_details_sha256=args.overlap_details_sha256,
        min_rows_per_config=args.min_rows_per_config, timeout_seconds=args.timeout_seconds), sort_keys=True))


if __name__ == '__main__':
    main()
