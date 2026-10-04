"""Actual fake-stream collision receipts and filter replay; no models/network."""
import hashlib
import json

import pytest

from flaxchat.embedding_development_quarantine import load_candidate_exclusions
from scripts.filter_representation_development import filter_candidate
from tests import test_gcs_parent_exposure_metadata as gcs_fixtures


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collision_case(root):
    helper = gcs_fixtures.GCSExposureScanTests()
    _, independent, _, objects, cli = helper.setup_case(root)
    candidate = independent / 'candidate'
    probe = next(candidate.glob('*.jsonl'))
    exposed = json.loads(probe.read_text().splitlines()[0])
    raw = root / 'historical-data/train.jsonl'
    rows = raw.read_text().splitlines()
    rows[0] = json.dumps(exposed)
    raw.write_text('\n'.join(rows)+'\n')
    manifest_path = root / 'historical-data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['raw_files']['train.jsonl'] = digest(raw)
    manifest_path.write_text(json.dumps(manifest))
    (root / 'historical-checkpoint-metadata.json').write_text(json.dumps({
        'resolved_config': {'data_manifests': {'pairs': digest(manifest_path)}}}))
    object_data = json.loads(objects.read_text())
    object_data['objects'][0]['bytes'] = raw.stat().st_size
    objects.write_text(json.dumps(object_data))
    details = root / 'details.json'
    helper.run_scan(root, objects, cli, development_exclusions=candidate,
                    overlap_details_output=details)
    return candidate, root / 'scan.json', details, exposed


def run_filter(root, candidate, scan, details, **kwargs):
    return filter_candidate(candidate, scan, details, root / 'filtered',
        scan_sha256=digest(scan), overlap_details_sha256=digest(details),
        min_rows_per_config=2, **kwargs)


def test_collision_filter_creates_new_identity_and_propagates_group_rejection(tmp_path):
    candidate, scan, details, exposed = collision_case(tmp_path)
    original = load_candidate_exclusions(candidate).identity
    result = run_filter(tmp_path, candidate, scan, details)
    filtered = tmp_path / 'filtered'
    assert result['original_candidate_identity'] == original
    assert result['candidate_identity'] != original
    assert load_candidate_exclusions(candidate).identity == original
    assert load_candidate_exclusions(filtered).identity == result['candidate_identity']
    assert not result['exposure_clean_claim'] and not result['parent_exposure_checked']
    receipt = json.loads((filtered / 'exclusions.json').read_text())
    assert exposed['group'] not in receipt['excluded_groups']
    # The aligned fixture uses the same group across en/ar. Both selected
    # members are rejected, even though only one text appeared in parent rows.
    assert all(count['rejected_rows'] == 1 for count in receipt['candidate_filter_counts'].values())
    assert all(count['retained_rows'] == 3 for count in receipt['candidate_filter_counts'].values())
    assert (filtered / 'filter-evidence/source-candidate/spec.json').read_bytes() == (candidate / 'spec.json').read_bytes()
    assert receipt['candidate_filter']['reservation_policy'] == 'retained-selected-rows-only'


@pytest.mark.parametrize('mutation', ['truncated', 'incomplete', 'foreign', 'aligned', 'wrong-scope'])
def test_unusable_collision_receipts_fail_closed(tmp_path, mutation):
    candidate, scan, details, _ = collision_case(tmp_path)
    detail = json.loads(details.read_text())
    summary = json.loads(scan.read_text())
    if mutation == 'truncated':
        detail['details_truncated'] = True
    elif mutation == 'incomplete':
        detail['coverage_complete'] = False
    elif mutation == 'foreign':
        detail['candidate_identity']['spec_sha256'] = 'a'*64
    elif mutation == 'wrong-scope':
        detail['scope'] = 'everything-clean'
    else:
        summary['sources'][0]['overlap_kinds'] = {'aligned_group': summary['sources'][0]['overlap_rows']}
        detail['sources'] = summary['sources']
    details.write_text(json.dumps(detail))
    summary['overlap_details']['sha256'] = digest(details)
    scan.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        run_filter(tmp_path, candidate, scan, details)
    assert not (tmp_path / 'filtered').exists()


def test_filter_replays_retained_evidence_after_rehashed_output_edit(tmp_path):
    candidate, scan, details, _ = collision_case(tmp_path)
    run_filter(tmp_path, candidate, scan, details)
    output = tmp_path / 'filtered'
    raw = next(output.glob('*.jsonl'))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]['query'] = 'silently replaced candidate'
    raw.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows))
    from flaxchat.embedding_data import row_identities
    receipt = json.loads((output / 'exclusions.json').read_text())
    receipt['sources'][raw.stem]['raw_sha256'] = digest(raw)
    receipt['excluded_text_sha256'] = sorted(set(receipt['excluded_text_sha256']) |
                                           set(row_identities(rows[0], raw.stem).values()))
    (output / 'exclusions.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='replay'):
        load_candidate_exclusions(output)


def test_authenticated_hash_and_required_coverage_are_not_optional(tmp_path):
    candidate, scan, details, _ = collision_case(tmp_path)
    with pytest.raises(ValueError, match='SHA mismatch'):
        filter_candidate(candidate, scan, details, tmp_path / 'filtered',
            scan_sha256='a'*64, overlap_details_sha256=digest(details), min_rows_per_config=2)
    with pytest.raises(ValueError, match='required rows'):
        filter_candidate(candidate, scan, details, tmp_path / 'filtered',
            scan_sha256=digest(scan), overlap_details_sha256=digest(details), min_rows_per_config=4)
    assert not (tmp_path / 'filtered').exists()


def test_filter_deadline_fails_without_output(tmp_path, monkeypatch):
    candidate, scan, details, _ = collision_case(tmp_path)
    clock = iter([0., 2.])
    monkeypatch.setattr('scripts.filter_representation_development.time.monotonic', lambda: next(clock))
    with pytest.raises(TimeoutError):
        run_filter(tmp_path, candidate, scan, details, timeout_seconds=1)
    assert not (tmp_path / 'filtered').exists()
