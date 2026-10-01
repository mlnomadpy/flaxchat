"""Authenticate candidate development exclusions; no model or backend calls."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from flaxchat.embedding_data import row_identities


def _read(path, limit=16 * 1024 * 1024):
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Development evidence exceeds bounded size')
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass
class DevelopmentExclusions:
    identity: dict
    text_hashes: frozenset
    aligned_groups: dict

    def matches(self, row, source_identity, source_name):
        hashes = row_identities(row, source_name)
        if any(hashes[field] in self.text_hashes for field in hashes):
            return 'exact_text'
        key = (source_identity.get('repo'), source_identity.get('revision'))
        groups = self.aligned_groups.get(key, frozenset())
        if groups:
            # This column must be retained by every aligned upstream adapter.
            # An opaque language-prefixed local group cannot prove exclusion.
            if row.get('upstream_group') is None:
                raise ValueError('Aligned development quarantine requires upstream_group')
            if str(row['upstream_group']) in groups:
                return 'aligned_group'
        return None


def load_candidate_exclusions(directory):
    """Bind exclusions to the pinned spec and every selected raw probe file.

    This proves exact exclusion identity, not parent cleanliness. Capped-out
    reserved groups remain included. Recheck files after reading to reject
    a moving input namespace.
    """
    directory = Path(directory)
    from scripts.prepare_representation_development import validate_spec
    spec_raw = _read(directory / 'spec.json')
    receipt_raw = _read(directory / 'exclusions.json')
    spec, receipt = json.loads(spec_raw), json.loads(receipt_raw)
    validate_spec(spec)
    if receipt.get('format') != 'flaxchat-development-exclusion-v1' or receipt.get('spec_sha256') != _sha(spec_raw):
        raise ValueError('Development exclusion/spec identity mismatch')
    hashes = receipt.get('excluded_text_sha256')
    groups = receipt.get('excluded_groups')
    if (not isinstance(hashes, list) or not hashes or
            any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes) or
            len(set(hashes)) != len(hashes) or not isinstance(groups, list) or not groups or
            any(not isinstance(g, str) or not g for g in groups) or len(set(groups)) != len(groups)):
        raise ValueError('Invalid development exclusion index')
    records = receipt.get('sources')
    expected = {source['name'] + '__' + config.replace('/', '_'): (source, config)
                for source in spec['sources'] for config in source['configs']}
    if len(expected) != sum(len(source['configs']) for source in spec['sources']):
        raise ValueError('Ambiguous development source/config filenames')
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError('Incomplete development source/config inventory')
    raw_hashes = {}
    aligned = {}
    hash_set, group_set = set(hashes), set(groups)
    for name, (source, config) in expected.items():
        metadata = records[name]
        if not isinstance(metadata, dict):
            raise ValueError('Invalid development source metadata')
        if any(metadata.get(k) != value for k, value in (
                ('repo', source['repo']), ('revision', source['revision']),
                ('config', config), ('source_split', source['split']), ('task', source['task']))):
            raise ValueError('Development source identity mismatch')
        mapped_language = source.get('config_language_map', {}).get(config)
        if mapped_language is not None:
            expected_fields = {key: source['fields'][key] for key in
                               ('query', 'positive', 'negative') if key in source['fields']}
            if (metadata.get('human_language') != mapped_language or
                    metadata.get('human_language_provenance') != 'pinned-spec-config-language-map' or
                    metadata.get('upstream_text_fields') != expected_fields):
                raise ValueError('Development mapped-language provenance mismatch')
        selected, scanned = metadata.get('selected_rows'), metadata.get('scanned_rows')
        if (type(selected) is not int or type(scanned) is not int or
                not 2 <= selected <= min(scanned, spec['max_selected_rows_per_config']) or
                scanned > spec['max_scanned_rows_per_config']):
            raise ValueError('Invalid bounded development row inventory')
        path = directory / (name + '.jsonl')
        raw = _read(path)
        if _sha(raw) != metadata.get('raw_sha256'):
            raise ValueError('Development raw checksum mismatch')
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != selected:
            raise ValueError('Development selected-row inventory mismatch')
        for row in rows:
            if mapped_language is not None and row.get('language') != mapped_language:
                raise ValueError('Development row language differs from pinned configuration map')
            identities = row_identities(row, source['name'])
            if row.get('group') not in group_set or any(identities[field] not in hash_set for field in identities if row.get(field) is not None):
                raise ValueError('Selected development probe missing from exclusion index')
        raw_hashes[path.name] = _sha(raw)
        if source['group_policy'] == 'aligned-id-across-configs':
            prefix = source['name'] + ':'
            aligned.setdefault((source['repo'], source['revision']), set()).update(
                group[len(prefix):] for group in groups if group.startswith(prefix))
        if _sha(_read(path)) != raw_hashes[path.name]:
            raise ValueError('Development raw input changed during admission')
    prefixes = tuple(source['name'] + ':' for source in spec['sources'])
    if any(not group.startswith(prefixes) for group in groups):
        raise ValueError('Unknown development group provenance')
    if _read(directory / 'spec.json') != spec_raw or _read(directory / 'exclusions.json') != receipt_raw:
        raise ValueError('Development exclusion input changed during admission')
    if 'candidate_filter' in spec or 'candidate_filter' in receipt:
        from scripts.filter_representation_development import validate_filtered_candidate
        validate_filtered_candidate(directory, spec, receipt)
    return DevelopmentExclusions(
        identity={'policy': 'candidate-development-quarantine-v1', 'spec_sha256': _sha(spec_raw),
                  'receipt_sha256': _sha(receipt_raw), 'raw_hashes': raw_hashes},
        text_hashes=frozenset(hashes),
        aligned_groups={key: frozenset(value) for key, value in aligned.items()})
