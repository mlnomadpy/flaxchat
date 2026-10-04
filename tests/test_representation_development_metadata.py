"""Candidate development exporter contracts with injected rows, no model/network."""
import json
from pathlib import Path
import pytest
from scripts.prepare_representation_development import validate_spec, convert, heldout, export


SPEC = Path(__file__).parents[1] / 'configs/data/representation-development-v1.json'


def test_pinned_candidate_excludes_final_test_and_benchmark_dev():
    spec = json.loads(SPEC.read_text())
    validate_spec(spec)
    for source in spec['sources']:
        original = source['split']
        for forbidden in ('test', 'dev'):
            source['split'] = forbidden
            with pytest.raises(ValueError, match='Final benchmark'):
                validate_spec(spec)
        source['split'] = original


def test_translated_query_id_is_reserved_across_languages():
    source = json.loads(SPEC.read_text())['sources'][0]
    english = convert(source, 'en', {'id': 42, 'query': 'English query', 'positive': 'English positive', 'negative': 'N'})
    arabic = convert(source, 'ar', {'id': 42, 'query': 'Arabic query', 'positive': 'Arabic positive', 'negative': 'AN'})
    assert english['group'] == arabic['group']
    assert heldout(english, 29, 5) == heldout(arabic, 29, 5)


def test_bounded_export_has_honest_quarantine_and_provenance(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec.update(max_scanned_rows_per_config=100, max_selected_rows_per_config=4)
    source = spec['sources'][0]
    source['configs'] = ['en', 'ar']
    spec['sources'] = [source]
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    requests = []
    def loader(repo, config, **kwargs):
        requests.append(kwargs)
        return ({'id': i, 'query': f'{config} query {i}', 'positive': f'{config} positive {i}',
                 'negative': f'{config} negative {i}'} for i in range(1000))
    receipt = export(path, tmp_path / 'out', loader=loader)
    assert all(x['streaming'] and x['split'] == 'train' and len(x['revision']) == 40 for x in requests)
    assert not receipt['quarantine_applied'] and not receipt['parent_exposure_checked'] and not receipt['prepared_arrays']
    assert all(x['scanned_rows'] == 100 and x['selected_rows'] == 4 for x in receipt['sources'].values())
    assert len(receipt['excluded_groups']) > 4
    assert len(receipt['excluded_text_sha256']) > 8
    assert (tmp_path / 'out/exclusions.json').exists()


def bounded_spec(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec.update(max_scanned_rows_per_config=20, max_selected_rows_per_config=4)
    spec['sources'] = [spec['sources'][0]]
    spec['sources'][0]['configs'] = ['en']
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    return path, spec


def test_export_does_not_fetch_an_extra_row_at_scan_boundary(tmp_path):
    path, spec = bounded_spec(tmp_path)
    reads = []
    def loader(*args, **kwargs):
        for i in range(21):
            if i == 20:
                raise AssertionError('Fetched beyond admitted scan boundary')
            reads.append(i)
            yield {'id': i, 'query': f'q {i}', 'positive': f'p {i}', 'negative': f'n {i}'}
    receipt = export(path, tmp_path / 'out', loader=loader)
    assert len(reads) == spec['max_scanned_rows_per_config']
    assert receipt['serialized_input_bytes'] > 0


def test_changed_spec_and_exhausted_bytes_leave_no_admitted_output(tmp_path):
    path, spec = bounded_spec(tmp_path)
    def changed_loader(*args, **kwargs):
        spec['seed'] += 1
        path.write_text(json.dumps(spec))
        return ({'id': i, 'query': f'q {i}', 'positive': f'p {i}', 'negative': f'n {i}'} for i in range(20))
    output = tmp_path / 'out'
    with pytest.raises(ValueError, match='changed during preparation'):
        export(path, output, loader=changed_loader)
    assert not output.exists() and not list(tmp_path.glob('.candidate-dev-*'))
    with pytest.raises(ValueError, match='input-byte budget'):
        export(path, output, max_input_bytes=1, loader=lambda *a, **kw: iter([
            {'id': 1, 'query': 'query', 'positive': 'positive', 'negative': 'negative'}]))
    assert not output.exists() and not list(tmp_path.glob('.candidate-dev-*'))


def test_aggregate_bounds_and_duplicate_configs_fail_before_loading(tmp_path):
    path, spec = bounded_spec(tmp_path)
    spec['sources'][0]['configs'] = ['en', 'en']
    with pytest.raises(ValueError, match='Unique bounded'):
        validate_spec(spec)
    spec['sources'][0]['configs'] = ['en', 'ar']
    spec['max_scanned_rows_per_config'] = 2000000
    with pytest.raises(ValueError, match='Aggregate'):
        validate_spec(spec)


def test_cooperative_deadline_removes_staged_evidence(tmp_path, monkeypatch):
    path, _ = bounded_spec(tmp_path)
    clock = iter([0., 2.])
    monkeypatch.setattr('scripts.prepare_representation_development.time.monotonic', lambda: next(clock))
    with pytest.raises(TimeoutError, match='deadline'):
        export(path, tmp_path / 'out', timeout_seconds=1,
               loader=lambda *a, **kw: iter([{'id': 1, 'query': 'q', 'positive': 'p', 'negative': 'n'}]))
    assert not (tmp_path / 'out').exists() and not list(tmp_path.glob('.candidate-dev-*'))


def native_miracl_spec():
    spec = json.loads(SPEC.read_text())
    spec.update(max_scanned_rows_per_config=100, max_selected_rows_per_config=4)
    spec['sources'] = [{
        'name': 'native_miracl_train', 'task': 'retrieval',
        'repo': 'sentence-transformers/miracl',
        'revision': '07e2b629250bf4185f4c87f640fac15949b8aa73',
        'configs': ['ar-triplet', 'fr-triplet'], 'split': 'train',
        'fields': {'query': 'anchor', 'positive': 'positive', 'negative': 'negative'},
        'group_policy': 'query-text',
        'config_language_map': {'ar-triplet': 'ar', 'fr-triplet': 'fr'},
    }]
    return spec


@pytest.mark.parametrize('mapping', [None, {}, {'ar-triplet': 'ar'},
    {'ar-triplet': 'ar', 'fr-triplet': 'fr', 'en-triplet': 'en'},
    {'ar-triplet': 'und', 'fr-triplet': 'fr'},
    {'ar-triplet': 'AR', 'fr-triplet': 'fr'},
    {'ar-triplet': 1, 'fr-triplet': 'fr'}])
def test_native_config_map_must_be_explicit_complete_and_human_language(mapping):
    spec = native_miracl_spec()
    if mapping is None:
        del spec['sources'][0]['config_language_map']
    else:
        spec['sources'][0]['config_language_map'] = mapping
    with pytest.raises(ValueError, match='config-to-human-language map'):
        validate_spec(spec)


def test_native_miracl_export_preserves_raw_text_and_has_no_invented_alignment(tmp_path):
    spec = native_miracl_spec()
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    requests = []
    def loader(repo, config, **kwargs):
        requests.append((repo, config, kwargs))
        return ({'anchor': f'  {config} raw query {i}\n',
                 'positive': f'{config} raw positive {i}\t',
                 'negative': f'{config} raw negative {i}'} for i in range(100))
    receipt = export(path, tmp_path / 'out', loader=loader)
    assert all(repo == 'sentence-transformers/miracl' and
               kwargs['revision'] == spec['sources'][0]['revision'] and
               kwargs['split'] == 'train' for repo, _, kwargs in requests)
    for config, language in spec['sources'][0]['config_language_map'].items():
        name = 'native_miracl_train__' + config
        rows = [json.loads(line) for line in (tmp_path / 'out' / (name + '.jsonl')).read_text().splitlines()]
        assert {row['language'] for row in rows} == {language}
        for row in rows:
            assert row['query'].startswith('  ') and row['query'].endswith('\n')
            assert row['positive'].endswith('\t') and config in row['negative']
            assert 'upstream_group' not in row
            from flaxchat.embedding_data import text_identity
            assert row['group'] == 'native_miracl_train:' + text_identity(row['query'])
        metadata = receipt['sources'][name]
        assert metadata['human_language'] == language
        assert metadata['human_language_provenance'] == 'pinned-spec-config-language-map'
        assert metadata['upstream_text_fields'] == {'query': 'anchor', 'positive': 'positive', 'negative': 'negative'}
    assert not receipt['parent_exposure_checked'] and not receipt['quarantine_applied']
