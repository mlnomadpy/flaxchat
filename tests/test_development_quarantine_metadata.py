"""Real preparation exclusion flow with injected rows; no model/network."""
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from flaxchat.embedding_development_quarantine import load_candidate_exclusions
from scripts.prepare_representation_development import export
from scripts import prepare_yat_embedding_finetune as preparation


def candidate(tmp_path):
    spec = json.loads((Path(__file__).parents[1] / 'configs/data/representation-development-v1.json').read_text())
    spec.update(max_scanned_rows_per_config=100, max_selected_rows_per_config=4)
    spec['sources'] = [spec['sources'][0]]
    spec['sources'][0]['configs'] = ['en', 'ar']
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    output = tmp_path / 'candidate'
    export(path, output, loader=lambda repo, config, **kwargs: (
        {'id': i, 'query': f'{config} query {i}', 'positive': f'{config} positive {i}',
         'negative': f'{config} negative {i}'} for i in range(100)))
    return output


def test_aligned_group_blocks_unseen_language_and_requires_original_id(tmp_path):
    index = load_candidate_exclusions(candidate(tmp_path))
    upstream = preparation.SOURCES['miracl']
    reserved = next(iter(index.aligned_groups[(upstream['repo'], upstream['revision'])]))
    row = dict(query='unseen zh query', positive='unseen zh document', negative=None,
               upstream_group=reserved, group='different-language-local-group')
    assert index.matches(row, upstream, 'miracl') == 'aligned_group'
    del row['upstream_group']
    with pytest.raises(ValueError, match='upstream_group'):
        index.matches(row, upstream, 'miracl')


def test_builtin_miracl_preserves_alignment_for_split_and_relevance(monkeypatch):
    fake = types.ModuleType('datasets')
    configs = ['en', 'ar'] + [f'language{i}' for i in range(49)]
    fake.get_dataset_config_names = lambda *args, **kwargs: configs
    fake.load_dataset = lambda repo, language, **kwargs: [
        {'id': 42, 'query': language + ' q', 'positive': language + ' p', 'negative': language + ' n'}]
    monkeypatch.setitem(sys.modules, 'datasets', fake)
    rows = list(preparation.selected_rows('miracl', 29))
    assert len(rows) == 51 and len({row['language'] for row in rows}) == 51
    assert {row['group'] for row in rows} == {'miracl:42'}
    assert {row['upstream_group'] for row in rows} == {'42'}


def test_tampered_raw_probe_or_missing_inventory_fails(tmp_path):
    path = candidate(tmp_path)
    raw = next(path.glob('*.jsonl'))
    raw.write_text(raw.read_text() + '{}\n')
    with pytest.raises(ValueError, match='checksum'):
        load_candidate_exclusions(path)
    receipt = json.loads((path / 'exclusions.json').read_text())
    receipt['sources'].pop(next(iter(receipt['sources'])))
    (path / 'exclusions.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='inventory'):
        load_candidate_exclusions(path)


def test_mapped_language_cannot_be_rehashed_or_relabelled(tmp_path):
    spec = json.loads((Path(__file__).parents[1] / 'configs/data/representation-development-v1.json').read_text())
    spec.update(max_scanned_rows_per_config=100, max_selected_rows_per_config=4)
    source = spec['sources'][0]
    source.update(configs=['ar-triplet'], config_language_map={'ar-triplet': 'ar'},
                  group_policy='query-text', fields={'query': 'anchor', 'positive': 'positive', 'negative': 'negative'})
    spec['sources'] = [source]
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))
    output = tmp_path / 'mapped'
    export(spec_path, output, loader=lambda *args, **kwargs: (
        {'anchor': f'query {i}', 'positive': f'positive {i}', 'negative': f'negative {i}'} for i in range(100)))
    load_candidate_exclusions(output)
    raw = next(output.glob('*.jsonl'))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]['language'] = 'ar-triplet'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt_path = output / 'exclusions.json'
    receipt = json.loads(receipt_path.read_text())
    metadata = next(iter(receipt['sources'].values()))
    metadata['raw_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='row language'):
        load_candidate_exclusions(output)
    rows[0]['language'] = 'ar'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    metadata['raw_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    metadata['human_language_provenance'] = 'guessed-from-config'
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='provenance'):
        load_candidate_exclusions(output)


def test_preparation_quarantines_before_train_dev_tokenization(tmp_path, monkeypatch):
    path = candidate(tmp_path)
    index = load_candidate_exclusions(path)
    upstream = preparation.SOURCES['miracl']
    reserved = next(iter(index.aligned_groups[(upstream['repo'], upstream['revision'])]))
    clean = [str(i) for i in range(1000, 2000)]
    heldout = [g for g in clean if int(hashlib.sha256(f'29:{g}'.encode()).hexdigest(), 16) % 100 < 1][:2]
    train = [g for g in clean if g not in heldout][:3]
    rows = [dict(query=f'fresh q{g}', positive=f'fresh p{g}', negative=None, group=g,
                 upstream_group=g, language='zh') for g in heldout + train]
    rows += [dict(query='translated unseen q', positive='translated unseen p', negative=None,
                  group='unseen local group', upstream_group=reserved, language='zh')]
    monkeypatch.setattr(preparation, 'selected_rows', lambda *args: iter(rows))
    def tokenize(directory, *args):
        saved = [json.loads(line) for split in ('train', 'dev') for line in (directory / f'{split}.jsonl').read_text().splitlines()]
        assert all(row['upstream_group'] != reserved for row in saved)
        return {'vocab_size': 128}
    monkeypatch.setattr(preparation, '_tokenize', tokenize)
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}')
    manifest = preparation.prepare('miracl', tmp_path / 'prepared', tokenizer, development_exclusions=path)
    assert manifest['development_quarantine']['identity']['receipt_sha256'] == index.identity['receipt_sha256']
    assert manifest['development_quarantine']['rejected'] == {'independent_dev_aligned_group': 1}
    assert not manifest['development_quarantine']['parent_exposure_checked']
