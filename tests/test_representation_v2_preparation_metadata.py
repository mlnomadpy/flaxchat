"""Real raw→tokenize→mixture→audit fixture; no model/provider execution."""
import argparse
import json
import shutil
import sys
import tarfile
import types

import pytest
import tokenizers

from scripts import prepare_representation_v2_training as runner
from scripts import prepare_yat_embedding_finetune as preparation
from scripts.qualify_yat_embedding_mixture import prepare_mixture
from tests.test_development_quarantine_metadata import candidate


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pack(root, path):
    with tarfile.open(path, 'w:gz') as archive:
        for item in root.rglob('*'):
            if item.is_file():
                archive.add(item, arcname=str(item.relative_to(root)))
    return runner.sha(path)


def fixture(tmp_path, monkeypatch, missing=False):
    portable, heldout = tmp_path / 'portable', tmp_path / 'heldout'
    portable.mkdir()
    shutil.copytree(candidate(tmp_path), portable / 'candidate')
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({'[PAD]': 0, '[UNK]': 1}, unk_token='[UNK]'))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = tmp_path / 'tokenizer.json'
    tok.save(str(tokenizer))
    dump(portable / 'parent-files.json', {'tokenizer.json': runner.sha(tokenizer)})
    dump(portable / 'metadata-inputs/producer-policies.json', {})
    for name in ['code', 'msmarco', 'miracl', runner.PAIR_SOURCE, 'contrastive_b', 'contrastive_c']:
        folder = portable / 'historical' / name
        folder.mkdir(parents=True)
        pair = name.startswith('contrastive_')
        train = []
        for i in range(600):
            row = dict(query=f'{name} question {i}', positive=f'{name} answer {i}', negative=None,
                       group=f'{name} group {i}', language='en')
            if pair:
                row = dict(sentence1=row['query'], sentence2=row['positive'], language1='en', language2='fr',
                           component=row['group'], provenance=[dict(dataset='sentence-transformers/parallel-sentences-global-voices',
                           revision='4cc20add371f246bb1559b543f8b0dea178a1803', split='train')])
            if name in ('code', 'msmarco'):
                row.pop('language')
            if name == 'code':
                row.pop('negative')
                if i % 2:
                    row['programming_language'] = 'python'
                    row['programming_language_provenance'] = 'historical-column:code_lang'
            train.append(row)
        (folder / 'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in train))
        held = heldout / name / 'dev.jsonl'
        dump(held, {'query': f'{name} original held query', 'positive': f'{name} original held answer'})
        if pair:
            manifest = dict(format='flaxchat-contrastive-text-pairs-v1', files={
                'train.jsonl': dict(rows=600, sha256=runner.sha(folder / 'train.jsonl')),
                'dev.jsonl': dict(sha256=runner.sha(held))})
        else:
            identity = ({'repo': 'sentence-transformers/codesearchnet', 'revision': '079a958b01dc87cf07b66a68414c4b4196d889cc'}
                        if name == 'code' else {'repo': f'fixture/{name}', 'revision': 'a'*40})
            manifest = dict(source_identity=identity,
                raw_files={'train.jsonl': runner.sha(folder/'train.jsonl'), 'dev.jsonl': runner.sha(held)})
        dump(folder / 'manifest.json', manifest)
        if missing and not pair:
            held.unlink()
    archive, supplement = tmp_path/'portable.tar.gz', tmp_path/'heldout.tar.gz'
    monkeypatch.setattr(runner, 'PORTABLE_SHA', pack(portable, archive))
    expected = pack(heldout, supplement)
    class Dataset(list):
        def shuffle(self, seed):
            assert seed == 29
            return self
    fake = types.ModuleType('datasets')
    fake.get_dataset_config_names = lambda *args, **kwargs: []
    fake.load_dataset = lambda repo, config, **kw: Dataset([
        dict(anchor=f'{config} native query {i}', positive=f'{config} native positive {i}', negative=f'{config} native negative {i}')
        for i in range(600)])
    monkeypatch.setitem(sys.modules, 'datasets', fake)
    output = tmp_path / 'output'
    output.mkdir()
    return argparse.Namespace(output=output, portable_archive=archive, heldout_archive=supplement,
                              heldout_sha256=expected, tokenizer=tokenizer)


def test_incomplete_original_supplement_reports_all_missing_sources(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, missing=True)
    with pytest.raises(ValueError, match='code/dev.jsonl, miracl/dev.jsonl, msmarco/dev.jsonl'):
        runner.raw(args)
    assert not (args.output/'registry.json').exists()


def test_full_real_preparation_and_quarantine_reaudit(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    runner.raw(args)
    registry = preparation.load_registry(args.output/'registry.json')
    assert len(registry) == 11
    assert {row['language'] for row in runner.rows(args.output/'raw/marco_replay.jsonl')} == {'und'}
    code_rows = list(runner.rows(args.output/'raw/code_replay.jsonl'))
    assert {row['language'] for row in code_rows} == {'und'}
    assert {row['programming_language'] for row in code_rows} == {'unknown', 'python'}
    assert all(row['negative'] is None for row in code_rows)
    monkeypatch.setattr(preparation, 'SOURCES', registry)
    sources = {}
    for name in registry:
        target = args.output/'prepared'/name
        preparation.prepare(name, target, args.tokenizer, seed=41, query_length=8, document_length=8,
                            development_exclusions=args.output/'development/candidate')
        sources[name] = target
    prepare_mixture(sources, args.output/'mixture', args.tokenizer)
    runner.audit(args)
    final_rows = list(runner.rows(args.output/'mixture/code_replay/train.jsonl'))
    assert all(row['programming_language_provenance'] == 'upstream-column-unavailable'
               for row in final_rows if row['programming_language'] == 'unknown')
    assert all(row['programming_language_provenance'] == 'historical-column:code_lang'
               for row in final_rows if row['programming_language'] == 'python')
    final_code = json.loads((args.output/'mixture/code_replay/manifest.json').read_text())
    assert set(final_code['tokenization']['programming_languages']['train']) == {'unknown', 'python'}
    result = json.loads((args.output/'final-quarantine.json').read_text())
    assert result['status'] == 'passed' and len(result['sources']) == 11
    assert all(item['train_rows'] > 256 for item in result['sources'].values())
    held = args.output/'original-heldout/code/dev.jsonl'
    held.write_text('{}')
    with pytest.raises(ValueError, match='Historical heldout changed'):
        runner.audit(args)


def test_supplement_digest_and_heldout_content_are_both_checked(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.heldout_sha256 = '0' * 64
    with pytest.raises(ValueError, match='Full archive hash mismatch'):
        runner.raw(args)
    shutil.rmtree(args.output/'development')
    held = tmp_path/'heldout/code/dev.jsonl'
    held.write_text('{"query":"changed"}')
    args.heldout_sha256 = pack(tmp_path/'heldout', args.heldout_archive)
    with pytest.raises(ValueError, match='Historical held-out bytes differ.*code/dev.jsonl'):
        runner.raw(args)
    assert not (args.output/'raw').exists()


def test_real_failed_subprocess_retains_diagnostics(tmp_path, monkeypatch):
    import subprocess
    real_popen = subprocess.Popen
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda command, **kw: real_popen(
        [sys.executable, '-c', 'import sys; print("missing heldout fixture", flush=True); sys.exit(7)'], **kw))
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--portable-archive', str(tmp_path/'portable'),
        '--heldout-archive', str(tmp_path/'heldout'), '--heldout-sha256', 'a'*64,
        '--tokenizer', str(tmp_path/'tokenizer'), '--output', str(output), '--total-seconds', '60'])
    with pytest.raises(RuntimeError, match='exited 7'):
        runner.main()
    report = json.loads((output/'terminal-data.json').read_text())
    assert report['status'] == 'failed'
    assert report['original_heldout_sha256'] == 'a'*64
    command = report['commands'][0]
    assert command['returncode'] == 7
    assert 'missing heldout fixture' in command['worker_error_tail']
    assert (output/command['worker_log']).exists()


@pytest.mark.parametrize('field,value', [('query', None), ('positive', ''), ('group', None),
                                         ('negative', 123), ('language', []), ('programming_language', {})])
def test_invalid_replay_schema_fails_before_registry_materialization(field, value):
    row = dict(query='query', positive='positive', group='group')
    row[field] = value
    with pytest.raises(ValueError, match='replay'):
        runner.replay_row(row, 'code_replay')


def test_replay_defaults_do_not_mutate_original_or_infer_language():
    original = dict(query='code query', positive='def f(): pass', group='original-id')
    normalized = runner.replay_row(original, 'code_replay')
    assert normalized['language'] == 'und'
    assert normalized['programming_language'] == 'unknown'
    assert normalized['programming_language_provenance'] == 'unavailable'
    assert normalized['negative'] is None
    assert set(original) == {'query', 'positive', 'group'}
