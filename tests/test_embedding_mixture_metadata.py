"""Metadata-only data contracts; no JAX import or model execution."""
import hashlib
import json
import numpy as np
import pytest
from flaxchat.embedding_data import (text_identity, qualify_mixture, ReplayRows,
                                      homogeneous_source)


def source(tmp_path, name, train, dev):
    folder = tmp_path / name
    folder.mkdir()
    raw = {}
    for split, rows in [('train', train), ('dev', dev)]:
        content = ''.join(json.dumps(row) + '\n' for row in rows)
        (folder / f'{split}.jsonl').write_text(content)
        raw[f'{split}.jsonl'] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {'rows': {'train': len(train), 'dev': len(dev)}, 'raw_files': raw}
    arrays = {split: {'negative_valid': np.array([r.get('negative') is not None for r in rows], np.bool_)}
              for split, rows in [('train', train), ('dev', dev)]}
    return folder, manifest, arrays


def row(query, positive, negative=None, group='g'):
    return dict(query=query, positive=positive, negative=negative, group=group, language='en')


def test_code_semantics():
    assert text_identity('Foo()', 'code') != text_identity('foo()', 'code')
    assert text_identity('x = "a b"', 'code') != text_identity('x = "a  b"', 'code')
    assert text_identity('  x', 'code') != text_identity('x', 'code')
    assert text_identity('x\r\n', 'code') == text_identity('x\n', 'code')


@pytest.mark.parametrize('negative', [None, 'N'])
def test_raw_negative_presence_authenticates_training_flag(tmp_path, negative):
    folder, manifest, arrays = source(tmp_path, 'a', [row('Q', 'P', negative)], [row('DQ', 'DP', group='dev')])
    arrays['train']['negative_valid'][0] = negative is None
    with pytest.raises(ValueError, match='Raw negative presence'):
        qualify_mixture({'a': arrays}, {'a': folder}, {'a': manifest})


def test_global_relevance_and_duplicate_identity(tmp_path):
    a, ma, aa = source(tmp_path, 'a', [row('Q', 'P1'), row('Q', 'P2')], [row('devA', 'devAP', group='da')])
    b, mb, ab = source(tmp_path, 'b', [row('other', 'P1', 'P2', 'b')], [row('devB', 'devBP', group='db')])
    receipt = qualify_mixture({'a': aa, 'b': ab}, {'a': a, 'b': b}, {'a': ma, 'b': mb})
    assert aa['train']['positive_text_ids'][0] == ab['train']['positive_text_ids'][0]
    assert ab['train']['negative_text_ids'][0] in aa['train']['known_positive_text_ids'][0]
    assert receipt['known_positive_width'] == 2


def test_cross_source_leak_fails(tmp_path):
    a, ma, aa = source(tmp_path, 'a', [row('Q', 'leak')], [row('DA', 'DPA', group='da')])
    b, mb, ab = source(tmp_path, 'b', [row('QB', 'PB')], [row('DB', 'leak', group='db')])
    with pytest.raises(ValueError, match='held-out overlap'):
        qualify_mixture({'a': aa, 'b': ab}, {'a': a, 'b': b}, {'a': ma, 'b': mb})


def test_replay_resume_and_homogeneous_selection():
    a = ReplayRows(np.array([1, 1, 2]), 29, .5)
    before = [a.batch(step, 8) for step in range(6)]
    b = ReplayRows(np.array([1, 1, 2]), 29, .5)
    assert all(np.array_equal(before[step], b.batch(step, 8)) for step in range(3, 6))
    names, weights = ['a', 'b'], {'a': .7, 'b': .3}
    assert homogeneous_source(7, names, weights, 29) == homogeneous_source(7, names[::-1], weights, 29)


def test_homogeneous_seek_does_not_skip_source_batches():
    from flaxchat.embedding_data import HomogeneousSchedule
    schedule = HomogeneousSchedule(['a', 'b'], {'a': .7, 'b': .3}, 29)
    counts = {'a': 0, 'b': 0}
    for step in range(35):
        name, cursor = schedule.at(step)
        assert cursor == counts[name]
        counts[name] += 1
    resumed = HomogeneousSchedule(['a', 'b'], {'a': .7, 'b': .3}, 29)
    assert resumed.at(30) == schedule.at(30)


def test_inherited_tokenizer_truncation_is_disabled(tmp_path):
    tokenizers = pytest.importorskip('tokenizers')
    from scripts.prepare_yat_embedding_finetune import _tokenize
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({'[UNK]': 0, 'word': 1}, unk_token='[UNK]'))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.enable_truncation(max_length=2)
    path = tmp_path / 'tokenizer.json'
    tok.save(str(path))
    folder = tmp_path / 'prepared'
    folder.mkdir()
    raw = row('word word word word word', 'word word word word word', group='g')
    for split in ('train', 'dev'):
        (folder / f'{split}.jsonl').write_text(json.dumps(raw) + '\n')
    receipt = _tokenize(folder, path, 4, 4, 0)
    assert receipt['truncated']['train:query'] == 1
    assert np.array_equal(np.load(folder / 'train/query_tokens.npy'), [[1, 1, 1, 1]])


def test_registry_requires_pinned_training_source(tmp_path):
    from scripts.prepare_yat_embedding_finetune import load_registry
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps({'bitext': {'repo': 'example/pairs', 'revision': 'main',
                                         'fields': {'query': 'a', 'positive': 'b'}}}))
    with pytest.raises(ValueError, match='pin'):
        load_registry(path)
    path.write_text(json.dumps({'bitext': {'repo': 'example/pairs', 'revision': 'a' * 40,
                  'fields': {'query': 'a', 'positive': 'b', 'language': 'lang'}}}))
    assert load_registry(path)['bitext']['train_limit'] is None


def test_cumulative_exposure_checkpoint_replays_unique_coverage():
    from flaxchat.embedding_data import ExposureTracker
    arrays = {'language_ids': np.array([1, 2, 2]),
              'negative_valid': np.array([False, True, False])}
    for field in ('query', 'positive', 'negative'):
        arrays[f'{field}_tokens'] = np.array([[1, 0], [2, 3], [4, 0]], np.int32)
    data = {'pairs': {'train': arrays}}
    tracker = ExposureTracker(data, 0)
    tracker.record('pairs', [0, 1, 1])
    resumed = ExposureTracker(data, 0)
    resumed.restore(tracker.state())
    resumed.record('pairs', [2, 0])
    tracker.record('pairs', [2, 0])
    assert resumed.report() == tracker.report()
    assert tracker.report()['pairs']['unique_rows'] == 3
    assert tracker.report()['pairs']['repeated_pairs'] == 2
    assert tracker.report()['pairs']['nonpadding_tokens']['negative'] == 4


def test_query_group_equivalence_closure_keeps_all_query_memberships_consistent(tmp_path):
    folder, manifest, arrays = source(tmp_path, 'pairs',
        [row('q1', 'p1', group='g1'), row('q1', 'p2', group='g2'), row('q2', 'p3', group='g2')],
        [row('held', 'heldpassage', group='heldgroup')])
    receipt = qualify_mixture({'pairs': arrays}, {'pairs': folder}, {'pairs': manifest})
    memberships = arrays['train']['known_positive_text_ids']
    assert all(np.array_equal(memberships[0], membership) for membership in memberships)
    assert set(memberships[0]) == set(arrays['train']['positive_text_ids'])
    assert receipt['relevance_policy'].endswith('closure-v1')
    assert len(set(arrays['train']['positive_group_ids'])) == 1


def test_mixture_quarantine_removes_connected_group_without_direct_text_overlap(tmp_path, monkeypatch):
    from scripts.qualify_yat_embedding_mixture import prepare_mixture
    a, ma, _ = source(tmp_path, 'a', [row('q1', 'p1', group='g1'), row('q2', 'p2', group='g1'),
                                     row('safea1', 'safeap1', group='safea1'), row('safea2', 'safeap2', group='safea2')],
                       [row('da1', 'dpa1', group='da1'), row('da2', 'dpa2', group='da2')])
    b, mb, _ = source(tmp_path, 'b', [row('safeb1', 'safebp1', group='safeb1'), row('safeb2', 'safebp2', group='safeb2')],
                       [row('q1', 'dpb1', group='gb'), row('db2', 'dpb2', group='db2')])
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}')
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    for name, folder, manifest in [('a', a, ma), ('b', b, mb)]:
        manifest.update(source=name, tokenizer_sha256=digest, query_length=8, document_length=8, pad_id=0)
        (folder / 'manifest.json').write_text(json.dumps(manifest))
    # Tokenization has its own real tokenizer tests; isolate quarantine metadata.
    monkeypatch.setattr('scripts.qualify_yat_embedding_mixture._tokenize', lambda *args: {'vocab_size': 128})
    output = tmp_path / 'qualified'
    receipt = prepare_mixture({'a': a, 'b': b}, output, tokenizer)
    kept = [json.loads(line)['query'] for line in (output / 'a/train.jsonl').read_text().splitlines()]
    assert kept == ['safea1', 'safea2']
    assert receipt['quarantined']['a']['global_dev_overlap'] == 2


def test_actual_code_preparation_deduplicates_without_conflating_semantics(tmp_path, monkeypatch):
    import tokenizers
    from scripts import prepare_yat_embedding_finetune as preparation
    from scripts.release_contract import checkpoint_identity, canonical_hash
    from flaxchat.encoder_data import file_hash

    def groups(heldout, count):
        return [str(i) for i in range(1, 5000)
                if (int(hashlib.sha256(f'29:{i}'.encode()).hexdigest(), 16) % 100 < 1) == heldout][:count]

    train_groups = groups(False, 4)
    dev_groups = groups(True, 2)
    code_variants = ['Foo()', 'foo()', '  Foo()', 'x = "a  b"']
    rows = [row('shared query', value, group=group)
            for value, group in zip(code_variants, train_groups, strict=True)]
    rows += [row(f'dev query {index}', f'dev_code_{index}()', group=group)
             for index, group in enumerate(dev_groups)]
    rows.append(dict(rows[0]))  # Exact duplicate, not a case/indentation variant.
    monkeypatch.setattr(preparation, 'selected_rows', lambda *args: iter([dict(value) for value in rows]))
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {'[UNK]': 0, '[PAD]': 1, 'shared': 2, 'query': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(tokenizer_path))
    first = tmp_path / 'prepared-code-v2'
    receipt = preparation.prepare('code', first, tokenizer_path, query_length=4,
                                  document_length=8, pad_id=1)
    actual_rows = [json.loads(line) for line in (first / 'train.jsonl').read_text().splitlines()]
    assert [value['positive'] for value in actual_rows] == code_variants
    assert receipt['rows'] == {'train': 4, 'dev': 2}
    assert receipt['rejected']['duplicate_triplet'] == 1
    assert len({value['text_hashes'][1] for value in actual_rows}) == 4
    assert receipt['identity_policy'] == 'modality-exact-v2'
    assert receipt['preparation_sha256'] == file_hash(preparation.__file__)
    assert receipt['identity_implementation_sha256'] == file_hash(
        preparation.Path(preparation.__file__).parents[1] / 'flaxchat/embedding_data.py')

    original_data_sha = file_hash(first / 'manifest.json')
    original_stage = checkpoint_identity({'resolved_config': {'data_manifests': {'code': original_data_sha}},
                                          'data_manifest_identity': original_data_sha})
    monkeypatch.setattr(preparation, 'IDENTITY_POLICY', 'modality-exact-v3-test')
    second = tmp_path / 'prepared-code-v3'
    changed = preparation.prepare('code', second, tokenizer_path, query_length=4,
                                  document_length=8, pad_id=1)
    changed_data_sha = file_hash(second / 'manifest.json')
    changed_stage = checkpoint_identity({'resolved_config': {'data_manifests': {'code': changed_data_sha}},
                                         'data_manifest_identity': changed_data_sha})
    assert changed['identity_policy'] == 'modality-exact-v3-test'
    assert changed_data_sha != original_data_sha
    assert canonical_hash(changed_stage) != canonical_hash(original_stage)
    # Re-preparation is a new identity; original prepared data remains unchanged.
    assert file_hash(first / 'manifest.json') == original_data_sha
