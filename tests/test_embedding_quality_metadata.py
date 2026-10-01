"""Host metric/selection tests only; no model execution."""
import numpy as np
import pytest
from flaxchat.embedding_quality import stratified_dev_indices, sts_metrics, quality_gate


def test_programming_regressions_do_not_multiply_task_selection_votes():
    from flaxchat.embedding_quality import programming_language_slices
    labels = np.array(['python', 'unknown', 'python', 'java', 'java', 'und'])
    slices = programming_language_slices(labels, np.arange(6))
    assert set(slices) == {'python', 'java'}
    assert np.array_equal(slices['python'], [0, 2])
    high = dict(mrr=.9, recall_at_1=.9, recall_at_10=1.)
    low = dict(mrr=.3, recall_at_1=.3, recall_at_10=.8)
    baseline = {'code': high, 'alignment': low,
                'code/programming-language/python': high, 'code/programming-language/java': high}
    gate = quality_gate(baseline, baseline, max_regression=.02)
    assert gate['score'] == pytest.approx(.6)
    degraded = {**baseline, 'code/programming-language/python': low}
    gate = quality_gate(degraded, baseline, max_regression=.02)
    assert not gate['passed']
    assert 'code/programming-language/python/mrr' in gate['regressions']
    assert gate['score'] == pytest.approx(.6)


def test_multi_positive_fractional_recall_and_duplicate_candidates():
    from flaxchat.embedding_quality import retrieval_metrics
    vectors = np.array([[1., 0.], [.8, .2], [1., 0.]])
    result = retrieval_metrics(vectors, vectors, [1, 1, 1], [10, 11, 10], [1, 1, 1])
    assert result['unique_documents'] == 2
    assert result['mrr'] == 1
    assert result['recall_at_1'] == .5
    assert result['recall_at_10'] == 1


def test_stratified_probe_includes_every_language_and_replays():
    languages = np.array(['en'] * 100 + ['sw'] * 3 + ['ar'] * 4)
    first = stratified_dev_indices(languages, 12, 29)
    assert np.array_equal(first, stratified_dev_indices(languages, 12, 29))
    assert set(languages[first]) == {'en', 'sw', 'ar'}
    assert all(np.count_nonzero(languages[first] == language) >= 2 for language in set(languages))
    with pytest.raises(ValueError):
        stratified_dev_indices(languages, 5, 29)


def test_sts_ties_correlation_and_regression_gate():
    q = np.repeat([[1., 0.]], 4, axis=0)
    p = np.array([[0., 1.], [.5, np.sqrt(.75)], [.5, np.sqrt(.75)], [1., 0.]])
    metrics = sts_metrics(q, p, [0, .5, .5, 1])
    assert metrics['pearson'] == pytest.approx(1)
    assert metrics['spearman'] == pytest.approx(1)
    baseline = {'sts/dev': metrics}
    assert quality_gate(baseline, baseline, max_regression=.01)['passed']
    degraded = {'sts/dev': {**metrics, 'spearman': .5}}
    assert not quality_gate(degraded, baseline, max_regression=.01)['passed']
    with pytest.raises(ValueError, match='Constant'):
        sts_metrics(q, p, [1, 1, 1, 1])


def test_pinned_sts_prepare_load_and_test_split_rejection(tmp_path):
    import json
    import tokenizers
    from flaxchat.encoder_data import file_hash
    from flaxchat.embedding_quality import load_sts_dev
    from scripts.prepare_embedding_sts_dev import prepare
    vocab = {'[PAD]': 0, '[UNK]': 1, 'a': 2, 'b': 3}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = tmp_path / 'tokenizer.json'
    tok.save(str(tokenizer))
    config = {'vocab_size': 4, 'pad_token_id': 0, 'max_position_embeddings': 8}
    configuration = tmp_path / 'config.json'
    configuration.write_text(json.dumps(config))
    raw = tmp_path / 'dev.jsonl'
    raw.write_text(''.join(json.dumps({'sentence1': 'a', 'sentence2': 'b', 'score': score, 'language': 'en'}) + '\n'
                           for score in [0., .5, 1.]))
    output = tmp_path / 'prepared'
    manifest = prepare(raw, file_hash(raw), tokenizer, configuration, output, source_split='validation', length=4)
    arrays, _, digest = load_sts_dev(output, config, file_hash(tokenizer), tokenizer_path=tokenizer)
    assert len(arrays['scores']) == 3 and len(digest) == 64
    tokens = np.load(output / 'query_tokens.npy')
    tokens[0, 0] = 3
    np.save(output / 'query_tokens.npy', tokens)
    manifest['files']['query_tokens.npy'] = file_hash(output / 'query_tokens.npy')
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='raw/token'):
        load_sts_dev(output, config, file_hash(tokenizer), tokenizer_path=tokenizer)
    manifest['source_split'] = 'test'
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='split'):
        load_sts_dev(output, config, file_hash(tokenizer))


def test_production_sts_binds_exact_candidate_pairs_scores_and_source(tmp_path):
    """Binding-layer fixture; parent proof execution is covered by loader tests."""
    import json
    from pathlib import Path
    from flaxchat.embedding_quality import bind_production_sts
    from flaxchat.embedding_development_quarantine import load_candidate_exclusions
    from scripts.prepare_representation_development import export
    from flaxchat.encoder_data import file_hash
    spec = json.loads((Path(__file__).parents[1] / 'configs/data/representation-development-v1.json').read_text())
    spec['sources'] = [spec['sources'][-1]]
    spec.update(max_scanned_rows_per_config=4, max_selected_rows_per_config=4)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    independent = tmp_path / 'independent'
    independent.mkdir()
    candidate = independent / 'candidate'
    export(path, candidate, loader=lambda *args, **kwargs: (
        dict(sentence1=f'q{i}', sentence2=f'p{i}', score=i / 3) for i in range(4)))
    index = load_candidate_exclusions(candidate)
    name = 'stsb_validation__default'
    sts = tmp_path / 'sts'
    sts.mkdir()
    (sts / 'raw.jsonl').write_bytes((candidate / (name + '.jsonl')).read_bytes())
    source = spec['sources'][0]
    manifest = {'source_split': 'validation', 'source_identity': {
        'repo': source['repo'], 'revision': source['revision'], 'sha256': file_hash(sts / 'raw.jsonl')}}
    receipt = {'candidate_identity': index.identity, 'parent_exposure': {'fixture_scope': 'already-verified-loader-receipt'}}
    binding = bind_production_sts(sts, manifest, independent, receipt)
    assert binding['candidate_identity'] == index.identity
    assert binding['parent_exposure'] == receipt['parent_exposure']
    manifest['source_identity']['revision'] = 'f' * 40
    with pytest.raises(ValueError, match='source differs'):
        bind_production_sts(sts, manifest, independent, receipt)
    manifest['source_identity']['revision'] = source['revision']
    rows = [json.loads(line) for line in (sts / 'raw.jsonl').read_text().splitlines()]
    for field, value in [('score', .125), ('sentence1', 'foreign pair')]:
        modified = [dict(row) for row in rows]
        modified[0][field] = value
        (sts / 'raw.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in modified))
        manifest['source_identity']['sha256'] = file_hash(sts / 'raw.jsonl')
        with pytest.raises(ValueError, match='actual authenticated candidate'):
            bind_production_sts(sts, manifest, independent, receipt)
