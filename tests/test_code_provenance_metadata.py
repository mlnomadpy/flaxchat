"""Programming-language metadata only; no model or backend execution."""
import hashlib
import json
import numpy as np
import pytest
from flaxchat.embedding_data import qualify_mixture, ExposureTracker
from scripts.prepare_yat_embedding_finetune import code_language_provenance, _registry_rows


def test_declared_code_language_is_separate_and_missing_stays_unknown():
    assert code_language_provenance({'language': ' Python '}) == {
        'programming_language': 'python', 'programming_language_provenance': 'column:language'}
    assert code_language_provenance({'comment': 'English comment', 'code': 'def function(): pass'}) == {
        'programming_language': 'unknown', 'programming_language_provenance': 'unavailable'}


def test_registry_preserves_distinct_human_and_programming_languages(tmp_path):
    path = tmp_path / 'source.jsonl'
    path.write_text(json.dumps({'q': 'query', 'p': 'code', 'human': 'en', 'code_lang': 'rust'}) + '\n')
    spec = {'jsonl': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'candidate_limit': None,
            'fields': {'query': 'q', 'positive': 'p', 'language': 'human', 'programming_language': 'code_lang'},
            'modalities': {'positive': 'code'}}
    row = next(_registry_rows(spec, 29))
    assert row['language'] == 'en' and row['programming_language'] == 'rust'
    assert row['programming_language_provenance'] == 'column:code_lang'


def test_authenticated_code_labels_survive_mixture_exposure_resume(tmp_path):
    data = {}
    raw = {}
    for split in ('train', 'dev'):
        rows = [{'query': f'{split} query {i}', 'positive': f'{split} code {i}', 'negative': None,
                 'group': f'{split}-{i}', 'language': 'en', 'modalities': {'positive': 'code'},
                 'programming_language': code} for i, code in enumerate(('python', 'rust'))]
        path = tmp_path / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        raw[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data[split] = {'negative_valid': np.zeros(2, bool),
                      **{f'{field}_tokens': np.ones((2, 4), np.int32) for field in ('query', 'positive', 'negative')}}
    receipt = qualify_mixture({'code': data}, {'code': tmp_path}, {'code': {'rows': {'train': 2, 'dev': 2}, 'raw_files': raw}})
    assert set(receipt['programming_language_ids']) == {'python', 'rust'}
    assert len(receipt['language_ids']) == 1
    tracker = ExposureTracker({'code': data}, 0)
    tracker.record('code', [0, 1, 0])
    assert tracker.report()['code']['programming_languages'] == {'python': 2, 'rust': 1}
    restored = ExposureTracker({'code': data}, 0)
    restored.restore(tracker.state())
    assert restored.report() == tracker.report()
    broken = tracker.state()
    counts = json.loads(bytes(broken['exposure_counts']).decode())
    counts['code']['programming_languages']['rust'] = 0
    broken['exposure_counts'] = np.frombuffer(json.dumps(counts).encode(), np.uint8)
    with pytest.raises(ValueError, match='programming-language exposure'):
        restored.restore(broken)
