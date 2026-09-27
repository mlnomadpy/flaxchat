import json
from types import SimpleNamespace

import numpy as np
import pytest

from flaxchat.encoder_data import file_hash
from scripts.audit_encoder_overlap import audit


def fixture(tmp_path):
    config = SimpleNamespace(vocab_size=30, pad_token_id=0, mask_token_id=1, max_position_embeddings=32)
    for name, rows in [('train', [[2, 3, 4, 0], [2, 3, 4, 0], [8, 9, 10, 0]]), ('validation', [[2, 3, 4, 0]])]:
        directory = tmp_path / name
        directory.mkdir()
        np.save(directory / 'tokens.npy', np.asarray(rows, dtype=np.int32))
        docs = [dict(id='first', first_row=0, end_row=2), dict(id='second', first_row=2, end_row=3)]
        (directory / 'documents.jsonl').write_text('\n'.join(map(json.dumps, docs)))
        manifest = dict(format='flaxchat-encoder-rows-v1', split=name, vocab_size=30,
                        pad_token_id=0, mask_token_id=1, special_token_ids=[0, 1],
                        tokenizer_sha256='tokenizer', tokens_sha256=file_hash(directory / 'tokens.npy'),
                        documents_sha256=file_hash(directory / 'documents.jsonl'), documents=2)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
    return config, tmp_path / 'train', tmp_path / 'validation'


def test_quarantines_complete_documents_and_keeps_evaluation_unchanged(tmp_path):
    config, train, candidate = fixture(tmp_path)
    before = file_hash(candidate / 'tokens.npy')
    report = audit(train, [candidate], config, width=3)
    assert report['matching_train_rows'] == 2
    assert report['quarantine_document_ids'] == ['first']
    assert report['passed'] is False
    assert file_hash(candidate / 'tokens.npy') == before
    assert audit(train, [candidate], config, width=4)['passed'] is True


@pytest.mark.parametrize('change', ['checksum', 'coverage', 'tokenizer', 'split', 'duplicate_candidates'])
def test_rejects_invalid_provenance(tmp_path, change):
    config, train, candidate = fixture(tmp_path)
    if change in ('checksum', 'coverage'):
        path = train / 'documents.jsonl'
        path.write_text(json.dumps(dict(id='first', first_row=1, end_row=3)))
        if change == 'coverage':
            mpath = train / 'manifest.json'
            manifest = json.loads(mpath.read_text())
            manifest['documents_sha256'] = file_hash(path)
            mpath.write_text(json.dumps(manifest))
    elif change != 'duplicate_candidates':
        path = candidate / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['tokenizer_sha256' if change == 'tokenizer' else 'split'] = 'wrong'
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit(train, [candidate] * (2 if change == 'duplicate_candidates' else 1), config, width=3)
