"""Data orchestration checks; no model or accelerator computation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from flaxchat.encoder_data import CoverageMixtureRows, file_hash, load_prepared_rows
from scripts.prepare_yat_mlm_continuation import (
    MISSING_LANGUAGES, SHARES, SOURCES, merge_shards, normalized_digest, prior_coordinates, run,
    split_for_digest,
)


def test_pinned_recipe_and_normalized_split():
    assert sum(SHARES.values()) == 1
    assert len(MISSING_LANGUAGES) == 6
    assert set(SOURCES) == set(SHARES)
    assert all(len(pin) == 40 for _, pin in SOURCES.values())
    digest = normalized_digest('A  document\nwith text')
    assert digest == normalized_digest('A document with text')
    assert split_for_digest(digest) in {'train', 'validation'}
    assert split_for_digest('0' * 64) == 'validation'
    assert split_for_digest('1' * 64) == 'train'


def test_prior_coordinates_exclude_only_pinned_source_group(tmp_path):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps({'chunks': [dict(dataset='a', revision='b', path='c', row_group=3)]}))
    assert prior_coordinates([path]) == {('a', 'b', 'c', 3)}


def test_merge_keeps_provenance_accounting_and_tokens(tmp_path):
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}')
    paths = []
    for index, language in enumerate(['eng_Latn', 'code:Python', 'fra_Latn', 'hin_Deva']):
        path = tmp_path / str(index)
        path.mkdir()
        np.save(path / 'tokens.npy', np.array([[2, 5 + index, 3, 0]], dtype=np.int32))
        (path / 'documents.jsonl').write_text(json.dumps(dict(id=str(index), first_row=0, end_row=1, language_script=language, source_group=['english', 'code', 'hq', 'missing'][index])) + '\n')
        (path / 'manifest.json').write_text(json.dumps(dict(format='flaxchat-encoder-rows-v1', split='train', rows=1, documents=1,
            vocab_size=10, pad_token_id=0, mask_token_id=4, special_token_ids=[0, 2, 3, 4], sequence_length=4, nonpadding_tokens=3, language_counts={language: dict(documents=1, rows=1, nonpadding_tokens=3)})))
        paths.append(path)
    result = merge_shards(paths, tmp_path / 'train', 'pin', tokenizer)
    assert result['rows'] == 4
    assert result['nonpadding_tokens'] == 12
    assert result['nonpadding_fraction'] == .75
    assert result['tokens_sha256'] == file_hash(tmp_path / 'train/tokens.npy')
    docs = [json.loads(line) for line in (tmp_path / 'train/documents.jsonl').read_text().splitlines()]
    assert [(d['first_row'], d['end_row']) for d in docs] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert np.load(tmp_path / 'train/tokens.npy').tolist() == [[2, 5, 3, 0], [2, 6, 3, 0], [2, 7, 3, 0], [2, 8, 3, 0]]

    config = SimpleNamespace(vocab_size=10, pad_token_id=0, mask_token_id=4, max_position_embeddings=4)
    tokens, manifest = load_prepared_rows(tmp_path / 'train', config)
    sampler = CoverageMixtureRows(tmp_path / 'train', tokens, manifest, 106, .5)
    assert set(sampler.sources) == set(SHARES)
    assert len(sampler.batch(0, 3)) == 3


def test_nonempty_root_fails_closed_before_corpus_network(tmp_path):
    (tmp_path / 'partial').write_text('data')
    with pytest.raises(ValueError, match='empty output root'):
        run(SimpleNamespace(root=str(tmp_path)))
