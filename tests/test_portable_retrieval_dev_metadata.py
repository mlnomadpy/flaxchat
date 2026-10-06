"""Relocation of real tiny authenticated raw/token artifacts, no model work."""
import json
from pathlib import Path
import shutil

import pytest

from flaxchat.embedding_quality import load_retrieval_dev
from scripts.prepare_embedding_retrieval_dev import prepare, portable_exposure_path
from tests.test_independent_retrieval_dev_metadata import fixture


def test_whole_bundle_move_preserves_full_exposure_proof(tmp_path):
    root = tmp_path / 'before'
    root.mkdir()
    args, data, original, config, hashes, _ = fixture(root)
    parent = Path(args.parent_public)
    output = root / 'dev'
    manifest = prepare(original / 'candidate',
        json.loads((original / 'manifest.json').read_text())['candidate_sources'],
        root / 'exposure.json', Path(args.parent_manifest), parent / 'tokenizer.json',
        parent / 'config.json', output, query_length=4, document_length=4, portable_root=root)
    spec = json.loads((output / 'parent-exposure-input.json').read_text())
    assert all(not Path(stage['checkpoint_metadata']).is_absolute() and
               all(not Path(source['directory']).is_absolute() for source in stage['sources'])
               for stage in spec['stages'])
    proof_bytes = (output / 'parent-exposure-proof.json').read_bytes()
    data_relative = data.relative_to(root)
    moved = tmp_path / 'after'
    shutil.move(root, moved)
    assert not root.exists()
    arrays, loaded, _ = load_retrieval_dev(moved / 'dev', config, hashes['tokenizer.json'],
        training_directories=[moved / data_relative], parent_hashes=hashes)
    assert len(arrays['query_tokens']) == 8
    assert loaded == manifest
    assert (moved / 'dev/parent-exposure-proof.json').read_bytes() == proof_bytes
    # Recomputed full proof reads the relocated raw data: tampering still fails.
    historical = moved / 'dev' / spec['stages'][0]['sources'][0]['directory']
    with (historical / 'train.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='Historical raw training checksum'):
        load_retrieval_dev(moved / 'dev', config, hashes['tokenizer.json'],
            training_directories=[moved / data_relative], parent_hashes=hashes)


@pytest.mark.parametrize('kind', ['external', 'external_output', 'link_inside', 'link_outside'])
def test_portability_rejects_external_or_symlink_inputs(tmp_path, kind):
    root = tmp_path / 'bundle'
    root.mkdir()
    inside = root / 'raw.jsonl'
    inside.write_text('{}\n')
    outside = tmp_path / 'raw.jsonl'
    outside.write_text('{}\n')
    output = root / 'dev'
    path = inside
    if kind == 'external':
        path = outside
    elif kind == 'external_output':
        output = tmp_path / 'outside-dev'
    else:
        path = root / 'link'
        path.symlink_to(inside if kind == 'link_inside' else outside)
    with pytest.raises(ValueError, match='inside portable root|symlink'):
        portable_exposure_path(path, root, output)
