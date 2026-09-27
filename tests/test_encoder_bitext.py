import gzip
import json
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from flaxchat.encoder_data import file_hash
from scripts.prepare_encoder_bitext import prepare


@pytest.mark.parametrize('defect', [None, 'hash', 'empty', 'long', 'limit', 'language'])
def test_bitext_preserves_alignment_and_rejects_invalid_inputs(tmp_path, defect):
    tokenizer = Tokenizer(models.WordLevel({'[PAD]':0, '[UNK]':1, 'hello':2, 'world':3, '[MASK]':4}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path/'tokenizer.json'
    tokenizer.save(str(path))
    rows = [dict(sentence1='hello', sentence2='world', lang='x-eng') for _ in range(2)]
    if defect == 'empty':
        rows[1]['sentence1'] = ''
    if defect == 'long':
        rows[1]['sentence1'] = 'hello '*10
    if defect == 'language':
        rows[1]['lang'] = 'y-eng'
    source = tmp_path/'pairs.jsonl.gz'
    with gzip.open(source, 'wt') as stream:
        stream.write('\n'.join(map(json.dumps, rows)))
    output = tmp_path/'prepared'
    kwargs = dict(source_sha256='wrong' if defect == 'hash' else file_hash(source),
                  dataset='fixture', revision='pinned', subset='x-eng', sequence_length=4,
                  max_pairs=1 if defect == 'limit' else 2)
    if defect:
        with pytest.raises(ValueError):
            prepare(source, path, output, **kwargs)
        assert not output.exists()
    else:
        result = prepare(source, path, output, **kwargs)
        assert result['pairs'] == 2
        task = json.loads((output/'judgments.json').read_text())
        assert task['qrels'] == {'q000000': {'d000000':1}, 'q000001': {'d000001':1}}
        np.testing.assert_array_equal(np.load(output/'queries/tokens.npy'), [[2,0,0,0]]*2)
        np.testing.assert_array_equal(np.load(output/'corpus/tokens.npy'), [[3,0,0,0]]*2)
        with pytest.raises(ValueError):
            prepare(source, path, output, **kwargs)
