import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from flaxchat.encoder_data import file_hash
from scripts.audit_encoder_documents import audit_sources


def fixture(tmp_path):
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'a':1,'b':2,'c':3,'d':4,'e':5},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    path=tmp_path/'tokenizer.json'
    tokenizer.save(str(path))
    sources, manifests=[],[]
    for split,text in [('train','a b c d e'),('validation','b c d e')]:
        source=tmp_path/f'{split}.jsonl'
        source.write_text(json.dumps(dict(id=split,text=text,split=split))+'\n')
        manifest=tmp_path/f'{split}.json'
        manifest.write_text(json.dumps(dict(format='flaxchat-encoder-rows-v1',split=split,
            documents=1,source_sha256=file_hash(source),tokenizer_sha256=file_hash(path),special_token_ids=[0])))
        sources.append(source)
        manifests.append(manifest)
    return (*sources,*manifests,path)


def test_complete_source_spans_and_read_only_audit(tmp_path):
    args=fixture(tmp_path)
    before=[file_hash(p) for p in args]
    report=audit_sources(*args,width=4)
    assert not report['passed']
    assert report['quarantine_document_ids']==['train']
    assert report['witnesses'][0]['train_start']==1
    assert [file_hash(p) for p in args]==before


@pytest.mark.parametrize('change',['source','tokenizer','count','split'])
def test_rejects_mismatched_source_identity(tmp_path,change):
    args=fixture(tmp_path)
    if change in ('source','tokenizer'):
        with args[0 if change=='source' else 4].open('a') as stream:
            stream.write(' ')
    else:
        path=args[2]
        manifest=json.loads(path.read_text())
        manifest['documents' if change=='count' else 'split']=2 if change=='count' else 'validation'
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit_sources(*args,width=4)
