import pytest
from flaxchat.ner import bio_spans, ner_metrics


def test_bio_boundaries_and_orphan_i_policy():
    assert bio_spans(['I-PER','I-PER','I-LOC','B-LOC','O','B-MISC']) == {
        ('PER',0,2),('LOC',2,3),('LOC',3,4),('MISC',5,6)}
    assert bio_spans(['O','O']) == set()
    for tag in ('B','S-PER','I-',None):
        with pytest.raises(ValueError):
            bio_spans([tag])


def test_exact_span_f1_differs_from_token_accuracy_and_is_language_scoped():
    result=ner_metrics([['B-PER','I-PER','O'],['B-LOC','O']],
                       [['B-PER','O','O'],['B-LOC','O']],['en','fr'])
    assert result['overall']['f1']==.5
    assert result['overall']['token_accuracy']==.8
    assert result['per_language']['en']['f1']==0
    assert result['per_language']['fr']['f1']==1
    assert result['overall']['gold_entities']==2


def test_sentences_do_not_merge_or_deduplicate_identical_spans():
    result=ner_metrics([['B-PER'],['I-PER']],[['B-PER'],['I-PER']],['en','en'])
    assert result['overall']['matched_entities']==2
    assert result['overall']['f1']==1
    assert ner_metrics([['O']],[['O']],['en'])['overall']['f1']==0


@pytest.mark.parametrize('gold,predicted,languages', [([],[],[]),([['O']],[],['en']),
    ([['O']],[['O','O']],['en']),([['O']],[['O']],['']),([[]],[[]],['en'])])
def test_rejects_missing_or_truncated_coverage(gold,predicted,languages):
    with pytest.raises(ValueError):
        ner_metrics(gold,predicted,languages)


def test_first_subword_alignment_uses_real_tokenizer_word_ids():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from flaxchat.ner import align_word_labels, align_encoding
    tokenizer=Tokenizer(models.WordPiece({'[UNK]':0,'[CLS]':1,'[SEP]':2,
                                          'play':3,'##ing':4,'Tokyo':5},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',
                                                           special_tokens=[('[CLS]',1),('[SEP]',2)])
    encoded=tokenizer.encode(['playing','Tokyo'],is_pretokenized=True)
    assert encoded.word_ids==[None,0,0,1,None]
    aligned,positions=align_word_labels(encoded.word_ids,[1,2],num_labels=3)
    assert aligned==[-100,1,-100,2,-100] and positions==[1,3]
    assert align_encoding(encoded,[1,2],num_labels=3)==(aligned,positions)
    tokenizer.enable_truncation(max_length=4)
    with pytest.raises(ValueError,match='dropped or truncated'):
        align_word_labels(tokenizer.encode(['playing','Tokyo'],is_pretokenized=True).word_ids,[1,2],num_labels=3)
    partial=tokenizer.encode(['Tokyo','playing'],is_pretokenized=True)
    assert set(w for w in partial.word_ids if w is not None)=={0,1}
    with pytest.raises(ValueError,match='overflow/truncation'):
        align_encoding(partial,[1,2],num_labels=3)


@pytest.mark.parametrize('words', [[0,2],[1,0],[0,1,0],[0,None,0,1],[0],[],[-1,0,1],[False,1]])
def test_alignment_rejects_partial_or_ambiguous_word_coverage(words):
    from flaxchat.ner import align_word_labels
    with pytest.raises(ValueError):
        align_word_labels(words,[0,1],num_labels=2)


def test_matches_frozen_seqeval_1_2_2_oracle_cases():
    import json
    from pathlib import Path
    fixture=json.loads((Path(__file__).parent/'fixtures/ner_seqeval_1_2_2.json').read_text())
    for case in fixture['cases']:
        result=ner_metrics(case['gold'],case['predicted'],case['languages'])
        for key,value in case['expected_overall'].items():
            assert result['overall'][key]==pytest.approx(value,abs=1e-15)
