import json
import numpy as np
import pytest
from scripts.evaluate_encoder import evaluation_order


def test_balanced_order_covers_languages_and_all_rows(tmp_path):
    docs = [dict(language_script=n, first_row=a, end_row=b) for n,a,b in [('ar',0,20),('en',20,24),('fr',24,30)]]
    (tmp_path/'documents.jsonl').write_text('\n'.join(map(json.dumps,docs)))
    order, labels = evaluation_order(tmp_path,30,seed=17,balanced_languages=True)
    assert sorted(order.tolist()) == list(range(30))
    assert [labels[i] for i in order[:9]] == ['ar','en','fr']*3
    np.testing.assert_array_equal(order,evaluation_order(tmp_path,30,seed=17,balanced_languages=True)[0])
    assert not np.array_equal(order,evaluation_order(tmp_path,30,seed=18,balanced_languages=True)[0])


@pytest.mark.parametrize('start,end',[(1,3),(0,4),(0,2),(0,0)])
def test_invalid_document_coverage_fails(tmp_path,start,end):
    (tmp_path/'documents.jsonl').write_text(json.dumps(dict(language_script='ar',first_row=start,end_row=end)))
    with pytest.raises(ValueError,match='coverage'):
        evaluation_order(tmp_path,3,seed=17,balanced_languages=True)


def test_legacy_prefix_does_not_need_document_sidecar(tmp_path):
    order, labels = evaluation_order(tmp_path,4,seed=17)
    assert order.tolist()==[0,1,2,3]
    assert labels is None


@pytest.mark.parametrize('processes', [1, 2, 4, 8])
def test_global_evaluation_batch_has_no_duplicates_across_hosts(processes):
    from scripts.evaluate_encoder import host_evaluation_rows
    x = np.arange(32).reshape(16, 2)
    y = x.copy()
    y[8:] = -1  # short global evaluation batches have ignored padding
    parts = [host_evaluation_rows(x, y, rank=i, processes=processes) for i in range(processes)]
    np.testing.assert_array_equal(np.concatenate([p[0] for p in parts]), x)
    np.testing.assert_array_equal(np.concatenate([p[1] for p in parts]), y)
    assert sum(int((p[1] >= 0).sum()) for p in parts) == int((y >= 0).sum())


def test_invalid_host_partition_fails():
    from scripts.evaluate_encoder import host_evaluation_rows
    with pytest.raises(ValueError):
        host_evaluation_rows(np.zeros((3, 2)), np.zeros((3, 2)), rank=0, processes=2)
