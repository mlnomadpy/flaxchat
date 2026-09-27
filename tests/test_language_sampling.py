import json
import numpy as np
import pytest
from flaxchat.encoder_data import LanguageRows, file_hash


def inputs(tmp_path):
    tokens = np.array(
        [[2, 3, 0, 0], [2, 3, 0, 0], [2, 3, 4, 5], [2, 3, 4, 5]], dtype=np.int32
    )
    docs = [
        dict(id="a", language_script="a", first_row=0, end_row=2),
        dict(id="b", language_script="b", first_row=2, end_row=4),
    ]
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(map(json.dumps, docs)))
    manifest = dict(
        split="train",
        documents_sha256=file_hash(path),
        documents=2,
        pad_token_id=0,
        language_counts={
            "a": dict(rows=2, nonpadding_tokens=4),
            "b": dict(rows=2, nonpadding_tokens=8),
        },
    )
    return tokens, manifest


def test_uniform_token_target_corrects_row_occupancy(tmp_path):
    tokens, m = inputs(tmp_path)
    sampler = LanguageRows(tmp_path, tokens, m, 17, 0)
    np.testing.assert_allclose(sampler.row_probabilities, [2 / 3, 1 / 3])
    counts = sampler.token_counts(sampler.batch(0, 100000))
    assert abs(counts["a"] / sum(counts.values()) - 0.5) < 0.01


def test_exponent_one_recovers_uniform_rows_and_exact_step_replay(tmp_path):
    tokens, m = inputs(tmp_path)
    a = LanguageRows(tmp_path, tokens, m, 17, 1)
    np.testing.assert_allclose(a.row_probabilities, [0.5, 0.5])
    expected = a.batch(18, 100)
    a.batch(999, 200)
    b = LanguageRows(tmp_path, tokens, m, 17, 1)
    np.testing.assert_array_equal(expected, b.batch(18, 100))
    np.testing.assert_array_equal(
        np.concatenate([expected[:50], expected[50:]]), b.batch(18, 100)
    )
    assert not np.array_equal(expected, b.batch(19, 100))
    assert sum(a.token_counts(expected).values()) == int((tokens[expected] != 0).sum())


@pytest.mark.parametrize("defect", ["checksum", "accounting", "coverage", "exponent"])
def test_invalid_provenance_or_policy_rejected(tmp_path, defect):
    tokens, m = inputs(tmp_path)
    if defect == "checksum":
        m["documents_sha256"] = "wrong"
    if defect == "accounting":
        m["language_counts"]["a"]["nonpadding_tokens"] = 7
    if defect == "coverage":
        m["documents"] = 3
    with pytest.raises(ValueError):
        LanguageRows(
            tmp_path, tokens, m, 17, float("nan") if defect == "exponent" else 0.5
        )


def mixture_inputs(tmp_path):
    tokens = np.array([[2, 3, 0, 0]]*4 + [[2, 3, 4, 5]]*4, dtype=np.int32)
    docs = [dict(id=str(i), language_script=lang, source_group=source, first_row=2*i, end_row=2*i+2)
            for i, (source, lang) in enumerate([('hq','a'),('english','b'),('missing','a'),('reference','b')])]
    path = tmp_path/'documents.jsonl'
    path.write_text('\n'.join(map(json.dumps, docs)))
    manifest = dict(split='train',documents=4,documents_sha256=file_hash(path),pad_token_id=0,
        language_counts={lang:dict(rows=4,nonpadding_tokens=12) for lang in ['a','b']},
        source_token_shares=dict(hq=.6,english=.2,missing=.15,reference=.05))
    return tokens, manifest


def test_source_mixture_preserves_token_shares_and_replay(tmp_path):
    from flaxchat.encoder_data import MixtureRows
    tokens, manifest = mixture_inputs(tmp_path)
    sampler = MixtureRows(tmp_path,tokens,manifest,17,.5)
    ids = sampler.batch(10,200000)
    counts = sampler.source_token_counts(ids)
    for name, share in manifest['source_token_shares'].items():
        assert abs(counts[name]/sum(counts.values())-share)<.005
    assert sum(sampler.token_counts(ids).values()) == sum(counts.values())
    np.testing.assert_array_equal(ids,MixtureRows(tmp_path,tokens,manifest,17,.5).batch(10,200000))


@pytest.mark.parametrize('shares', [dict(hq=.6),dict(hq=.6,english=.2,missing=.2,reference=0),dict(hq=float('nan'),english=.2,missing=.15,reference=.05)])
def test_mixture_rejects_missing_sources_and_invalid_shares(tmp_path,shares):
    from flaxchat.encoder_data import MixtureRows
    tokens, manifest = mixture_inputs(tmp_path)
    manifest['source_token_shares']=shares
    with pytest.raises(ValueError):
        MixtureRows(tmp_path,tokens,manifest,17,.5)


def test_coverage_mixture_no_repeat_before_pool_exhaustion_and_exact_seek(tmp_path):
    from flaxchat.encoder_data import CoverageMixtureRows
    tokens, manifest = mixture_inputs(tmp_path)
    a = CoverageMixtureRows(tmp_path, tokens, manifest, 17, .5)
    batches = [a.batch(step, 7) for step in range(12)]
    assert int(a.row_exposures.sum()) == 84
    for indices in a.indices:
        selected = np.concatenate(batches)
        selected = selected[np.isin(selected, indices)]
        for start in range(0, len(selected), len(indices)):
            assert len(set(selected[start:start + len(indices)])) == len(selected[start:start + len(indices)])
        exposures = a.row_exposures[indices]
        assert int(exposures.max() - exposures.min()) <= 1
    b = CoverageMixtureRows(tmp_path, tokens, manifest, 17, .5)
    b.seek(8, 7)
    np.testing.assert_array_equal(b.batch(8, 7), batches[8])
    np.testing.assert_array_equal(b.batch(11, 7), batches[11])
    assert int(b.row_exposures.sum()) == 84
    assert all(v['total_exposures'] >= v['seen_rows'] for v in b.exposure_summary().values())


def test_coverage_mixture_seed_and_cursor_validation(tmp_path):
    from flaxchat.encoder_data import CoverageMixtureRows
    tokens, manifest = mixture_inputs(tmp_path)
    a = CoverageMixtureRows(tmp_path, tokens, manifest, 17, .5)
    b = CoverageMixtureRows(tmp_path, tokens, manifest, 18, .5)
    assert not np.array_equal(a.batch(0, 64), b.batch(0, 64))
    with pytest.raises(ValueError):
        a.seek(-1, 64)
    with pytest.raises(ValueError):
        a.batch(1, 0)


def test_coverage_mixture_reuses_one_permutation_per_pool(tmp_path):
    from flaxchat.encoder_data import CoverageMixtureRows
    tokens, manifest = mixture_inputs(tmp_path)
    sampler = CoverageMixtureRows(tmp_path, tokens, manifest, 17, .5)
    first = sampler._order(0, 0)
    second = sampler._order(1, 0)
    assert sampler._order(0, 0) is first
    assert sampler._order(1, 0) is second
    assert len(sampler._order_cache) == 2
    next_epoch = sampler._order(0, 1)
    assert next_epoch is sampler._order(0, 1)
    assert sampler._order(1, 0) is second
    assert len(sampler._order_cache) == 2
