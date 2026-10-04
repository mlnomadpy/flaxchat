"""Historical semantics from literal metadata; no model/backend/network."""
import copy
import pytest
from flaxchat.embedding_data import row_identities
from flaxchat.embedding_historical_rows import historical_row_view, CODESEARCHNET, MIRACL, VERIFIED_LEGACY_PRODUCERS
from flaxchat.embedding_development_quarantine import DevelopmentExclusions


def identity(pair):
    return dict(zip(('repo', 'revision'), pair, strict=True))


def test_code_alias_cannot_normalize_away_exact_decomposed_unicode_overlap():
    row = {'query': 'comment', 'positive': 'return "e\u0301"', 'negative': None, 'group': 'g'}
    original = copy.deepcopy(row)
    candidate = {**row, 'modalities': {'positive': 'code'}}
    raw_code_hash = row_identities(candidate, 'code')['positive']
    assert row_identities(row, 'legacy_alias')['positive'] != raw_code_hash
    view = historical_row_view(row, identity(CODESEARCHNET), 'legacy_alias')
    assert row_identities(view, 'legacy_alias')['positive'] == raw_code_hash
    index = DevelopmentExclusions({}, frozenset({raw_code_hash}), {})
    assert index.matches(view, identity(CODESEARCHNET), 'legacy_alias') == 'exact_text'
    assert row == original


def test_wrong_pin_or_conflicting_schema_never_guesses_code_semantics():
    row = {'query': 'q', 'positive': 'p', 'negative': None}
    wrong = {**identity(CODESEARCHNET), 'revision': '0' * 40}
    with pytest.raises(ValueError, match='revision lacks'):
        historical_row_view(row, wrong, 'code-alias')
    with pytest.raises(ValueError, match='modalities conflict'):
        historical_row_view({**row, 'modalities': {'positive': 'text'}}, identity(CODESEARCHNET), 'code')


def test_legacy_miracl_missing_alignment_fails_without_verified_source():
    row = {'query': 'q', 'positive': 'p', 'negative': None, 'group': 'miracl:fil-PH:42', 'language': 'fil-PH'}
    with pytest.raises(ValueError, match='verified legacy producer'):
        historical_row_view(row, identity(MIRACL), 'miracl')
    with pytest.raises(ValueError, match='fingerprints are unverified'):
        historical_row_view(row, identity(MIRACL), 'miracl', producer_policy={'policy': 'guess'})
    assert historical_row_view({**row, 'upstream_group': '42'}, identity(MIRACL), 'miracl')['upstream_group'] == '42'


def test_verified_actual_0929_group_and_coordinate_migrate_without_raw_mutation():
    policy = {'policy': 'yat-embedding-src-0929-v1', **VERIFIED_LEGACY_PRODUCERS['yat-embedding-src-0929-v1']}
    row = {'query': 'q', 'positive': 'p', 'negative': 'n',
           'group': 'miracl:fil-PH:42', 'coordinate': 'fil-PH:15'}
    original = copy.deepcopy(row)
    view = historical_row_view(row, identity(MIRACL), 'old_alias', producer_policy=policy)
    assert view['upstream_group'] == '42' and row == original
    text_row = {**row, 'positive': 'e\u0301'}
    text_view = historical_row_view(text_row, identity(MIRACL), 'code', producer_policy=policy)
    assert row_identities(text_view, 'code')['positive'] == row_identities(text_row, 'miracl')['positive']
    index = DevelopmentExclusions({}, frozenset(), {MIRACL: frozenset({'42'})})
    assert index.matches(view, identity(MIRACL), 'old_alias') == 'aligned_group'
    for patch in ({'coordinate': 'en:15'}, {'coordinate': None}, {'language': 'en'},
                  {'group': 'miracl:madeup:42'}, {'group': 'miracl:fil-PH:0042'}, {'group': 'miracl:42'}):
        with pytest.raises(ValueError, match='verified language/id encoding'):
            historical_row_view({**row, **patch}, identity(MIRACL), 'alias', producer_policy=policy)
    with pytest.raises(ValueError, match='fingerprints are unverified'):
        historical_row_view(row, identity(MIRACL), 'alias', producer_policy={**policy, 'producer_sha256': 'a' * 64})
