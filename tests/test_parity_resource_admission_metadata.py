"""Host capacity admission only: no model execution or cloud calls."""
import json
from unittest.mock import patch

import pytest

from scripts.run_yat_parity_campaign import resource_admission, run_campaign, verify_host_capacity


def models(tmp_path, maximum=32768, width=768, depth=22):
    source, target = tmp_path / 'source', tmp_path / 'target'
    for folder in (source, target):
        folder.mkdir()
        (folder / 'model.safetensors').write_bytes(b'fixture')
    (target / 'config.json').write_text(json.dumps(dict(hidden_size=width,
        max_position_embeddings=maximum, num_hidden_layers=depth)))
    return source, target


def budgets(amount=1024**4):
    return dict(host_ram_budget_bytes=amount, scratch_budget_bytes=amount, evidence_budget_bytes=amount)


def test_maximum_context_estimate_covers_retained_matrix_and_comparison(tmp_path):
    source, target = models(tmp_path)
    result = resource_admission(source, target, **budgets())
    largest = 72 * 32768 * 768
    assert result['basis']['retained_sequence_tensors'] == 4
    assert result['cases'][-1]['output_bytes_per_backend'] == 4 * (4 * largest + 72 * 768)
    assert result['required']['host_ram_bytes'] == 2 * 1024**3 + 3 * len(b'fixture') + 3 * 4 * (4 * largest + 72 * 768) + 64 * largest
    assert len(result['cases']) == 8
    assert result['required']['evidence_bytes'] > 2 * result['cases'][-1]['output_bytes_per_backend']
    assert result['basis']['physical_peak_measured'] is False


@pytest.mark.parametrize('key', ['host_ram_budget_bytes', 'scratch_budget_bytes', 'evidence_budget_bytes'])
def test_insufficient_budget_fails_before_identity_or_model_and_creates_no_evidence(tmp_path, key):
    source, target = models(tmp_path)
    configured = budgets()
    configured[key] = 1024**3
    evidence = tmp_path / 'evidence'
    with patch('scripts.run_yat_parity_campaign.subprocess.run') as probe, patch('scripts.run_yat_parity_campaign.run_case') as worker:
        with pytest.raises(ValueError, match='budgets insufficient'):
            run_campaign(source, target, 'jax', 'torch', evidence, **configured)
        probe.assert_not_called()
        worker.assert_not_called()
    assert not evidence.exists()


def test_missing_budget_or_width_rejected_and_preflight_preserves_full_coverage(tmp_path):
    source, target = models(tmp_path, maximum=512, width=4, depth=1)
    with pytest.raises(ValueError, match='Explicit positive'):
        run_campaign(source, target, 'jax', 'torch', tmp_path/'evidence')
    with patch('scripts.run_yat_parity_campaign.subprocess.run') as probe:
        result = run_campaign(source, target, 'jax', 'torch', tmp_path/'evidence', **budgets(), preflight_only=True)
        probe.assert_not_called()
    assert len(result['resource_admission']['cases']) == 6
    assert result['resource_admission']['basis']['retained_sequence_tensors'] == 3
    assert result['full_matrix_qualified'] is False
    assert not (tmp_path/'evidence').exists()
    (target/'config.json').write_text(json.dumps(dict(num_hidden_layers=1, max_position_embeddings=512)))
    with pytest.raises(ValueError, match='hidden_size'):
        resource_admission(source, target, **budgets())


@pytest.mark.parametrize('invalid', [True, 0, -1, 1.5, float('inf')])
def test_budget_types_fail_closed(tmp_path, invalid):
    source, target = models(tmp_path)
    configured = budgets()
    configured['host_ram_budget_bytes'] = invalid
    with pytest.raises(ValueError, match='positive integer'):
        resource_admission(source, target, **configured)


def test_actual_evidence_bytes_and_free_scratch_fail_closed(tmp_path):
    source, target = models(tmp_path, maximum=512, width=4)
    admission = resource_admission(source, target, **budgets())
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence/'oversized.bin').write_bytes(b'0123456789')
    admission['budgets']['evidence_bytes'] = 9
    with pytest.raises(ValueError, match='Actual retained'):
        verify_host_capacity(evidence, admission)
    admission['budgets']['evidence_bytes'] = 1024**4
    with patch('scripts.run_yat_parity_campaign.shutil.disk_usage') as usage:
        usage.return_value.free = 0
        with pytest.raises(ValueError, match='free scratch'):
            verify_host_capacity(evidence, admission)
