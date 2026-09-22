import copy
import hashlib
import struct

import pytest

from scripts.multihost_acceptance import validate_records


def _record(index: int) -> dict:
    global_order = list(range(8))
    return {
        "passed": True,
        "source_revision": "a" * 40,
        "source_python_sha256": "e" * 64,
        "hostname": f"worker-{index}",
        "environment_sha256": "b" * 64,
        "topology": {
            "backend": "tpu",
            "local_device_count": 8,
            "process_index": index,
            "process_count": 2,
            "device_count": 16,
        },
        "data": {
            "local_indices": global_order[index * 4:(index + 1) * 4],
            "global_order_sha256": hashlib.sha256(
                struct.pack(f"<{len(global_order)}i", *global_order)
            ).hexdigest(),
        },
        "training": {"loss": 39.375, "gradient": -52.5, "updated_parameter": .5525},
    }


def test_summary_requires_disjoint_complete_matching_workers():
    summary = validate_records([_record(0), _record(1)], cost_usd=1.25)
    assert summary["status"] == "probe_passed"
    assert summary["topology"] == {
        "process_count": 2,
        "device_count": 16,
        "single_host": False,
    }
    assert summary["data"]["global_indices"] == list(range(8))
    assert summary["cost_usd"] == 1.25


@pytest.mark.parametrize(
    "mutation", ["overlap", "duplicate", "loss", "nonfinite", "revision", "digest", "failed"]
)
def test_summary_fails_closed_on_inconsistent_evidence(mutation):
    records = [_record(0), _record(1)]
    if mutation == "overlap":
        records[1]["data"]["local_indices"][0] = 3
    elif mutation == "duplicate":
        records[1]["data"]["local_indices"][1] = 4
    elif mutation == "loss":
        records[1]["training"]["loss"] = 2.0
    elif mutation == "nonfinite":
        records[0]["training"]["loss"] = float("inf")
        records[1]["training"]["loss"] = float("inf")
    elif mutation == "revision":
        records[1]["source_revision"] = "c" * 40
    elif mutation == "digest":
        records[0]["data"]["global_order_sha256"] = "d" * 64
        records[1]["data"]["global_order_sha256"] = "d" * 64
    else:
        records[1]["passed"] = False
    with pytest.raises(ValueError):
        validate_records(copy.deepcopy(records), cost_usd=1.0)


def test_summary_rejects_single_host_and_invalid_cost():
    with pytest.raises(ValueError, match="at least two"):
        validate_records([_record(0)], cost_usd=0.1)
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_records([_record(0), _record(1)], cost_usd=float("nan"))


@pytest.mark.parametrize('mutation', ['cpu', 'same_host', 'wrong_gradient', 'wrong_devices', 'empty', 'bad_sha'])
def test_physical_acceptance_rejects_false_positive_evidence(mutation):
    records = [_record(0), _record(1)]
    if mutation == 'cpu':
        for record in records:
            record['topology']['backend'] = 'cpu'
    elif mutation == 'same_host':
        records[1]['hostname'] = records[0]['hostname']
    elif mutation == 'wrong_gradient':
        for record in records:
            record['training']['gradient'] = 0.
    elif mutation == 'wrong_devices':
        records[0]['topology']['local_device_count'] = 1
    elif mutation == 'empty':
        records[0]['data']['local_indices'] = []
    else:
        for record in records:
            record['source_revision'] = 'main'
    with pytest.raises(ValueError):
        validate_records(records, cost_usd=0.)
