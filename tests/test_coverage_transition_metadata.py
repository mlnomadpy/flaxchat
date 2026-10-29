"""Pure metadata tests: no JAX or model execution."""

from copy import deepcopy
import pytest
from flaxchat.coverage_transition import admit_coverage_transition


def case():
    old = dict(
        coverage_sampling="pool_epoch_v1",
        batch_size=64,
        steps=500000,
        seed=1006,
        learning_rate=0.0001,
        runtime={"jax": "pinned"},
        model={"hidden_size": 768},
        optimizer="adamw",
    )
    parent = dict(
        step=57000,
        model_family="modernbert",
        resolved_config=old,
        tokenizer_identity="tokenizer",
        data_manifest_identity="data",
        source_python_sha256="c" * 64,
        committed_receipt=dict(
            step=57000, manifest_sha256="a" * 64, metadata_sha256="b" * 64
        ),
    )
    new = dict(
        deepcopy(old),
        coverage_sampling="remaining_pool_epoch_v1",
        coverage_transition_step=57000,
        coverage_transition_batch_size=64,
    )
    return dict(
        parent_metadata=parent,
        current_identity=new,
        tokenizer_identity="tokenizer",
        data_manifest_identity="data",
        transition_step=57000,
        parent_manifest_sha256="a" * 64,
        parent_metadata_sha256="b" * 64,
        parent_source_sha256="c" * 64,
    )


def test_admits_only_pinned_transition_without_mutation():
    args = case()
    before = deepcopy(args)
    expected, lineage = admit_coverage_transition(**args)
    assert args == before
    assert expected["resolved_config"] == args["parent_metadata"]["resolved_config"]
    assert expected["source_python_sha256"] == "c" * 64
    assert lineage["physical_gate_required"] is True
    assert lineage["physical_gate_passed"] is False
    expected["resolved_config"]["model"]["hidden_size"] = 1
    assert args == before


@pytest.mark.parametrize(
    "key", ["parent_manifest_sha256", "parent_metadata_sha256", "parent_source_sha256"]
)
@pytest.mark.parametrize("bad", ["", "A" * 64, "g" * 64, "a" * 63, None, "d" * 64])
def test_rejects_bad_or_mismatched_pins(key, bad):
    args = case()
    args[key] = bad
    with pytest.raises(ValueError):
        admit_coverage_transition(**args)


@pytest.mark.parametrize(
    "key",
    ["batch_size", "steps", "seed", "learning_rate", "runtime", "model", "optimizer"],
)
def test_rejects_each_immutable_config_change(key):
    args = case()
    args["current_identity"][key] = "changed"
    with pytest.raises(ValueError):
        admit_coverage_transition(**args)


@pytest.mark.parametrize(
    "key,value",
    [
        ("coverage_sampling", "other"),
        ("coverage_transition_step", 57001),
        ("coverage_transition_batch_size", 65),
        ("batch_size", 64.0),
        ("new_field", 1),
    ],
)
def test_rejects_inexact_transition(key, value):
    args = case()
    args["current_identity"][key] = value
    with pytest.raises(ValueError):
        admit_coverage_transition(**args)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tokenizer_identity", "other"),
        ("data_manifest_identity", "other"),
        ("transition_step", 57001),
        ("transition_step", True),
        ("transition_step", -1),
    ],
)
def test_rejects_identity_or_step_changes(key, value):
    args = case()
    args[key] = value
    with pytest.raises(ValueError):
        admit_coverage_transition(**args)


@pytest.mark.parametrize(
    "mutation",
    [
        "receipt_missing",
        "receipt_step",
        "parent_step",
        "family",
        "sampler",
        "already_transitioned",
    ],
)
def test_rejects_invalid_parent(mutation):
    args = case()
    parent = args["parent_metadata"]
    if mutation == "receipt_missing":
        parent.pop("committed_receipt")
    if mutation == "receipt_step":
        parent["committed_receipt"]["step"] = 1
    if mutation == "parent_step":
        parent["step"] = 1
    if mutation == "family":
        parent["model_family"] = "other"
    if mutation == "sampler":
        parent["resolved_config"]["coverage_sampling"] = "other"
    if mutation == "already_transitioned":
        parent["resolved_config"]["coverage_transition_step"] = 1
    with pytest.raises(ValueError):
        admit_coverage_transition(**args)
