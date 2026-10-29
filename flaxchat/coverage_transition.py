"""Metadata-only admission for the one supported coverage sampler migration.

The caller must use authenticated ``read_committed_metadata(...,
include_receipt=True)`` output and qualify the admitted transition on a TPU.
Admission never authorizes an arbitrary configuration or source migration.
"""

from copy import deepcopy
import re


def _pin(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 pin")
    return value


def admit_coverage_transition(
    parent_metadata,
    current_identity,
    tokenizer_identity,
    data_manifest_identity,
    transition_step,
    parent_manifest_sha256,
    parent_metadata_sha256,
    parent_source_sha256,
):
    """Return ``(expected_identity, lineage)`` for a pinned full-state restore.

    ``expected_identity`` describes the *old* checkpoint. ``lineage`` records
    the narrowly allowed sampler change and requires separate physical TPU
    qualification; its ``physical_gate_passed`` is always False here.
    """
    manifest_pin = _pin(parent_manifest_sha256, "parent_manifest_sha256")
    metadata_pin = _pin(parent_metadata_sha256, "parent_metadata_sha256")
    source_pin = _pin(parent_source_sha256, "parent_source_sha256")
    if not isinstance(parent_metadata, dict) or not isinstance(current_identity, dict):
        raise ValueError("Parent metadata and current identity must be dictionaries")
    receipt = parent_metadata.get("committed_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("Authenticated committed receipt is required")
    if (
        receipt.get("manifest_sha256") != manifest_pin
        or receipt.get("metadata_sha256") != metadata_pin
        or parent_metadata.get("source_python_sha256") != source_pin
    ):
        raise ValueError("Parent receipt or source pin mismatch")
    if (
        type(transition_step) is not int
        or transition_step < 0
        or type(parent_metadata.get("step")) is not int
        or parent_metadata["step"] != transition_step
        or type(receipt.get("step")) is not int
        or receipt["step"] != transition_step
    ):
        raise ValueError("Transition step must match the authenticated committed step")
    if parent_metadata.get("model_family") != "modernbert":
        raise ValueError("Only modernbert coverage transitions are supported")
    old = parent_metadata.get("resolved_config")
    if not isinstance(old, dict) or old.get("coverage_sampling") != "pool_epoch_v1":
        raise ValueError("Parent must use pool_epoch_v1")
    if "coverage_transition_step" in old or "coverage_transition_batch_size" in old:
        raise ValueError("Parent already contains a coverage transition")
    batch = old.get("batch_size")
    if type(batch) is not int or batch <= 0:
        raise ValueError("Parent batch size must be a positive integer")
    required = deepcopy(old)
    required.update(
        coverage_sampling="remaining_pool_epoch_v1",
        coverage_transition_step=transition_step,
        coverage_transition_batch_size=batch,
    )

    # Python considers True == 1 and 1.0 == 1; identities must retain types too.
    def exact(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                exact(left[k], right[k]) for k in left
            )
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                exact(a, b) for a, b in zip(left, right, strict=True)
            )
        return left == right

    if not exact(current_identity, required):
        raise ValueError(
            "Only the declared coverage sampler transition may change configuration"
        )
    if (
        parent_metadata.get("tokenizer_identity") != tokenizer_identity
        or parent_metadata.get("data_manifest_identity") != data_manifest_identity
        or not isinstance(tokenizer_identity, str)
        or not tokenizer_identity
        or not isinstance(data_manifest_identity, str)
        or not data_manifest_identity
    ):
        raise ValueError("Tokenizer and data manifest identities must remain unchanged")
    expected = dict(
        resolved_config=deepcopy(old),
        tokenizer=tokenizer_identity,
        data_manifest=data_manifest_identity,
        source_python_sha256=source_pin,
        manifest_sha256=manifest_pin,
    )
    lineage = dict(
        schema_version=1,
        kind="coverage_sampler_transition",
        from_sampler="pool_epoch_v1",
        to_sampler="remaining_pool_epoch_v1",
        transition_step=transition_step,
        transition_batch_size=batch,
        parent_manifest_sha256=manifest_pin,
        parent_metadata_sha256=metadata_pin,
        parent_source_sha256=source_pin,
        physical_gate_required=True,
        physical_gate_passed=False,
    )
    return expected, lineage
