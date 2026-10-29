"""Metadata-only lineage admission for the first contrastive GOAT stage.

The allowlist is independently retained MLM release evidence, not a caller flag.
It establishes zero earlier contrastive stages for this one known parent; it
makes no claim of unseen MLM text or semantic/translation independence.
"""

from pathlib import Path
import hashlib
import json

# Release379079 and its immutable committed lineage were authenticated before
# this new objective was requested. Do not infer MLM-only history from a filename.
MLM_ONLY_PARENTS = {
    "b9ee8e39171a431c1944f9919931dec6a6b8857664f89aaa64679e7637577281": {
        "step": 379079,
        "export_sha256": "081b2ca33c33096198fbd7c8ac5884962955b9000ccaf37f3ec8a73f4b5eb36d",
        "metadata": "98af89f076e0939ee5bfbb3465a2f6a085b1587b11f865d3d822a9937943b7b0",
        "manifest_identity": "941efae8b2bd3b91e578d32f27c58ec7a11e8f6095f54a3c97a8ead56d2dbf4b",
        "source": "fb1c824d7b6db719e7b47c46efb94410ddf705a14f1fd69badb28029cf3ffd02",
    }
}


def validate_mlm_only_parent(root, parent_hashes):
    """Rehash real metadata/config/tokenizer and bind the known parent digest.

    Weight byte authentication belongs to the caller's parent admission. This
    function does not construct a model or assert physical restoration.
    """
    from scripts.release_contract import canonical_hash, checkpoint_identity

    root = Path(root)
    if set(parent_hashes) != {"model.safetensors", "config.json", "tokenizer.json"}:
        raise ValueError("Complete parent file identity required")
    expected = MLM_ONLY_PARENTS.get(parent_hashes["model.safetensors"])
    if expected is None:
        raise ValueError(
            "No independently authenticated MLM-only lineage for this parent"
        )
    docs, digests = {}, {}
    for name in [
        "config.json",
        "tokenizer.json",
        "checkpoint-metadata.json",
        "checkpoint-manifest.json",
        "export.json",
    ]:
        p = root / name
        maximum = 32 * 1024**2 if name == "tokenizer.json" else 8 * 1024**2
        if not p.is_file() or p.is_symlink() or p.stat().st_size > maximum:
            raise ValueError("Bounded regular MLM lineage artifacts required")
        with p.open("rb") as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("MLM lineage artifact exceeded bound")
        digests[name] = hashlib.sha256(raw).hexdigest()
        docs[name] = json.loads(raw)
    export, metadata, manifest = [
        docs[n]
        for n in ["export.json", "checkpoint-metadata.json", "checkpoint-manifest.json"]
    ]
    artifacts = dict(digests)
    artifacts.pop("export.json")
    artifacts["model.safetensors"] = parent_hashes["model.safetensors"]
    if (
        digests["export.json"] != expected["export_sha256"]
        or export.get("format") != "flaxchat-authenticated-mlm-export-v1"
        or export.get("source_model_family") != "modernbert"
        or metadata.get("model_family") != "modernbert"
        or export.get("artifacts_sha256") != artifacts
        or export.get("sha256") != parent_hashes["model.safetensors"]
        or any(
            digests[n] != parent_hashes[n] for n in ["config.json", "tokenizer.json"]
        )
        or canonical_hash(metadata) != expected["metadata"]
        or manifest.get("metadata_sha256") != expected["metadata"]
        or manifest.get("identity_sha256") != expected["manifest_identity"]
        or canonical_hash(manifest.get("identity")) != expected["manifest_identity"]
        or manifest.get("identity") != checkpoint_identity(metadata)
        or export.get("source_checkpoint_metadata_sha256") != expected["metadata"]
        or export.get("source_checkpoint_manifest_identity_sha256")
        != expected["manifest_identity"]
        or export.get("source_checkpoint_step") != expected["step"]
        or manifest.get("step") != expected["step"]
        or metadata.get("source_python_sha256") != expected["source"]
        or metadata.get("resolved_config", {}).get("encoder") != docs["config.json"]
        or metadata.get("resolved_config", {}).get("contrastive_objective") is not None
        or metadata.get("resolved_config", {}).get("mask_probability") != 0.15
        or metadata.get("tokenizer_identity") != parent_hashes["tokenizer.json"]
        or export.get("tokenizer_identity") != parent_hashes["tokenizer.json"]
    ):
        raise ValueError("MLM-only parent lineage identity mismatch")
    return {
        "format": "authenticated-mlm-only-parent-v1",
        "step": expected["step"],
        "parent_files_sha256": dict(parent_hashes),
        "lineage_files_sha256": digests,
        "known_contrastive_stages": [],
        "model_execution": False,
    }
