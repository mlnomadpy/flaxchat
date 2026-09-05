import json
from pathlib import Path

import pytest

from flaxchat.artifact import artifact_checksums, resolve_artifact_path, verify_artifact
from flaxchat.chat import GenerationConfig, load_chat_service_from_artifact


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "examples" / "tinystories-v0.1.1"


def test_published_artifact_is_intact_and_generates_deterministically():
    expected = json.loads((ARTIFACT / "SHA256SUMS.json").read_text(encoding="utf-8"))
    assert artifact_checksums(ARTIFACT) == expected
    manifest = verify_artifact(ARTIFACT)
    assert manifest["release_compatibility"] == {
        "project": "flaxchat",
        "published": False,
        "required_tag": "v0.1.1",
        "version": "0.1.1",
    }
    service = load_chat_service_from_artifact(ARTIFACT)
    config = GenerationConfig(max_tokens=4, temperature=0, seed=42)
    assert service.generate_text("Once upon a time", config) == " he4it("


def test_artifact_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact_path(tmp_path, "../checkpoint")
    with pytest.raises(ValueError, match="relative"):
        resolve_artifact_path(tmp_path, tmp_path.as_posix())


def test_artifact_checksum_verification_rejects_tampering(tmp_path):
    manifest = {
        "artifacts": {},
        "format_version": 1,
        "model_config": {},
        "release_compatibility": {
            "project": "flaxchat",
            "published": False,
            "required_tag": "v0.1.1",
            "version": "0.1.1",
        },
        "resolved_config": {},
        "source_revision": "a" * 40,
        "tokenizer_sha256": "b" * 64,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "payload").write_text("original", encoding="utf-8")
    (tmp_path / "SHA256SUMS.json").write_text(
        json.dumps(artifact_checksums(tmp_path)), encoding="utf-8"
    )
    assert verify_artifact(tmp_path) == manifest
    (tmp_path / "payload").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifact(tmp_path)
