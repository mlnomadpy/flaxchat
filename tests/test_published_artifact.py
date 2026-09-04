import json
from pathlib import Path

from flaxchat.chat import GenerationConfig, load_chat_service
from scripts.verify_artifact import checksums


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "examples" / "tinystories-v0.1.1"


def test_published_artifact_is_intact_and_generates_deterministically():
    expected = json.loads((ARTIFACT / "SHA256SUMS.json").read_text())
    assert checksums(ARTIFACT) == expected
    service = load_chat_service(
        "manifest",
        "base",
        checkpoint_path=str(ARTIFACT / "checkpoint"),
        tokenizer_path=str(ARTIFACT / "tokenizer"),
    )
    config = GenerationConfig(max_tokens=4, temperature=0, seed=42)
    assert service.generate_text("Once upon a time", config) == " he4it("
