import pytest

from flaxchat.chat import ChatService, GenerationConfig, load_chat_service
from flaxchat.tokenizer import ByteTokenizer


class TinyTokenizer:
    specials = {
        "<|bos|>": 0, "<|user_start|>": 1, "<|user_end|>": 2,
        "<|assistant_start|>": 3, "<|assistant_end|>": 4,
    }

    def encode_special(self, token):
        return self.specials[token]

    def encode(self, text):
        return [5 + ord(char) % 20 for char in text]

    def decode(self, tokens):
        return "".join(chr(65 + token % 26) for token in tokens)


def test_generation_is_seed_deterministic(tiny_model):
    service = ChatService(tiny_model, TinyTokenizer())
    config = GenerationConfig(max_tokens=3, temperature=0.7, seed=123)
    assert service.generate_tokens("hi", config) == service.generate_tokens("hi", config)


def test_context_limit_fails_before_generation(tiny_model):
    service = ChatService(tiny_model, TinyTokenizer())
    with pytest.raises(ValueError, match="exceeds context limit"):
        service.generate_tokens("x" * 60, GenerationConfig(max_tokens=8))


def test_empty_prompt_is_rejected(tiny_model):
    service = ChatService(tiny_model, TinyTokenizer())
    with pytest.raises(ValueError, match="non-empty"):
        service.prompt_tokens("  ", GenerationConfig(max_tokens=1))


def test_generate_text_stops_at_assistant_end(tiny_model, monkeypatch):
    service = ChatService(tiny_model, TinyTokenizer())
    monkeypatch.setattr(service, "generate_tokens", lambda *_, **__: [7, 4, 8])
    assert service.generate_text("hi", GenerationConfig(max_tokens=3)) == "H"


def test_stream_honors_cancellation_before_decode(tiny_model):
    service = ChatService(tiny_model, TinyTokenizer())
    chunks = list(service.stream_text(
        "hello", GenerationConfig(max_tokens=4, temperature=0),
        cancelled=lambda: True,
    ))
    assert chunks == []


def test_stream_preserves_byte_tokenizer_unicode(tiny_model, monkeypatch):
    tokenizer = ByteTokenizer()
    service = ChatService(tiny_model, tokenizer)
    tokens = tokenizer.encode("Hi 👋") + [tokenizer.encode_special("<|assistant_end|>")]
    monkeypatch.setattr(service, "generate_tokens", lambda *_, **__: tokens)
    assert "".join(service.stream_text(
        "hello", GenerationConfig(max_tokens=len(tokens), temperature=0)
    )) == "Hi 👋"


def test_loader_rejects_unknown_checkpoint_type_before_io():
    with pytest.raises(ValueError, match="base, sft, or rl"):
        load_chat_service("d4", "unknown")
