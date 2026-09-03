import pytest

from flaxchat.chat import ChatService, GenerationConfig


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


def test_stream_honors_cancellation(tiny_model):
    service = ChatService(tiny_model, TinyTokenizer())
    calls = iter((False, True))
    chunks = list(service.stream_text(
        "hello", GenerationConfig(max_tokens=4, temperature=0),
        cancelled=lambda: next(calls),
    ))
    assert len(chunks) == 1
