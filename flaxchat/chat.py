"""Shared, validated generation service for CLI and web chat adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import os

from flax import nnx

from flaxchat.checkpoint import restore_model_from_checkpoint
from flaxchat.common import get_base_dir
from flaxchat.config import FlaxChatConfig, GenerationConfig
from flaxchat.engine import generate_with_cache
from flaxchat.gpt import GPT
from flaxchat.tokenizer import get_tokenizer


class ChatService:
    """One generation contract shared by every user-facing adapter."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prompt_tokens(self, user_text: str, config: GenerationConfig) -> list[int]:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")
        tokens = [
            self.tokenizer.encode_special("<|bos|>"),
            self.tokenizer.encode_special("<|user_start|>"),
            *self.tokenizer.encode(user_text),
            self.tokenizer.encode_special("<|user_end|>"),
            self.tokenizer.encode_special("<|assistant_start|>"),
        ]
        if len(tokens) + config.max_tokens > self.model.config.sequence_len:
            raise ValueError(
                f"prompt ({len(tokens)} tokens) plus generation ({config.max_tokens}) "
                f"exceeds context limit {self.model.config.sequence_len}"
            )
        return tokens

    def generate_tokens(self, user_text: str, config: GenerationConfig) -> list[int]:
        prompt = self.prompt_tokens(user_text, config)
        return generate_with_cache(
            self.model,
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            seed=config.seed,
        )[len(prompt):]

    def stream_text(
        self,
        user_text: str,
        config: GenerationConfig,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        for token in self.generate_tokens(user_text, config):
            if cancelled is not None and cancelled():
                return
            if token == assistant_end:
                return
            yield self.tokenizer.decode([token])

    def generate_text(self, user_text: str, config: GenerationConfig) -> str:
        return "".join(self.stream_text(user_text, config))


def load_chat_service(model_tag: str, checkpoint_type: str = "sft") -> ChatService:
    """Load a checkpoint/tokenizer pair for thin UI adapters."""
    if checkpoint_type not in {"base", "sft", "rl"}:
        raise ValueError("checkpoint_type must be base, sft, or rl")
    tokenizer = get_tokenizer()
    depth = int(model_tag.removeprefix("d"))
    config = FlaxChatConfig.from_depth(
        depth=depth, vocab_size=tokenizer.get_vocab_size()
    )
    model = GPT(config.model, rngs=nnx.Rngs(0))
    checkpoint_dir = os.path.join(
        get_base_dir(), f"{checkpoint_type}_checkpoints", model_tag
    )
    restore_model_from_checkpoint(model, checkpoint_dir)
    return ChatService(model, tokenizer)
