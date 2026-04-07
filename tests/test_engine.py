"""
Tests for the inference engine (both simple and KV-cached).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import GPTConfig
from flaxchat.engine import generate, generate_with_cache, generate_fast, generate_speculative, use_calculator
from flaxchat.execution import execute_code


class TestGenerate:
    def test_basic_generation(self, tiny_model, tiny_config):
        prompt = [0, 1, 2, 3]
        output = generate(tiny_model, prompt, max_tokens=8, temperature=1.0, seed=42)
        assert len(output) == len(prompt) + 8
        for t in output:
            assert 0 <= t < tiny_config.vocab_size

    def test_greedy_deterministic(self, tiny_model):
        prompt = [0, 1, 2]
        out1 = generate(tiny_model, prompt, max_tokens=5, temperature=0)
        out2 = generate(tiny_model, prompt, max_tokens=5, temperature=0)
        assert out1 == out2

    def test_top_k(self, tiny_model, tiny_config):
        prompt = [0, 1, 2]
        output = generate(tiny_model, prompt, max_tokens=5, temperature=1.0, top_k=5, seed=42)
        assert len(output) == len(prompt) + 5

    def test_single_token_prompt(self, tiny_model):
        output = generate(tiny_model, [0], max_tokens=3, temperature=0)
        assert len(output) == 4

    def test_shakespeare_generation(self, tiny_model, shakespeare_text):
        prompt = list(shakespeare_text[:20].encode("utf-8"))
        prompt = [t % tiny_model.config.vocab_size for t in prompt]
        output = generate(tiny_model, prompt, max_tokens=10, temperature=0.8, seed=42)
        assert len(output) == len(prompt) + 10


class TestGenerateWithCache:
    def test_basic_cached_generation(self, tiny_model, tiny_config):
        prompt = [0, 1, 2, 3]
        output = generate_with_cache(tiny_model, prompt, max_tokens=8, temperature=1.0, seed=42)
        assert len(output) == len(prompt) + 8
        for t in output:
            assert 0 <= t < tiny_config.vocab_size

    def test_greedy_deterministic(self, tiny_model):
        prompt = [0, 1, 2]
        out1 = generate_with_cache(tiny_model, prompt, max_tokens=5, temperature=0)
        out2 = generate_with_cache(tiny_model, prompt, max_tokens=5, temperature=0)
        assert out1 == out2

    def test_matches_simple_generate(self, tiny_model):
        """KV-cached generation should produce identical output to simple generation (greedy)."""
        prompt = [0, 1, 2, 3, 4]
        out_simple = generate(tiny_model, prompt, max_tokens=8, temperature=0)
        out_cached = generate_with_cache(tiny_model, prompt, max_tokens=8, temperature=0)
        assert out_simple == out_cached, f"Mismatch:\nsimple: {out_simple}\ncached: {out_cached}"

    def test_top_k_cached(self, tiny_model, tiny_config):
        prompt = [0, 1, 2]
        output = generate_with_cache(tiny_model, prompt, max_tokens=5, temperature=1.0, top_k=5, seed=42)
        assert len(output) == len(prompt) + 5




class TestGenerateFast:
    def test_fast_generation_basic(self, tiny_model, tiny_config):
        """generate_fast produces correct length output with valid token ids."""
        prompt = [0, 1, 2, 3]
        output = generate_fast(tiny_model, prompt, max_tokens=8, temperature=1.0,
                               top_k=5, seed=42, eos_token=-1)
        assert len(output) == len(prompt) + 8
        for t in output:
            assert 0 <= t < tiny_config.vocab_size

    def test_fast_greedy_deterministic(self, tiny_model):
        """Greedy decoding with generate_fast is deterministic across runs."""
        prompt = [0, 1, 2]
        out1 = generate_fast(tiny_model, prompt, max_tokens=5, temperature=0,
                             top_k=40, eos_token=-1)
        out2 = generate_fast(tiny_model, prompt, max_tokens=5, temperature=0,
                             top_k=40, eos_token=-1)
        assert out1 == out2

    def test_fast_matches_cached(self, tiny_model):
        """Greedy output from generate_fast should match generate_with_cache."""
        prompt = [0, 1, 2, 3, 4]
        out_cached = generate_with_cache(tiny_model, prompt, max_tokens=8,
                                         temperature=0, top_k=40)
        out_fast = generate_fast(tiny_model, prompt, max_tokens=8,
                                 temperature=0, top_k=40, eos_token=-1)
        assert out_fast == out_cached, (
            f"Mismatch:\ncached: {out_cached}\nfast:   {out_fast}"
        )


class TestSpeculativeDecoding:
    @pytest.fixture
    def draft_config(self):
        """Even smaller config for draft model (1 layer vs 2)."""
        return GPTConfig(
            sequence_len=64,
            vocab_size=256,  # must match tiny_config
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="S",
        )

    @pytest.fixture
    def draft_model(self, draft_config):
        return GPT(draft_config, rngs=nnx.Rngs(1))

    def test_speculative_basic(self, tiny_model, draft_model, tiny_config):
        """Speculative decoding produces correct output length with valid tokens."""
        prompt = [0, 1, 2, 3]
        max_tokens = 12
        output = generate_speculative(
            tiny_model, draft_model, prompt,
            max_tokens=max_tokens, temperature=1.0, top_k=40, seed=42, draft_steps=4,
        )
        assert len(output) == len(prompt) + max_tokens
        for t in output:
            assert 0 <= t < tiny_config.vocab_size

    def test_speculative_greedy_matches(self, tiny_model, tiny_config):
        """With temp=0, speculative decoding (using same model as draft) matches generate_with_cache."""
        prompt = [0, 1, 2, 3, 4]
        max_tokens = 8
        # Use the same model as both main and draft — greedy output must be identical
        out_cached = generate_with_cache(
            tiny_model, prompt, max_tokens=max_tokens, temperature=0, top_k=40,
        )
        out_spec = generate_speculative(
            tiny_model, tiny_model, prompt,
            max_tokens=max_tokens, temperature=0, top_k=40, seed=42, draft_steps=4,
        )
        assert out_spec == out_cached, (
            f"Mismatch:\ncached:      {out_cached}\nspeculative: {out_spec}"
        )


class TestCalculator:
    def test_basic_math(self):
        assert use_calculator("2 + 3") == 5
        assert use_calculator("10 * 5") == 50
        assert use_calculator("100 / 4") == 25.0

    def test_string_count(self):
        assert use_calculator("'hello'.count('l')") == 2
        assert use_calculator("'strawberry'.count('r')") == 3

    def test_rejects_dangerous(self):
        assert use_calculator("__import__('os')") is None
        assert use_calculator("exec('bad')") is None

    def test_rejects_power(self):
        assert use_calculator("2**100") is None

    def test_comma_removal(self):
        assert use_calculator("1,000 + 2,000") == 3000


class TestExecuteCode:
    def test_execute_code_tool(self):
        result = execute_code("print(2+2)")
        assert result.success is True
        assert result.stdout == "4\n"
