import jax
import jax.numpy as jnp
import pytest

from flaxchat.rl import centered_advantages, preference_loss
from flaxchat.sft import make_sft_batch


class ConversationTokenizer:
    def render_conversation(self, conversation, max_tokens):
        del conversation
        ids = [1, 2, 3, 4, 5][:max_tokens]
        return ids, [0, 0, 0, 1, 1][:max_tokens]


def test_sft_batch_supervises_only_assistant_tokens():
    inputs, targets = make_sft_batch(
        [{"messages": []}], ConversationTokenizer(), 2, 4, jax.random.key(0)
    )
    assert inputs.shape == targets.shape == (2, 4)
    assert jnp.array_equal(targets[0], jnp.asarray([-1, -1, 4, 5]))


def test_sft_batch_fails_without_examples():
    with pytest.raises(ValueError, match="at least one"):
        make_sft_batch([], ConversationTokenizer(), 1, 4, jax.random.key(0))


def test_preference_objective_is_finite_and_centered():
    logits = jnp.zeros((2, 3, 8))
    targets = jnp.asarray([[1, 2, -1], [1, 3, 4]])
    advantages = centered_advantages([1.0, 0.0])
    assert jnp.array_equal(advantages, jnp.asarray([0.5, -0.5]))
    assert jnp.isfinite(preference_loss(logits, targets, advantages))
