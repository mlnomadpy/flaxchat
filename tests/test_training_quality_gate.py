import math
import pytest
from scripts.validate_training_quality import evaluate


def summary(final=3.):
    return {'model_config': {'vocab_size': 100}, 'initial_validation_loss': 6.,
            'final_validation_loss': final, 'completed_steps': 110, 'checkpoint_step': 110,
            'resolved_config': {'steps': 110}}


def test_improvement_alone_cannot_pass_uniform_baseline():
    from scripts.ci_scope import select_scope
    assert 'tests/test_training_quality_gate.py' in select_scope(['scripts/validate_training_quality.py'])['tests']
    report = evaluate(summary(5.))
    assert report['checks']['heldout_loss_improved']
    assert not report['passed']
    assert report['uniform_loss'] == math.log(100)


def test_good_loss_still_requires_committed_horizon_and_declared_threshold():
    data = summary()
    assert evaluate(data)['passed']
    assert not evaluate(data, max_validation_loss=2.)['passed']
    data['checkpoint_step'] = 100
    assert not evaluate(data)['passed']


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1.])
def test_invalid_losses_fail_closed(invalid):
    report = evaluate(summary(invalid))
    assert not report['passed']
    assert report['final_validation_loss'] is None


@pytest.mark.parametrize('step,stop,every,elapsed,expected', [
    (1, 100, None, 299., False), (1, 100, None, 300., True),
    (2, 100, 2, 0., True), (1, 100, 2, 1000., False),
    (3, 3, None, 0., True), (3, 3, 100, 0., True),
])
def test_checkpoint_cadence_always_commits_clean_stop(step, stop, every, elapsed, expected):
    from scripts.train_gpt2 import checkpoint_due
    assert checkpoint_due(step, stop, every, elapsed, 300.) is expected


def test_standard_tied_gpt_starts_near_uniform_instead_of_overconfident():
    import jax.numpy as jnp
    from flax import nnx
    from flaxchat.gpt import GPT, GPTConfig

    model = GPT(GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=128, vocab_size=1024,
                          sequence_len=16, standard_gpt=True, tie_embeddings=True), rngs=nnx.Rngs(42))
    tokens = jnp.arange(16, dtype=jnp.int32)[None, :]
    loss = float(model(tokens, (tokens + 1) % 1024))
    assert abs(loss - math.log(1024)) < 0.25
    assert model.lm_head is None  # Fix the shared head without adding parameters.
