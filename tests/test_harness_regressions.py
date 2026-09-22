"""Regression tests for the September harness audit."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.gpt import GPT
from flaxchat.common import COMPUTE_DTYPE
from flaxchat.optim import muon, setup_optimizer


def test_muon_uses_checkpointed_schedule_count():
    params = {'w': jnp.ones((4, 8))}
    grads = {'w': jnp.arange(32, dtype=jnp.float32).reshape(4, 8)}
    scheduled = muon(lambda count: jnp.where(count == 0, 0., .01))
    state = scheduled.init(params)
    first, state = scheduled.update(grads, state, params)
    np.testing.assert_array_equal(first['w'], 0)
    updates, _ = scheduled.update(grads, state, params)
    reference, _ = muon(.01).update(grads, state, params)
    np.testing.assert_array_equal(updates['w'], reference['w'])
    assert np.any(np.asarray(updates['w']) != 0)


def test_zero_schedule_freezes_every_parameter(tiny_model, tiny_config):
    optimizer = setup_optimizer(tiny_model, FlaxChatConfig(model=tiny_config), lr_schedule_fn=lambda _: 0.)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(tiny_model, nnx.Param))
    grads = jax.tree.map(jnp.ones_like, nnx.state(tiny_model, nnx.Param))
    optimizer.update(tiny_model, grads)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(nnx.state(tiny_model, nnx.Param)), strict=False):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('standard', [False, True])
@pytest.mark.parametrize('remat', [False, True])
def test_scan_matches_loop_values_and_gradients(standard, remat):
    config = GPTConfig(sequence_len=8, vocab_size=31, n_layer=2, n_head=2, n_kv_head=2,
                       n_embd=32, standard_gpt=standard, use_remat=remat)
    loop = GPT(config, rngs=nnx.Rngs(7))
    scan = GPT(replace(config, use_scan=True), rngs=nnx.Rngs(7))
    x = jnp.arange(16).reshape(2, 8) % 31
    loss_a, grad_a = nnx.value_and_grad(lambda m: m(x, x))(loop)
    loss_b, grad_b = nnx.value_and_grad(lambda m: m(x, x))(scan)
    # BF16 scan fusion changes rounding even though the final loss is FP32.
    np.testing.assert_allclose(loss_a, loss_b,
                               atol=1e-6 if COMPUTE_DTYPE == jnp.float32 else 5e-5)
    for a, b in zip(jax.tree.leaves(grad_a), jax.tree.leaves(grad_b), strict=False):
        if COMPUTE_DTYPE == jnp.float32:
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
        else:
            # Compare each leaf's norm rather than dividing by individual
            # near-zero gradients. BF16 fusion/reduction order is not exact.
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            error = np.linalg.norm(a - b)
            scale = max(np.linalg.norm(a), np.linalg.norm(b))
            print(f'BF16 scan gradient leaf relative L2 error: {error / max(scale, 1e-30):.8g}')
            # Retain the original BF16 absolute allowance for leaves whose
            # entire gradient is near zero (e.g. normalization scale symmetry).
            assert np.isfinite(error) and error <= .02 * scale + 5e-5


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('all_masked', [False, True])
def test_chunked_loss_matches_reference_and_gradients(tied, all_masked):
    config = GPTConfig(sequence_len=8, vocab_size=31, n_layer=1, n_head=2, n_kv_head=2,
                       n_embd=32, standard_gpt=True, tie_embeddings=tied)
    full = GPT(config, rngs=nnx.Rngs(7))
    chunked = GPT(replace(config, loss_chunk_size=3), rngs=nnx.Rngs(7))
    x = jnp.arange(16).reshape(2, 8) % 31
    targets = jnp.full_like(x, -1) if all_masked else x.at[0, :3].set(-1)
    la, ga = nnx.value_and_grad(lambda m: m(x, targets))(full)
    lb, gb = nnx.value_and_grad(lambda m: m(x, targets))(chunked)
    np.testing.assert_allclose(la, lb, atol=1e-6)
    for a, b in zip(jax.tree.leaves(ga), jax.tree.leaves(gb), strict=False):
        np.testing.assert_allclose(a, b, atol=2e-6 if COMPUTE_DTYPE == jnp.float32 else 5e-5,
                                   rtol=2e-5 if COMPUTE_DTYPE == jnp.float32 else .02)


def test_all_master_parameters_are_fp32(tiny_model):
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(nnx.state(tiny_model, nnx.Param)))


def test_v5p_runtime_alias_is_not_v5e():
    from flaxchat.common import get_peak_flops
    assert get_peak_flops('TPU v5') == 459e12
    assert get_peak_flops('TPU v5 lite') == 197e12


def test_winogrande_scores_suffix_not_option(monkeypatch):
    from flaxchat import eval as evaluation
    item = evaluation.normalize_core_item({'sentence': 'The trophy did not fit because _ was big.',
              'option1': 'it', 'option2': 'the suitcase', 'answer': '1'})
    assert item['contexts'] == ['The trophy did not fit because it', 'The trophy did not fit because the suitcase']
    assert item['continuation'] == ' was big.'
    class Tokenizer:
        def get_bos_token_id(self): return 0
        def __call__(self, texts, prepend): return [[prepend] + list(text.encode()) for text in texts]
    seen = []
    def forward(model, tokens):
        seen.append(np.asarray(tokens))
        return jnp.ones(tokens.shape), tokens
    monkeypatch.setattr(evaluation, 'forward_model', forward)
    _, record = evaluation.evaluate_example_partial(None, Tokenizer(), item, [], '\n', True)
    assert record['scores'] == [len(' was big.')] * 2
    assert record['scoring'] == 'suffix_sum_nll'


@pytest.mark.parametrize('stage', ['sft', 'rl'])
def test_nonfinite_finetuning_preserves_optimizer_and_model(stage):
    from flaxchat import sft, rl
    class BadModel(nnx.Module):
        def __init__(self): self.weight = nnx.Param(jnp.ones((2, 2)))
        def __call__(self, inputs, targets=None):
            logits = jnp.broadcast_to(self.weight[0] * jnp.nan, (*inputs.shape, 2))
            return logits if targets is None else logits.sum()
    model = BadModel()
    optimizer = nnx.Optimizer(model, optax.adamw(.01), wrt=nnx.Param)
    before = jax.tree.map(lambda a: np.asarray(a).copy(), nnx.state(optimizer))
    x = jnp.zeros((1, 2), jnp.int32)
    if stage == 'sft':
        sft.train_step(model, optimizer, x, x)
    else:
        rl.train_step(model, optimizer, x, x, jnp.ones(1))
    np.testing.assert_array_equal(model.weight[...], 1)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(nnx.state(optimizer)), strict=False):
        np.testing.assert_array_equal(a, b)


def test_prefetch_eof_can_stop_with_full_queue():
    import time
    from flaxchat.prefetch import BackgroundPrefetcher
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    source = iter([(np.zeros((1,)), np.zeros((1,)))])
    pf = BackgroundPrefetcher(lambda: next(source), mesh, NamedSharding(mesh, PartitionSpec()), 1)
    time.sleep(.2)
    pf.stop()
    assert not pf._thread.is_alive()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0, -1])
def test_launch_rejects_invalid_budgets(value):
    from flaxchat.launch import LaunchSpec
    with pytest.raises(ValueError):
        LaunchSpec('gcp', 'v4-8', 'local', 'a' * 40, ('python',), budget={'max_cost_usd': value})


def test_sharded_initialization_matches_reference():
    from flaxchat.training import initialize_sharded
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices()).reshape(-1, 1), ('data', 'fsdp'))
    def factory():
        model = nnx.Linear(4, 8, rngs=nnx.Rngs(17))
        return model, nnx.Optimizer(model, optax.adamw(.01), wrt=nnx.Param)
    expected = factory()
    actual = initialize_sharded(factory, mesh)
    for a, b in zip(jax.tree.leaves(nnx.state(expected)), jax.tree.leaves(nnx.state(actual)), strict=False):
        # Jitted random-normal initialization may fuse FP32 operations. Allow
        # four FP32 epsilons, while still checking every parameter/optimizer leaf.
        np.testing.assert_allclose(a, b, atol=4 * np.finfo(np.float32).eps, rtol=0)


@pytest.mark.parametrize('standard', [False, True])
def test_cached_generation_matches_full_model_after_nonzero_updates(standard):
    from flaxchat.engine import generate, generate_with_cache
    config = GPTConfig(sequence_len=16, vocab_size=31, n_layer=2, n_head=2, n_kv_head=2,
                       n_embd=32, standard_gpt=standard, tie_embeddings=True)
    model = GPT(config, rngs=nnx.Rngs(7))
    if not standard:
        model.smear_lambda[...] = jnp.full_like(model.smear_lambda[...], .4)
    for block in model.blocks:
        block.attn.c_proj.kernel[...] = jnp.full_like(block.attn.c_proj.kernel[...], .02)
    tokens = [1, 2, 3, 4]
    assert generate_with_cache(model, tokens, max_tokens=3, temperature=0, top_k=None) == generate(model, tokens, max_tokens=3, temperature=0)


def test_host_prefetch_preserves_consumed_cursor():
    from flaxchat.prefetch import BackgroundPrefetcher
    cursor = {'next_batch': 0}
    def load():
        cursor['next_batch'] += 1
        return np.asarray([cursor['next_batch']]), cursor
    pf = BackgroundPrefetcher(load, None, None, prefetch_count=2, place=False)
    try:
        first, state = next(pf)
        second, later = next(pf)
        assert int(first[0]) == state['next_batch'] == 1
        assert int(second[0]) == later['next_batch'] == 2
        assert state['next_batch'] == 1
    finally:
        pf.stop()


def test_global_batch_contract_with_accumulation():
    from flaxchat.training import place_host_batch
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    local_rows = 3 * jax.local_device_count()
    rows = np.arange(2 * local_rows * 4).reshape(2, local_rows, 4)
    placed = place_host_batch(rows, mesh, batch_axis=1)
    assert placed.shape == (2, 3 * jax.device_count(), 4)
    assert placed.size == 2 * 3 * jax.device_count() * 4
    np.testing.assert_array_equal(placed, rows)


def test_budget_guard_covers_setup_and_defaults_to_cleanup(monkeypatch):
    from scripts import train_tpu
    events = []
    class Timer:
        def __init__(self, duration, callback):
            assert duration == 1800
            events.append('guard-created')
        def start(self): events.append('guard-started')
        def cancel(self): events.append('guard-cancelled')
    class VM:
        def __getattr__(self, name):
            def call(*args, **kwargs): events.append(name)
            return call
    monkeypatch.setattr(train_tpu.threading, 'Timer', Timer)
    args = train_tpu.build_parser().parse_args(['--name', 'test', '--max-cost', '5', '--hourly-rate', '10'])
    spec = train_tpu.build_launch_spec(args, revision='a' * 40)
    assert spec.teardown == 'always'
    assert train_tpu.run_adapter(args, spec, VM(), None) == 0
    assert events.index('guard-started') < events.index('up') < events.index('setup')
    assert events[-1] == 'down'
    assert 'set_budget' not in events


def test_delayed_launch_waits_before_allocation(monkeypatch):
    from scripts import train_tpu
    events = []
    class VM:
        def __getattr__(self, name):
            def call(*args, **kwargs): events.append(name)
            return call
    monkeypatch.setattr(train_tpu.time, 'sleep', lambda _: events.append('wait'))
    args = train_tpu.build_parser().parse_args(['--name', 'test', '--start-after', '23:59', '--run-once'])
    spec = train_tpu.build_launch_spec(args, revision='a' * 40)
    train_tpu.run_adapter(args, spec, VM(), None)
    assert events.index('wait') < events.index('up')
    assert events[-1] == 'down'


def test_projection_uses_compute_dtype(tiny_model):
    projection = tiny_model.blocks[0].attn.c_q
    ir = jax.make_jaxpr(projection)(jnp.ones((1, 4, tiny_model.config.n_embd), COMPUTE_DTYPE))
    dots = [eq for eq in ir.jaxpr.eqns if eq.primitive.name == 'dot_general']
    assert dots
    assert all(eq.invars[0].aval.dtype == COMPUTE_DTYPE and eq.invars[1].aval.dtype == COMPUTE_DTYPE for eq in dots)


def test_source_identity_mismatch_rejected_before_live_mutation(tmp_path):
    from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint, CheckpointCompatibilityError
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(.01), wrt=nnx.Param)
    manager = create_checkpoint_manager(str(tmp_path), async_checkpointing=False)
    try:
        save_checkpoint(manager, 1, model, optimizer, {'source_python_sha256': 'old-source'})
        manager.wait_until_finished()
    finally:
        manager.close()
    model.kernel[...] = jnp.zeros_like(model.kernel[...])
    with pytest.raises(CheckpointCompatibilityError, match='identity mismatch'):
        restore_model_from_checkpoint(model, str(tmp_path), optimizer=optimizer,
                                      expected_identity={'source_python_sha256': 'changed-source'})
    np.testing.assert_array_equal(model.kernel[...], 0)
