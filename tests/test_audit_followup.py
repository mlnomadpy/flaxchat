"""Adversarial regressions found after the first physical acceptance campaign."""
import json
from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from flaxchat.training import apply_gradients_if_finite
from flaxchat.eval import evaluate_example_mc
from flaxchat.config import GPTConfig
from flaxchat.gpt import GPT
from flaxchat.token_pool import TokenPool, write_streaming_pool
from scripts.ci_scope import select_scope


@pytest.mark.parametrize('kind', ['adam', 'sgd'])
def test_finite_gradients_overflow_rolls_back_every_state(kind):
    class Model(nnx.Module):
        def __init__(self): self.w = nnx.Param(jnp.ones((1,), jnp.float32))
    model = Model()
    tx = optax.adamw(1e-3) if kind == 'adam' else optax.sgd(1e20)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    before = [np.array(x) for x in jax.tree.leaves(nnx.state((model, optimizer)))]
    grads = jax.tree.map(lambda p: jnp.full_like(p, 1e20), nnx.state(model, nnx.Param))
    accepted = nnx.jit(apply_gradients_if_finite)(model, optimizer, grads, jnp.array(1.))
    assert not bool(accepted)
    for expected, actual in zip(before, jax.tree.leaves(nnx.state((model, optimizer))), strict=True):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_invalid_likelihood_never_scores_correct(invalid):
    class Tokenizer:
        def get_bos_token_id(self): return 0
        def __call__(self, prompts, prepend=None): return [[0, 1, 2], [0, 1, 3]]
    class Model:
        def __call__(self, x): return jnp.full((*x.shape, 4), invalid)
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate_example_mc(Model(), Tokenizer(), {'query': 'q', 'choices': ['a', 'b'], 'gold': 0}, [], ' ')


def test_unlimited_and_excessive_task_limits():
    from tasks.common import Task
    class Dataset(Task):
        def num_examples(self): return 3
    assert len(Dataset(stop=None)) == 3
    assert len(Dataset(stop=100)) == 3
    assert len(Dataset(start=5)) == 0


def test_forced_ci_really_runs_all_gates():
    scope = select_scope([], force_full=True)
    assert all(scope[name] for name in ('run_multidevice', 'run_e2e', 'run_audit', 'run_build'))
    for module in ('training', 'common', 'gpt'):
        assert select_scope([f'flaxchat/{module}.py'])['run_multidevice']
    assert 'tests/test_gcp_cleanup_guard.py' in select_scope(['scripts/gcp_cleanup_guard.py'])['tests']
    assert 'tests/test_spot_watchdog.py' in select_scope(['infra/tpu/spot_watchdog.py'])['tests']


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_explicit_model_precision_controls_projection_and_cache(dtype):
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_kv_head=2,
                       n_embd=32, standard_gpt=True, compute_dtype=dtype)
    model = GPT(config, rngs=nnx.Rngs(0))
    assert jnp.dtype(model.wte.dtype).name == dtype
    assert jnp.dtype(model.blocks[0].attn.c_q.dtype).name == dtype
    assert bool(jnp.all(jnp.isfinite(model(jnp.ones((1, 8), jnp.int32)))))
    from flaxchat.engine import generate_with_cache
    assert len(generate_with_cache(model, [1, 2], max_tokens=2, temperature=0)) == 4


def test_same_shape_wrong_architecture_rejected(tmp_path):
    from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_kv_head=2,
                       n_embd=32, standard_gpt=True, window_pattern='L')
    model = GPT(config, rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(model, optax.sgd(.01), wrt=nnx.Param)
    with create_checkpoint_manager(str(tmp_path), async_checkpointing=False) as manager:
        save_checkpoint(manager, 1, model, opt, {'model_config': asdict(config)})
    other = GPT(replace(config, window_pattern='S'), rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match='configuration'):
        restore_model_from_checkpoint(other, str(tmp_path))


def test_same_vocabulary_size_different_tokenizer_rejected():
    from flaxchat.dataloader import _tokenizer_identity
    from flaxchat.checkpoint import validate_checkpoint_tokenizer
    class Tokenizer:
        def __init__(self, reverse): self.reverse = reverse
        def get_vocab_size(self): return 2
        def decode(self, ids): return str(1 - ids[0] if self.reverse else ids[0])
    metadata = {'tokenizer_identity': _tokenizer_identity(Tokenizer(False))}
    validate_checkpoint_tokenizer(metadata, Tokenizer(False))
    with pytest.raises(ValueError, match='tokenizer identity'):
        validate_checkpoint_tokenizer(metadata, Tokenizer(True))


def test_metadata_resolution_binds_and_validates_selected_step(tmp_path):
    from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint, load_checkpoint_metadata
    class Model(nnx.Module):
        def __init__(self): self.w = nnx.Param(jnp.ones(1))
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.01), wrt=nnx.Param)
    with create_checkpoint_manager(str(tmp_path), async_checkpointing=False) as manager:
        save_checkpoint(manager, 1, model, optimizer, {'legacy': True})
        save_checkpoint(manager, 2, model, optimizer, {'step': 999})
    assert load_checkpoint_metadata(str(tmp_path), step=1)['step'] == 1
    with pytest.raises(ValueError, match='step differs'):
        load_checkpoint_metadata(str(tmp_path), step=2)


def test_shard_boundaries_and_lazy_integrity(tmp_path):
    values = np.arange(129) % 32
    path = write_streaming_pool(tmp_path / 'pool',
                               {s: (values[i:i+3] for i in range(0, len(values), 3)) for s in ('train', 'validation')}, {}, 32, shard_tokens=11)
    pool = TokenPool(path)
    assert len(pool.arrays['train'].verified) == 0
    x, y = pool.batch('train', 2, 4, 8)
    np.testing.assert_array_equal(x.ravel(), values[64:96])
    np.testing.assert_array_equal(y.ravel(), values[65:97])
    metadata = json.loads(path.read_text())
    last = path.parent / metadata['splits']['validation']['shards'][-1]['file']
    last.write_bytes(bytes(last.stat().st_size))
    pool = TokenPool(path)
    with pytest.raises(ValueError, match='Corrupt'):
        pool.batch('validation', 3, 4, 8)


def test_failed_stream_never_publishes_pool(tmp_path):
    def broken():
        yield [1, 2, 3]
        raise RuntimeError('source interrupted')
    with pytest.raises(RuntimeError):
        write_streaming_pool(tmp_path / 'pool', {'train': broken(), 'validation': [[1, 2]]}, {}, 32, shard_tokens=2)
    assert not (tmp_path / 'pool').exists()
    assert not list(tmp_path.glob('*.partial-*'))


@pytest.mark.parametrize('task_name', ['mmlu', 'gsm8k', 'arc'])
def test_eval_default_runs_full_task_and_clamps_limit(tmp_path, monkeypatch, task_name):
    from flaxchat.stages import eval as stage
    from tasks.common import Task
    from flaxchat.dataloader import _tokenizer_identity
    from tests.test_finetuning_resume import Tokenizer
    # Keep this a stage contract test: model execution is covered separately.
    class EvalTokenizer(Tokenizer):
        def encode(self, *args, **kwargs):
            assert kwargs['prepend'] == self.get_bos_token_id()
            return [1, 2]
        def decode(self, ids): return 'A'
    tokenizer = EvalTokenizer()
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2,
                       n_kv_head=2, n_embd=32, standard_gpt=True)
    monkeypatch.setattr(stage, 'get_base_dir', lambda: str(tmp_path))
    monkeypatch.setattr(stage, 'get_tokenizer', lambda: tokenizer)
    monkeypatch.setattr(stage, 'load_checkpoint_metadata', lambda _: {
        'model_config': asdict(config), 'step': 4, 'tokenizer_identity': _tokenizer_identity(tokenizer)})
    monkeypatch.setattr(stage, 'restore_model_from_checkpoint', lambda *a, **k: None)
    monkeypatch.setattr(stage, 'generate_with_cache', lambda m, tokens, **k: tokens + [3])
    class Examples(Task):
        def __init__(self, subset, split, **kwargs): super().__init__(**kwargs)
        def num_examples(self): return 3
        def get_example(self, index):
            assert 0 <= index < 3
            return {'messages': [{'content': 'question'}], 'letters': ['A']}
        def evaluate(self, *args): return True
    import importlib
    module = importlib.import_module('tasks.' + task_name)
    monkeypatch.setattr(module, {'mmlu': 'MMLU', 'gsm8k': 'GSM8K', 'arc': 'ARC'}[task_name], Examples)
    for limit, expected in [(0, 3), (1, 1), (99, 3)]:
        result = stage.run(stage.EvalRequest(tasks=task_name, max_per_task=limit))
        assert result.exit_code == 0
        assert result.metrics[task_name]['total'] == expected


def test_incomplete_core_exits_unsuccessfully(tmp_path, monkeypatch):
    from flaxchat.stages import eval as stage
    from flaxchat.dataloader import _tokenizer_identity
    from tests.test_finetuning_resume import Tokenizer
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2,
                       n_kv_head=2, n_embd=32, standard_gpt=True)
    monkeypatch.setattr(stage, 'get_tokenizer', Tokenizer)
    monkeypatch.setattr(stage, 'get_base_dir', lambda: str(tmp_path))
    monkeypatch.setattr(stage, 'load_checkpoint_metadata', lambda _: {
        'model_config': asdict(config), 'step': 4, 'tokenizer_identity': _tokenizer_identity(Tokenizer())})
    monkeypatch.setattr(stage, 'restore_model_from_checkpoint', lambda *a, **k: None)
    monkeypatch.setattr(stage, 'evaluate_core', lambda *a, **k: {'core_metric': None, 'status': 'incomplete'})
    assert stage.run(stage.EvalRequest()).exit_code == 1
    with pytest.raises(ValueError, match='Unknown'):
        stage.run(stage.EvalRequest(tasks='misspelled'))


@pytest.mark.integration
def test_process_loss_during_save_preserves_last_committed_checkpoint(tmp_path):
    """Kill inside Orbax's async handler, before directory finalization."""
    import os
    import subprocess
    import sys
    import time
    marker = tmp_path / 'write-started'
    code = r'''
import asyncio, os, sys
from pathlib import Path
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
import optax
from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
class Model(nnx.Module):
    def __init__(self): self.w = nnx.Param(jnp.ones((8,), jnp.float32))
m = Model(); opt = nnx.Optimizer(m, optax.adamw(.01), wrt=nnx.Param)
root = Path(sys.argv[1])
manager = create_checkpoint_manager(str(root / 'checkpoints'), async_checkpointing=True)
save_checkpoint(manager, 1, m, opt, {'step': 1})
manager.wait_until_finished()
original = ocp.JsonCheckpointHandler.async_save
async def blocked(self, directory, *args, **kwargs):
    futures = await original(self, directory, *args, **kwargs)
    (root / 'write-started').write_text(str(directory))
    await asyncio.sleep(60)
    return futures
ocp.JsonCheckpointHandler.async_save = blocked
m.w[...] = m.w[...] + 1
save_checkpoint(manager, 2, m, opt, {'step': 2})
manager.wait_until_finished()
'''
    with (tmp_path / 'child.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)], stdout=log,
                                 stderr=subprocess.STDOUT, env={**os.environ, 'JAX_PLATFORMS': 'cpu'})
        try:
            deadline = time.monotonic() + 30
            while not marker.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    pytest.fail((tmp_path / 'child.log').read_text())
                time.sleep(.05)
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    from flaxchat.checkpoint import create_checkpoint_manager, load_checkpoint
    with create_checkpoint_manager(str(tmp_path / 'checkpoints'), async_checkpointing=False) as manager:
        assert manager.latest_step() == 1
        state, metadata = load_checkpoint(manager)
        assert metadata['step'] == 1
        np.testing.assert_array_equal(state['w'], np.ones(8))
