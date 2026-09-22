"""Opt-in localhost multiprocess regression (requires loopback socket access)."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('FLAXCHAT_RUN_DISTRIBUTED_CPU') != '1',
                    reason='Opt-in localhost processes: FLAXCHAT_RUN_DISTRIBUTED_CPU=1')
def test_checkpoint_time_policy_uses_coordinator_despite_worker_clock_skew(tmp_path):
    import numpy as np
    from flaxchat.token_pool import write_pool
    manifest = write_pool(tmp_path / 'pool',
                          {'train': np.arange(65) % 32, 'validation': np.arange(17) % 32},
                          {'tokenizer': 'synthetic-test-only'}, 32)
    root = Path(__file__).resolve().parents[1]
    worker = '''
import argparse, os, sys
from scripts import train_gpt2
# These processes share one known local filesystem. Bypass only the production
# GCS placement guard, retaining the actual distributed trainer and checkpoint.
original_error = argparse.ArgumentParser.error
def local_shared_path(self, message):
    if message != 'Multi-host checkpoints require a shared gs:// directory':
        original_error(self, message)
argparse.ArgumentParser.error = local_shared_path
# Simulate a worker whose elapsed clock always says a save is overdue.
# Only the coordinator's decision may initiate a collective checkpoint.
if os.environ['JAX_PROCESS_INDEX'] == '1':
    train_gpt2.checkpoint_due = lambda *args: True
raise SystemExit(train_gpt2.main(sys.argv[1:]))
'''
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    jobs = []
    try:
        for rank in range(2):
            env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_COORDINATOR_ADDRESS=f'127.0.0.1:{port}',
                       JAX_PROCESS_COUNT='2', JAX_PROCESS_INDEX=str(rank),
                       XLA_FLAGS='--xla_force_host_platform_device_count=1')
            log = (tmp_path / f'worker-{rank}.log').open('w')
            command = [sys.executable, '-c', worker, '--token-manifest', str(manifest),
                       '--depth', '1', '--seq-len', '8', '--global-batch-size', '2', '--tokens', '64',
                       '--warmup-steps', '1', '--eval-every', '4', '--fsdp', '2',
                       '--checkpoint-interval-seconds', '100000', '--ckpt-dir', str(tmp_path / 'checkpoints'),
                       '--artifact-dir', str(tmp_path / f'results-{rank}')]
            jobs.append((subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for rank, (process, _log) in enumerate(jobs):
            assert process.wait(timeout=90) == 0, (tmp_path / f'worker-{rank}.log').read_text()
    finally:
        for process, log in jobs:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            log.close()
    for rank in range(2):
        result = json.loads((tmp_path / f'results-{rank}/training_summary-process-{rank}.json').read_text())
        assert result['checkpoint_step'] == result['completed_steps'] == 4
        assert [row['step'] for row in result['metrics'] if 'checkpoint_seconds' in row] == [4]


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('FLAXCHAT_RUN_DISTRIBUTED_CPU') != '1',
                    reason='Opt-in localhost processes: FLAXCHAT_RUN_DISTRIBUTED_CPU=1')
def test_two_process_initialization_collective_and_canonical_batch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    jobs = []
    try:
        for rank in range(2):
            env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_COORDINATOR_ADDRESS=f'127.0.0.1:{port}',
                       JAX_PROCESS_COUNT='2', JAX_PROCESS_INDEX=str(rank),
                       XLA_FLAGS='--xla_force_host_platform_device_count=2')
            log = (tmp_path / f'worker-{rank}.log').open('w')
            interleaved = '''
from flaxchat import common
import jax, numpy as np
from jax.sharding import Mesh
def mesh():
    devices = jax.devices()
    return Mesh(np.array([devices[0], devices[2], devices[1], devices[3]]).reshape(4, 1, 1), ('data', 'fsdp', 'tensor'))
common.setup_mesh = mesh
from scripts.multihost_acceptance import main
raise SystemExit(main())
'''
            command = [sys.executable, '-c', interleaved, 'run', '--no-require-tpu',
                       '--project', 'local', '--zone', 'local', '--slice', 'cpu-processes',
                       '--local-batch', '4', '--output-dir', str(tmp_path)]
            jobs.append((subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for rank, (process, log) in enumerate(jobs):
            code = process.wait(timeout=45)
            log.close()
            assert code == 0, (tmp_path / f'worker-{rank}.log').read_text()
    finally:
        for process, log in jobs:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            log.close()
    records = [json.loads((tmp_path / f'process-{rank}.json').read_text()) for rank in range(2)]
    assert all(record['passed'] and record['topology']['process_count'] == 2 for record in records)
    assert records[0]['data']['local_indices'] == [0, 1, 4, 5]
    assert records[1]['data']['local_indices'] == [2, 3, 6, 7]
    assert records[0]['training']['loss'] == records[1]['training']['loss'] == 39.375
    from scripts.multihost_acceptance import validate_records
    with pytest.raises(ValueError, match='requires TPU'):
        validate_records(records, cost_usd=0.)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('FLAXCHAT_RUN_DISTRIBUTED_CPU') != '1',
                    reason='Opt-in localhost processes: FLAXCHAT_RUN_DISTRIBUTED_CPU=1')
def test_two_process_sharded_checkpoint_integrity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    worker = r'''
import sys
from flaxchat.common import compute_init, setup_mesh, place_array, replicate_on_mesh
compute_init()
import jax
import numpy as np
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint, _state_manifest
mesh = setup_mesh(fsdp=2)
model = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(.001), wrt=nnx.Param)
def place(state):
    return jax.tree.map(lambda x: place_array(x, NamedSharding(mesh, P('fsdp') if np.ndim(x) == 2 else P())), state)
nnx.update(model, place(nnx.state(model)))
nnx.update(optimizer, place(nnx.state(optimizer)))
before = _state_manifest(nnx.to_pure_dict(nnx.state(model)))
manager = create_checkpoint_manager(sys.argv[1], async_checkpointing=False)
try:
    save_checkpoint(manager, 1, model, optimizer, {'model_config': {'test': True}}, training_state=replicate_on_mesh({'next_batch': np.asarray(1, np.int32)}, mesh))
    manager.wait_until_finished()
finally:
    manager.close()
restore_model_from_checkpoint(model, sys.argv[1], optimizer=optimizer, load_training_state=True)
assert _state_manifest(nnx.to_pure_dict(nnx.state(model))) == before
print('DISTRIBUTED_CHECKPOINT_PASSED', flush=True)
'''
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    jobs = []
    try:
        for rank in range(2):
            env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_COORDINATOR_ADDRESS=f'127.0.0.1:{port}',
                       JAX_PROCESS_COUNT='2', JAX_PROCESS_INDEX=str(rank),
                       XLA_FLAGS='--xla_force_host_platform_device_count=1')
            log = (tmp_path / f'worker-{rank}.log').open('w')
            jobs.append((subprocess.Popen([sys.executable, '-c', worker, str(tmp_path / 'shared-checkpoints')],
                         cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for rank, (process, log) in enumerate(jobs):
            code = process.wait(timeout=60)
            log.close()
            assert code == 0, (tmp_path / f'worker-{rank}.log').read_text()
    finally:
        for process, log in jobs:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            log.close()
