"""Inject a rank-zero SIGKILL after a fully committed distributed checkpoint."""
from __future__ import annotations

import os
import signal
import sys


def coordinator_process(environment, runtime_rank):
    """Use the launcher's coordination-service rank, not TPU device ordering."""
    return int(environment.get('JAX_PROCESS_INDEX', runtime_rank)) == 0


def main():
    from flaxchat import checkpoint
    from scripts.train_gpt2 import main as train
    import jax
    from jax.experimental import multihost_utils
    original = checkpoint.save_checkpoint
    def interrupt(manager, step, *args, **kwargs):
        result = original(manager, step, *args, **kwargs)
        manager.wait_until_finished()
        multihost_utils.sync_global_devices(f'kill-after-committed-{step}')
        if coordinator_process(os.environ, jax.process_index()):
            print(f'FAULT_INJECTION: SIGKILL coordinator_rank=0 committed_step={step} runtime_rank={jax.process_index()}', flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        # Peers must observe loss of the coordinator, not continue training.
        multihost_utils.sync_global_devices('coordinator-killed')
        return result
    checkpoint.save_checkpoint = interrupt
    return train(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
