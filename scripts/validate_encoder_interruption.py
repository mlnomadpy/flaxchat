"""Deliberately SIGKILL every encoder worker after the step-3 checkpoint commits."""
import os
import signal

from scripts import train_encoder


original_save = train_encoder.save_checkpoint


def save_then_kill(manager, step, *args, **kwargs):
    result = original_save(manager, step, *args, **kwargs)
    if step == 3:
        manager.wait_until_finished()
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices('encoder-committed-fault')
        print('FAULT_INJECTION: encoder SIGKILL after committed step 3', flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    return result


if __name__ == '__main__':
    train_encoder.save_checkpoint = save_then_kill
    train_encoder.run(train_encoder.parser().parse_args())
