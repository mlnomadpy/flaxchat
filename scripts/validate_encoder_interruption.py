"""SIGKILL every encoder worker after a chosen checkpoint commits (default: 3)."""
import os
import signal

from scripts import train_encoder


original_save = train_encoder.save_checkpoint
fault_step = 3


def save_then_kill(manager, step, *args, **kwargs):
    result = original_save(manager, step, *args, **kwargs)
    if step == fault_step:
        manager.wait_until_finished()
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices('encoder-committed-fault')
        print(f'FAULT_INJECTION: encoder SIGKILL after committed step {fault_step}', flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    return result


if __name__ == '__main__':
    train_encoder.save_checkpoint = save_then_kill
    parser = train_encoder.parser()
    parser.add_argument('--fault-step', type=int, default=3)
    args = parser.parse_args()
    fault_step = args.fault_step
    if not 0 < fault_step <= args.steps or fault_step % args.save_every:
        parser.error('Fault step must coincide with a committed checkpoint')
    train_encoder.run(args)
