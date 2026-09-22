"""Restore a distributed GPT checkpoint on one host and write a portable copy.

No optimizer update occurs during transfer. All state digests must remain exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--step', type=int)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from flaxchat.common import compute_init, replicate_on_mesh
    mesh = compute_init()
    import jax
    from flax import nnx
    from etils import epath
    from flaxchat.gpt import GPT, GPTConfig
    from flaxchat.checkpoint import (load_checkpoint_metadata, restore_model_from_checkpoint,
                                    create_checkpoint_manager, save_checkpoint)
    if jax.process_count() != 1:
        raise ValueError('Bridge must execute on exactly one host')
    if epath.Path(args.destination).exists():
        raise ValueError('Bridge destination must be new')
    metadata = load_checkpoint_metadata(args.source, args.step)
    config = metadata['resolved_config']
    model = GPT(GPTConfig(**metadata['model_config']), rngs=nnx.Rngs(42))
    from flaxchat.training import pretraining_optimizer
    recipes = {'adamw-clip1-wd0.01': 'adamw',
               'normuon-adamw-default-groups-matrix-wd0.01': 'muon'}
    if config['optimizer'] not in recipes:
        raise ValueError('Unsupported checkpoint optimizer recipe')
    optimizer, _ = pretraining_optimizer(model, kind=recipes[config['optimizer']],
                learning_rate=config['lr'], warmup_steps=config['warmup_steps'], steps=config['steps'])
    _, state = restore_model_from_checkpoint(model, args.source, step=args.step, optimizer=optimizer, load_training_state=True)
    if state is None:
        raise ValueError('Checkpoint has no resumable training state')
    step = int(state['update_step'])
    if step != int(state['next_batch']):
        raise ValueError('Invalid data cursor')
    optimizer.step[...] = step
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    nnx.update(optimizer, replicate_on_mesh(nnx.state(optimizer), mesh))
    manager = create_checkpoint_manager(args.destination, async_checkpointing=False)
    try:
        save_checkpoint(manager, step, model, optimizer, metadata, training_state=state)
        manager.wait_until_finished()
    finally:
        manager.close()
    original = json.loads((epath.Path(args.source) / str(step) / 'manifest/metadata').read_text())
    copied = json.loads((epath.Path(args.destination) / str(step) / 'manifest/metadata').read_text())
    equal = {key: original[key] == copied[key] for key in ('model_state', 'optimizer_state', 'training_state')}
    result = {'passed': all(equal.values()), 'equal': equal, 'backend': jax.default_backend(),
              'process_count': jax.process_count(), 'devices': jax.device_count(), 'step': step,
              'source': args.source, 'destination': args.destination}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
