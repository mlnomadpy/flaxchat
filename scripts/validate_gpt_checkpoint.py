"""Restore the prepared-token GPT checkpoint and generate deterministic tokens."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--token-manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from flax import nnx
    from flaxchat.checkpoint import load_checkpoint_metadata, restore_model_from_checkpoint
    from flaxchat.common import compute_init
    from flaxchat.gpt import GPT, GPTConfig
    from flaxchat.engine import generate
    from flaxchat.token_pool import TokenPool
    compute_init()
    pool = TokenPool(args.token_manifest)
    metadata = load_checkpoint_metadata(args.checkpoint_dir)
    model = GPT(GPTConfig(**metadata['model_config']), rngs=nnx.Rngs(0))
    restore_model_from_checkpoint(model, args.checkpoint_dir, expected_identity={'data_manifest': pool.identity})
    prompt = [int(x) for x in pool.arrays['validation'][:4]]
    generated = generate(model, prompt, max_tokens=4, temperature=0)
    assert len(generated) == 8 and generated[:4] == prompt
    assert all(0 <= token < pool.vocab_size for token in generated)
    args.output.write_text(json.dumps({'checkpoint_step': metadata['step'], 'prompt_ids': prompt,
                                      'generated_ids': generated, 'temperature': 0}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
