"""Continue-pretrain a 300–500M mmBERT-shaped encoder with native BF16 YAT.

Uses the standard trainer, checkpoint format and input identity checks. This
entry point fixes the architecture; it does not provision cloud resources.
"""
import json
import math
from pathlib import Path

from flaxchat.encoder import EncoderConfig
from scripts import train_encoder


SETTINGS = dict(ffn_type='yat_glu', attention_score='yat_softmax',
                yat_compute_mode='bf16_adaptive', yat_epsilon=.01,
                dtype='bfloat16', residual_dtype='bfloat16', attention_backend='xla',
                mlm_projection='masked')


def parser():
    p = train_encoder.parser()
    p.description = __doc__
    p.set_defaults(**SETTINGS)
    return p


def validate_recipe(args):
    for name, expected in SETTINGS.items():
        if getattr(args, name) != expected:
            raise ValueError(f'YAT mmBERT requires {name}={expected}')
    if not args.pretrained:
        raise ValueError('YAT mmBERT adaptation requires a pretrained snapshot')
    raw = json.loads(Path(args.config).read_text())
    if raw.get('model_type') != 'modernbert':
        raise ValueError('Use the pretrained HF ModernBERT config for conversion')
    config = EncoderConfig.from_hf(raw, compute_dtype=args.dtype,
        residual_dtype=args.residual_dtype, attention_backend=args.attention_backend,
        yat_local_shards=args.yat_local_shards, yat_attention_block_size=args.yat_attention_block_size, yat_global_attention_block_size=args.yat_global_attention_block_size,
        ffn_type=args.ffn_type, attention_score=args.attention_score,
        yat_compute_mode=args.yat_compute_mode, yat_epsilon=args.yat_epsilon,
        yat_softmax_backward=args.yat_softmax_backward or 'factored',
        yat_alpha=1. if args.yat_alpha is None else args.yat_alpha,
        yat_attention_alpha=args.yat_attention_alpha if args.yat_attention_alpha is not None
                            else raw.get('yat_attention_alpha'))
    # Tied decoder is counted once. Two new trainable scalars per block.
    count = sum(math.prod(shape) for shape in train_encoder.pretrained_shapes(config).values())
    count += 2 * config.num_hidden_layers
    if not 300_000_000 <= count <= 500_000_000:
        raise ValueError(f'YAT mmBERT requires 300–500M parameters; got {count}')
    return config, count


def main(argv=None):
    args = parser().parse_args(argv)
    validate_recipe(args)
    train_encoder.run(args)


if __name__ == '__main__':
    main()
