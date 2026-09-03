"""Canonical reproducible TinyStories end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flaxchat.pipeline import (
    PipelineConfig,
    TINYSTORIES_DATASET,
    TINYSTORIES_LICENSE,
    TINYSTORIES_REVISION,
    load_fixture_stories,
    load_tinystories,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tinystories"))
    parser.add_argument("--smoke", action="store_true", help="use the committed offline corpus")
    parser.add_argument("--max-train-stories", type=int, default=2_000)
    parser.add_argument("--max-validation-stories", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pretrain-steps", type=int, default=2)
    parser.add_argument("--sft-steps", type=int, default=1)
    parser.add_argument("--rl-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--attention-backend", choices=("auto", "xla", "splash"), default="auto"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        heads=args.heads,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        pretrain_steps=args.pretrain_steps,
        sft_steps=args.sft_steps,
        rl_steps=args.rl_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        attention_backend=args.attention_backend,
    )
    if args.smoke:
        fixture = ROOT / "tests" / "fixtures" / "tiny_corpus.txt"
        train, validation = load_fixture_stories(fixture)
        identity = {
            "dataset": "tests/fixtures/tiny_corpus.txt",
            "revision": "committed-with-source-revision",
            "license": "repository-test-fixture",
        }
    else:
        train, validation = load_tinystories(
            args.max_train_stories, args.max_validation_stories
        )
        identity = {
            "dataset": TINYSTORIES_DATASET,
            "revision": TINYSTORIES_REVISION,
            "license": TINYSTORIES_LICENSE,
            "train_indices": [0, args.max_train_stories],
            "validation_indices": [0, args.max_validation_stories],
        }
    manifest = run_pipeline(train, validation, args.output_dir, config, dataset_identity=identity)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
