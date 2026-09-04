"""Local adapter for the canonical TinyStories training service."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from flaxchat.launch import LaunchSpec
from scripts.run_tinystories import main as run_tinystories


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=2, dest="layers")
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--export-dir", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/local-launch.json"))
    parser.add_argument(
        "--export-tflite",
        action="store_true",
        help="deprecated; run scripts.convert_to_tflite on the resulting checkpoint",
    )
    return parser


def build_launch_spec(
    args: argparse.Namespace, *, revision: str | None = None
) -> LaunchSpec:
    heads = max(1, args.n_embd // 32)
    command = [
        "--output-dir", str(args.export_dir),
        "--layers", str(args.layers),
        "--embedding-dim", str(args.n_embd),
        "--heads", str(heads),
        "--sequence-length", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--pretrain-steps", str(args.steps),
        "--sft-steps", "1",
        "--rl-steps", "1",
        "--learning-rate", str(args.lr),
    ]
    if args.smoke:
        command.append("--smoke")
    resolved_revision = revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return LaunchSpec(
        platform="local",
        accelerator="auto",
        source_repository="local-worktree",
        source_revision=resolved_revision,
        argv=("python", "-m", "scripts.run_tinystories", *command),
        resolved_config={
            "layers": args.layers,
            "embedding_dim": args.n_embd,
            "heads": heads,
            "sequence_length": args.seq_len,
            "batch_size": args.batch_size,
            "pretrain_steps": args.steps,
            "learning_rate": args.lr,
        },
        artifacts=(str(args.export_dir),),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.export_tflite:
        raise SystemExit(
            "--export-tflite is no longer coupled to training; run "
            "`python -m scripts.convert_to_tflite` on the checkpoint"
        )
    spec = build_launch_spec(args)
    spec.write(args.manifest)
    if args.dry_run:
        print(spec.to_json())
        return 0
    return run_tinystories(list(spec.argv[3:]))


if __name__ == "__main__":
    raise SystemExit(main())
