"""Launch the canonical TinyStories training pipeline through the Kaggle CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from flaxchat.launch import LaunchSpec
from scripts.kaggle_tpu_tests import command, kaggle_cli, monitor_kernel


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "accelerators" / "kaggle" / "train.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True, help="owner/kernel-slug")
    parser.add_argument("--accelerator", choices=("tpu", "gpu"), default="tpu")
    parser.add_argument("--revision", help="full source SHA; defaults to local HEAD")
    parser.add_argument("--repository", default="https://github.com/mlnomadpy/flaxchat.git")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "kaggle-training")
    parser.add_argument("--artifact-dir", default="artifacts/tinystories")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--budget-hours", type=float)
    parser.add_argument("--secret", action="append", default=[], dest="secrets")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_launch_spec(args: argparse.Namespace) -> LaunchSpec:
    revision = args.revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    artifact_dir = args.artifact_dir
    argv = (
        "python", "-m", "scripts.run_tinystories",
        "--output-dir", artifact_dir,
        "--layers", str(args.layers),
        "--pretrain-steps", str(args.steps),
        "--sft-steps", "1", "--rl-steps", "1",
        "--batch-size", str(args.batch_size),
        "--sequence-length", str(args.sequence_length),
    )
    return LaunchSpec(
        platform="kaggle",
        accelerator=args.accelerator,
        source_repository=args.repository,
        source_revision=revision,
        argv=argv,
        resolved_config={
            "layers": args.layers,
            "pretrain_steps": args.steps,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
        },
        artifacts=(artifact_dir,),
        secret_names=tuple(args.secrets),
        budget_hours=args.budget_hours,
        budget=({"max_hours": args.budget_hours} if args.budget_hours else {}),
        recovery=True,
        teardown="always",
    )


def render_bundle(spec: LaunchSpec, kernel_id: str, destination: Path) -> None:
    owner, separator, slug = kernel_id.partition("/")
    if not separator or not owner or not slug:
        raise ValueError("--kernel-id must use owner/kernel-slug")
    encoded = json.dumps(spec.to_json())[1:-1]
    source = TEMPLATE.read_text(encoding="utf-8").replace("__LAUNCH_SPEC_JSON__", encoded)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "launch.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": kernel_id,
        "title": slug.replace("-", " ").title(),
        "code_file": "launch.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": str(spec.accelerator == "gpu").lower(),
        "enable_tpu": str(spec.accelerator == "tpu").lower(),
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if kaggle_cli() is None and not args.dry_run:
        parser.error("Kaggle CLI not found; install with `python -m pip install kaggle`)")
    try:
        spec = build_launch_spec(args)
        if args.dry_run:
            render_bundle(spec, args.kernel_id, args.output_dir / "bundle")
            spec.write(args.output_dir / "launch_spec.json")
            print(spec.to_json())
            return 0
        with tempfile.TemporaryDirectory(prefix="flaxchat-kaggle-training-") as directory:
            render_bundle(spec, args.kernel_id, Path(directory))
            command("kernels", "push", "-p", directory)
        spec.write(args.output_dir / "launch_spec.json")
        print(f"Submitted {args.kernel_id} at exact revision {spec.source_revision}")
        if args.no_wait:
            return 0
        return monitor_kernel(args.kernel_id, args.output_dir)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
