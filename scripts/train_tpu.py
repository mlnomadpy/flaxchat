"""Provision GCP compute and invoke the shared pretraining service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from flaxchat.launch import LaunchSpec


ROOT = Path(__file__).resolve().parents[1]
REMOTE_MANIFEST = Path("artifacts/gcp-launch.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--accelerator", "-a", default="v4-8")
    parser.add_argument("--gpu", help="GPU shorthand (a100, h100x8, t4)")
    parser.add_argument("--zone", default="us-central2-b")
    parser.add_argument("--zones", help="comma-separated multi-zone failover")
    parser.add_argument("--project", default=os.environ.get("GCLOUD_PROJECT", ""))
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument("--preemptible", action="store_true", dest="preemptible")
    capacity.add_argument("--on-demand", action="store_false", dest="preemptible")
    parser.set_defaults(preemptible=True)
    parser.add_argument("--queued", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--gcs")
    parser.add_argument("--secrets", nargs="*", default=["WANDB_API_KEY", "HF_TOKEN"])
    parser.add_argument("--notify")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--teardown", action="store_true", default=True)
    parser.add_argument("--keep-resource", action="store_false", dest="teardown", help="Explicitly retain the resource after the run")
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--hourly-rate", type=float, help="Verified total slice USD/hour, required with --max-cost")
    parser.add_argument("--start-after")
    parser.add_argument("--collect", nargs="*")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/gcp-launch.json"))
    parser.add_argument("--save-profile")
    parser.add_argument("--repo", help="Git repo to clone")
    parser.add_argument("--guarded-manifest", type=Path, help="Manifest-driven guarded replacement")
    parser.add_argument("--manifest-uri", help="Immutable GCS URI of guarded manifest")
    parser.add_argument("--output", type=Path, help="Local lifecycle receipts directory")
    return parser


def build_launch_spec(
    args: argparse.Namespace, *, revision: str | None = None
) -> LaunchSpec:
    resolved_revision = revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    argv = ["python", "-m", "scripts.pretrain", "--depth", str(args.depth), "--run", args.run_name]
    if args.steps > 0:
        argv.extend(("--num-iterations", str(args.steps)))
    argv.extend(shlex.split(args.extra_args))
    accelerator = args.gpu or args.accelerator
    return LaunchSpec(
        platform="gcp",
        accelerator=accelerator,
        source_repository=args.repo or "local-sync",
        source_revision=resolved_revision,
        argv=tuple(argv),
        resolved_config={
            "depth": args.depth,
            "steps": args.steps,
            "run_name": args.run_name,
        },
        artifacts=tuple(args.collect or ()),
        secret_names=tuple(args.secrets),
        budget={"max_cost_usd": args.max_cost} if args.max_cost is not None else {},
        budget_hours=(args.max_cost / args.hourly_rate if args.max_cost is not None and args.hourly_rate else None),
        recovery=args.recover or bool(args.gcs),
        teardown="always" if args.run_once or args.teardown else "never",
    )


def run_adapter(args, spec: LaunchSpec, vm, gcs) -> int:
    """Run one GCP lifecycle with teardown guaranteed after attempted provision."""
    command = "python -m flaxchat.launch --manifest artifacts/gcp-launch.json"
    if args.dry_run:
        vm.dry_run(command, sync=".", secrets=args.secrets)
        return 0
    raise RuntimeError(
        "Legacy tpuz paid execution is disabled: its local watchdog cannot survive "
        "controller loss. Use --guarded-manifest with --manifest-uri and --output, "
        "or scripts.representation_run launch for a verified cloud lease."
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.guarded_manifest:
        if not args.manifest_uri or not args.output:
            parser.error("--guarded-manifest requires --manifest-uri and --output")
        from scripts.representation_run import main as launch
        return launch(["launch", "--manifest", str(args.guarded_manifest),
                       "--manifest-uri", args.manifest_uri, "--output", str(args.output)])
    if not args.dry_run:
        parser.error("Paid legacy execution disabled; use --guarded-manifest (see docs/INFRASTRUCTURE_RUNS.md)")
    try:
        from tpuz import GCE, GCS, TPU
    except ImportError:
        parser.error("tpuz is required; install flaxchat[tpu]")
    spec = build_launch_spec(args)
    spec.write(args.manifest)
    if args.manifest != REMOTE_MANIFEST:
        spec.write(REMOTE_MANIFEST)
    if args.profile:
        vm = TPU.from_profile(args.profile, args.name)
    elif args.gpu:
        vm = GCE.gpu(args.name, gpu=args.gpu, zone=args.zone, project=args.project)
    elif args.zones:
        vm = TPU.create_multi_zone(args.name, args.accelerator, args.zones.split(","), args.project)
    else:
        vm = TPU(args.name, args.accelerator, args.zone, args.project, args.preemptible)
    gcs = GCS(args.gcs) if args.gcs else None
    return run_adapter(args, spec, vm, gcs)


if __name__ == "__main__":
    raise SystemExit(main())
