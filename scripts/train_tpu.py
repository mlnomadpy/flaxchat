"""Provision GCP compute and invoke the shared pretraining service."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import shlex
import subprocess
import time
import math
import threading

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
    if args.save_profile:
        vm.save_profile(args.save_profile)
    if args.max_cost is not None:
        if not math.isfinite(args.max_cost) or args.max_cost <= 0:
            raise ValueError("--max-cost must be finite and positive")
        if args.hourly_rate is None or not math.isfinite(args.hourly_rate) or args.hourly_rate <= 0:
            raise ValueError("--max-cost requires a verified finite positive --hourly-rate for the entire slice")
        if not args.teardown:
            raise ValueError("A budgeted run cannot retain an allocated resource")
    # Waiting must happen before allocation, including the run-once path.
    if args.start_after:
        now = datetime.datetime.now()
        hour, minute = map(int, args.start_after.split(":"))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        time.sleep((target - now).total_seconds())
    expired = threading.Event()
    cleanup_errors = []
    def expire():
        expired.set()
        try:
            vm.down()
        except Exception as error:
            cleanup_errors.append(error)
    def check_deadline():
        if expired.is_set():
            raise TimeoutError("Allocation budget expired; cleanup was requested")
    guard = None
    provision_attempted = False
    try:
        provision_attempted = True
        if args.max_cost is not None:
            guard = threading.Timer(args.max_cost / args.hourly_rate * 3600, expire)
            guard.daemon = True
            guard.start()
        vm.up_queued() if args.queued else vm.up()
        check_deadline()
        vm.setup(extra_pip="flaxchat")
        vm.verify()
        check_deadline()
        if args.repo:
            vm.clone_repo(args.repo, install=True)
        if gcs:
            vm.run_with_resume(
                command,
                gcs=gcs,
                run_name=args.run_name,
                sync=".",
                secrets=args.secrets,
            )
        else:
            vm.run(command, sync=".", secrets=args.secrets)
        check_deadline()
        if args.recover:
            vm.watch_notify(command, notify_url=args.notify) if args.notify else vm.watch(command)
        else:
            vm.logs(follow=True)
        check_deadline()
        vm.cost_summary()
        if args.collect:
            vm.collect(args.collect)
        return 0
    finally:
        if guard is not None:
            guard.cancel()
        if spec.teardown == "always" and provision_attempted:
            vm.down()
        if cleanup_errors:
            raise RuntimeError("Budget watchdog resource deletion failed") from cleanup_errors[0]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
