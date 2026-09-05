"""Provision GCP compute and invoke the shared pretraining service."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import shlex
import subprocess
import time
from contextlib import suppress

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
    parser.add_argument("--teardown", action="store_true")
    parser.add_argument("--max-cost", type=float)
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
        budget={"max_cost_usd": args.max_cost} if args.max_cost else {},
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
    if args.run_once:
        vm.run_once(
            command,
            sync=".",
            collect_files=args.collect,
            gcs=gcs,
            notify_url=args.notify,
        )
        return 0
    provision_attempted = False
    try:
        provision_attempted = True
        vm.up_queued() if args.queued else vm.up()
        vm.setup(extra_pip="flaxchat")
        vm.verify()
        if args.start_after:
            now = datetime.datetime.now()
            hour, minute = map(int, args.start_after.split(":"))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += datetime.timedelta(days=1)
            time.sleep((target - now).total_seconds())
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
        if args.max_cost:
            vm.set_budget(args.max_cost, notify_url=args.notify)
        elif args.recover:
            vm.watch_notify(command, notify_url=args.notify) if args.notify else vm.watch(command)
        else:
            vm.logs(follow=True)
        vm.cost_summary()
        if args.collect:
            vm.collect(args.collect)
        return 0
    finally:
        if spec.teardown == "always" and provision_attempted:
            with suppress(Exception):
                vm.down()


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
