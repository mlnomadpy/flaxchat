"""
Train flaxchat on GCP TPU/GPU using tpuz.

Full lifecycle: provision -> setup -> sync -> train -> monitor -> collect -> teardown.
Handles preemption, GCS checkpoints, secrets, budget, notifications.

Usage:
    # Simple
    python -m scripts.train_tpu --name=d24 -a v4-8 --depth=24

    # GPU instead of TPU
    python -m scripts.train_tpu --name=d24 --gpu=a100 --depth=24

    # Full production run
    python -m scripts.train_tpu --name=d24 -a v4-8 --depth=24 \
        --gcs=gs://bucket --secrets WANDB_API_KEY HF_TOKEN \
        --notify=https://hooks.slack.com/... --recover --collect model.pkl

    # Scheduled overnight
    python -m scripts.train_tpu --name=d24 -a v4-8 --depth=24 \
        --start-after=22:00 --max-cost=50

    # Run-once (auto cleanup)
    python -m scripts.train_tpu --name=d24 -a v4-8 --depth=24 --run-once

    # Dry run
    python -m scripts.train_tpu --name=d24 -a v4-8 --depth=24 --dry-run

    # From profile
    python -m scripts.train_tpu --name=d24 --profile=big-run --depth=24
"""

import os
import argparse

parser = argparse.ArgumentParser(description="Train flaxchat on GCP TPU/GPU")

# VM
parser.add_argument("--name", required=True)
parser.add_argument("--accelerator", "-a", default="v4-8")
parser.add_argument("--gpu", default=None, help="GPU shorthand (a100, h100x8, t4)")
parser.add_argument("--zone", default="us-central2-b")
parser.add_argument("--zones", default=None, help="Multi-zone failover")
parser.add_argument("--project", default=os.environ.get("GCLOUD_PROJECT", ""))
parser.add_argument("--preemptible", action="store_true", default=True)
parser.add_argument("--queued", action="store_true")
parser.add_argument("--profile", default=None)

# Training
parser.add_argument("--depth", type=int, default=24)
parser.add_argument("--steps", type=int, default=-1)
parser.add_argument("--run-name", default="default")
parser.add_argument("--extra-args", default="")

# Infrastructure
parser.add_argument("--gcs", default=None)
parser.add_argument("--secrets", nargs="*", default=["WANDB_API_KEY", "HF_TOKEN"])
parser.add_argument("--notify", default=None)
parser.add_argument("--recover", action="store_true")
parser.add_argument("--teardown", action="store_true")
parser.add_argument("--max-cost", type=float, default=None)
parser.add_argument("--start-after", default=None)
parser.add_argument("--collect", nargs="*", default=None)
parser.add_argument("--run-once", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--save-profile", default=None)
parser.add_argument("--repo", default=None, help="Git repo to clone")

args = parser.parse_args()

try:
    from tpuz import TPU, GCE, GCS
except ImportError:
    print("Install tpuz: pip install tpuz (or pip install flaxchat[tpu])")
    exit(1)

# Build command
cmd = f"python -m scripts.pretrain --depth={args.depth} --run={args.run_name}"
if args.steps > 0: cmd += f" --num-iterations={args.steps}"
if args.extra_args: cmd += f" {args.extra_args}"

# Create VM
if args.profile:
    vm = TPU.from_profile(args.profile, args.name)
elif args.gpu:
    vm = GCE.gpu(args.name, gpu=args.gpu, zone=args.zone, project=args.project)
elif args.zones:
    vm = TPU.create_multi_zone(args.name, args.accelerator, args.zones.split(","), args.project)
else:
    vm = TPU(args.name, args.accelerator, args.zone, args.project, args.preemptible)

gcs = GCS(args.gcs) if args.gcs else None

# Dry run
if args.dry_run:
    vm.dry_run(cmd, sync=".", secrets=args.secrets)
    exit(0)

# Save profile
if args.save_profile:
    vm.save_profile(args.save_profile)

# Preflight
vm.preflight()

# Execute
if args.run_once:
    vm.run_once(cmd, sync=".", collect_files=args.collect, gcs=gcs, notify_url=args.notify)
elif args.start_after or args.max_cost:
    vm.up_queued() if args.queued else vm.up()
    vm.setup(extra_pip="flaxchat")
    vm.verify()
    if args.start_after:
        import datetime, time
        now = datetime.datetime.now()
        h, m = map(int, args.start_after.split(":"))
        target = now.replace(hour=h, minute=m, second=0)
        if target <= now: target += datetime.timedelta(days=1)
        print(f"Waiting until {args.start_after}...")
        time.sleep((target - now).total_seconds())
    if args.repo: vm.clone_repo(args.repo, install=True)
    if gcs: vm.run_with_resume(cmd, gcs=gcs, run_name=args.run_name, sync=".", secrets=args.secrets)
    else: vm.run(cmd, sync=".", secrets=args.secrets)
    if args.max_cost: vm.set_budget(args.max_cost, notify_url=args.notify)
    else: vm.wait()
    vm.cost_summary()
    if args.collect: vm.collect(args.collect)
    if args.teardown: vm.down()
elif args.recover:
    vm.up_queued() if args.queued else vm.up()
    vm.setup(extra_pip="flaxchat")
    vm.verify()
    if args.repo: vm.clone_repo(args.repo, install=True)
    if gcs: vm.run_with_resume(cmd, gcs=gcs, run_name=args.run_name, sync=".", secrets=args.secrets)
    else: vm.run(cmd, sync=".", secrets=args.secrets)
    if args.notify: vm.watch_notify(cmd, notify_url=args.notify)
    else: vm.watch(cmd)
    vm.cost_summary()
    if args.collect: vm.collect(args.collect)
    if args.teardown: vm.down()
else:
    vm.up_queued() if args.queued else vm.up()
    vm.setup(extra_pip="flaxchat")
    vm.verify()
    if args.repo: vm.clone_repo(args.repo, install=True)
    if gcs: vm.run_with_resume(cmd, gcs=gcs, run_name=args.run_name, sync=".", secrets=args.secrets)
    else: vm.run(cmd, sync=".", secrets=args.secrets)
    vm.logs(follow=True)
    vm.cost_summary()
    if args.collect: vm.collect(args.collect)
    if args.teardown: vm.down()
