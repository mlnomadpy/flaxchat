"""
Train flaxchat on Kaggle GPUs/TPUs using kgz.

Usage:
    python -m scripts.train_kaggle --url=https://kkb-... --depth=8 --steps=5000
    python -m scripts.train_kaggle --url=https://kkb-... --depth=8 --steps=5000 \
        --notify=https://hooks.slack.com/... --budget=8
"""

import os
import argparse

parser = argparse.ArgumentParser(description="Train flaxchat on Kaggle")
parser.add_argument("--url", required=True, help="Kaggle Jupyter proxy URL")
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--dataset", default="tinystories")
parser.add_argument("--name", default="flaxchat-kaggle")
parser.add_argument("--notify", default=None, help="Slack webhook")
parser.add_argument("--budget", type=float, default=None, help="Max hours")
args = parser.parse_args()

try:
    from kgz import Kernel
except ImportError:
    print("Install kgz: pip install kgz (or pip install flaxchat[kaggle])")
    exit(1)

k = Kernel(args.url, name=args.name)
print(f"Connected: {k}")

# Health check
k.health_check()

# Set secrets
hf = os.environ.get("HF_TOKEN", "")
wandb = os.environ.get("WANDB_API_KEY", "")
if hf: k.set_env(HF_TOKEN=hf)
if wandb: k.set_env(WANDB_API_KEY=wandb)

# Pipeline
steps = [
    ("Install deps", """
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "flax", "optax", "orbax-checkpoint", "rustbpe", "tiktoken", "datasets"])
print("Done!")
"""),
    ("Check devices", """
import jax
print(f"JAX {jax.__version__}, Backend: {jax.default_backend()}, Devices: {jax.devices()}")
"""),
]

results = k.pipeline(steps, notify_url=args.notify, use_cache=True)

# Check all passed
if not all(r.success for _, r in results):
    print("Setup failed!")
    exit(1)

# Training
print(f"\nTraining: depth={args.depth}, steps={args.steps}")

# Start quota + budget tracking
k.start_quota_tracking()
if args.budget:
    print(f"Budget: {args.budget}h")

# Execute training (inline code for the TinyStories pipeline)
train_code = f"""
print("Training would use run_tinystories.py inline code")
print("depth={args.depth}, steps={args.steps}, batch={args.batch_size}")
print("For full pipeline, paste the run_tinystories code here")
"""
k.execute_notify(train_code, notify_url=args.notify, label=f"Training d{args.depth}")

k.stop_quota_tracking()
k.quota_summary()

# Save
k.save_session()
k.to_notebook(f"{args.name}.ipynb")
print(f"\nSession: {args.name}, Notebook: {args.name}.ipynb")
k.close()
