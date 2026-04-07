"""
Supervised Fine-Tuning (SFT) on conversation data.

Port of nanochat's chat_sft.py for JAX/TPU.

Usage:
    python -m scripts.sft --base-model=d24 --dataset=smoltalk
"""

import os
import gc
import json
import time
import math
import argparse
from functools import partial
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.common import (
    compute_init, setup_mesh, print0, print_banner,
    get_base_dir, COMPUTE_DTYPE, DummyWandb,
)
from flaxchat.tokenizer import get_tokenizer
from flaxchat.optim import setup_optimizer
from flaxchat.checkpoint import (
    create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint,
)

print_banner()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--base-model", type=str, default="d12", help="base model tag (e.g. d12, d24)")
parser.add_argument("--dataset", type=str, default="smoltalk", help="dataset name or path to JSONL")
parser.add_argument("--num-iterations", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--save-every", type=int, default=-1)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
compute_init()
master_process = jax.process_index() == 0

# wandb
import wandb
wandb_run = DummyWandb() if args.run == "dummy" or not master_process else wandb.init(
    project="flaxchat-sft", name=args.run
)

# Tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# ---------------------------------------------------------------------------
# Load base model
# ---------------------------------------------------------------------------
base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.base_model)
print0(f"Loading base model from {checkpoint_dir}")

ckpt_manager = create_checkpoint_manager(checkpoint_dir, max_to_keep=999)
# Load metadata to get model config
_, metadata = ckpt_manager.restore(ckpt_manager.latest_step(), args=None), None

# For now, reconstruct config from base model tag
depth = int(args.base_model.replace("d", ""))
config = FlaxChatConfig.from_depth(depth=depth, vocab_size=vocab_size)
model = GPT(config.model, rngs=nnx.Rngs(0))
restore_model_from_checkpoint(model, checkpoint_dir)
print0(f"Loaded base model: {model.num_params():,} params")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_conversations(dataset_name):
    """Load conversation dataset. Supports HuggingFace datasets or local JSONL."""
    if os.path.exists(dataset_name):
        # Local JSONL file
        conversations = []
        with open(dataset_name, 'r') as f:
            for line in f:
                conversations.append(json.loads(line))
        return conversations
    else:
        # HuggingFace dataset
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
        conversations = []
        for item in ds:
            if "messages" in item:
                conversations.append(item)
            if len(conversations) >= 100000:
                break
        return conversations


print0(f"Loading dataset: {args.dataset}")
conversations = load_conversations(args.dataset)
print0(f"Loaded {len(conversations)} conversations")


def make_sft_batch(conversations, tokenizer, batch_size, max_seq_len, rng_key):
    """Create a training batch from conversations."""
    key = rng_key
    indices = jax.random.randint(key, (batch_size,), 0, len(conversations))

    all_ids = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    all_targets = np.full((batch_size, max_seq_len), -1, dtype=np.int32)

    for b in range(batch_size):
        conv = conversations[int(indices[b])]
        ids, mask = tokenizer.render_conversation(conv, max_tokens=max_seq_len + 1)

        seq_len = min(len(ids) - 1, max_seq_len)
        all_ids[b, :seq_len] = ids[:seq_len]
        # Targets: only supervise assistant tokens (mask=1)
        for t in range(seq_len):
            if mask[t + 1] == 1:
                all_targets[b, t] = ids[t + 1]

    return jnp.array(all_ids), jnp.array(all_targets)


# ---------------------------------------------------------------------------
# Optimizer (simpler than pretraining — just AdamW)
# ---------------------------------------------------------------------------
import optax

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=args.learning_rate,
    warmup_steps=args.warmup_steps,
    decay_steps=args.num_iterations,
    end_value=args.learning_rate * 0.1,
)
tx = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.01)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------
@partial(nnx.jit, donate_argnames=("optimizer",))
def train_step(model, optimizer, inputs, targets):
    def loss_fn(model):
        return model(inputs, targets)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


# ---------------------------------------------------------------------------
# SFT checkpoint dir
# ---------------------------------------------------------------------------
sft_checkpoint_dir = os.path.join(base_dir, "sft_checkpoints", args.base_model)
sft_ckpt_manager = create_checkpoint_manager(sft_checkpoint_dir, max_to_keep=3)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"\nStarting SFT for {args.num_iterations} steps...")

for step in range(args.num_iterations):
    t0 = time.time()

    key = jax.random.key(step)
    inputs, targets = make_sft_batch(conversations, tokenizer, args.batch_size, args.max_seq_len, key)
    loss = train_step(model, optimizer, inputs, targets)

    loss_val = float(loss)
    dt = time.time() - t0

    if step % 10 == 0:
        print0(f"step {step:04d}/{args.num_iterations} | loss: {loss_val:.4f} | dt: {dt*1000:.0f}ms")
        wandb_run.log({"step": step, "sft/loss": loss_val, "sft/dt": dt})

    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        save_checkpoint(sft_ckpt_manager, step, model, optimizer, {"step": step})

# Final save
save_checkpoint(sft_ckpt_manager, args.num_iterations, model, optimizer, {
    "step": args.num_iterations,
    "base_model": args.base_model,
    "dataset": args.dataset,
})

print0("SFT complete!")
wandb_run.finish()
