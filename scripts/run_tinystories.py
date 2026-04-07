"""
Full pipeline on TinyStories dataset.

Automatically uses ALL available devices (CPU, GPU, or TPU) via JAX SPMD.
On multi-GPU/TPU: data-parallel training across all devices.

Usage:
    python -m scripts.run_tinystories
    python -m scripts.run_tinystories --depth=4 --n-embd=128 --steps=1000
    python -m scripts.run_tinystories --depth=8 --n-embd=256 --steps=5000  # ~19M params
"""

import os
import sys
import time
import argparse
import pickle
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import optax

from flaxchat.gpt import GPT
from flaxchat.config import GPTConfig
from flaxchat.common import (
    compute_init, get_mesh, replicate_on_mesh, print0, get_base_dir,
)
from flaxchat.engine import generate, generate_with_cache

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Full pipeline on TinyStories")
# Model
parser.add_argument("--depth", type=int, default=4, help="Model depth")
parser.add_argument("--n-embd", type=int, default=128, help="Embedding dim")
parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
parser.add_argument("--vocab-size", type=int, default=4096, help="BPE vocab size")
# Training
parser.add_argument("--steps", type=int, default=500, help="Training steps")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--warmup-steps", type=int, default=50, help="LR warmup steps")
parser.add_argument("--eval-every", type=int, default=50, help="Eval interval")
# Data
parser.add_argument("--max-train-stories", type=int, default=50000, help="Max stories to load (-1=all)")
parser.add_argument("--tokenizer-stories", type=int, default=10000, help="Stories for tokenizer training")
# Output
parser.add_argument("--export-dir", type=str, default="exports/tinystories", help="Export directory")
parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training (reuse cached)")
args = parser.parse_args()

print0("=" * 60)
print0("  flaxchat — TinyStories Full Pipeline")
print0("=" * 60)

# Initialize: distributed setup + mesh over ALL devices
mesh = compute_init()
num_devices = jax.device_count()

os.makedirs(args.export_dir, exist_ok=True)
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tinystories_tokenizer")

# =========================================================================
# STAGE 1: Download TinyStories
# =========================================================================
print0("=" * 60)
print0("  Stage 1: Loading TinyStories dataset")
print0("=" * 60)

from datasets import load_dataset

t0 = time.time()
print0("Loading from HuggingFace (roneneldan/TinyStories)...")
ds = load_dataset("roneneldan/TinyStories", split="train")
val_ds = load_dataset("roneneldan/TinyStories", split="validation")

if args.max_train_stories > 0:
    ds = ds.select(range(min(args.max_train_stories, len(ds))))

print0(f"Train: {len(ds):,} stories, Val: {len(val_ds):,} stories")
print0(f"Loaded in {time.time() - t0:.1f}s")

# Quick look at a sample
sample = ds[0]["text"]
print0(f"\nSample story ({len(sample)} chars):")
print0(sample[:200] + "..." if len(sample) > 200 else sample)

# =========================================================================
# STAGE 2: Train BPE tokenizer
# =========================================================================
print0("\n" + "=" * 60)
print0("  Stage 2: Training BPE tokenizer")
print0("=" * 60)

tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")

if args.skip_tokenizer and os.path.exists(tokenizer_pkl):
    print0(f"Reusing cached tokenizer from {tokenizer_dir}")
    from flaxchat.tokenizer import RustBPETokenizer
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
else:
    t0 = time.time()
    from flaxchat.tokenizer import RustBPETokenizer

    # Iterator over stories for tokenizer training
    def tokenizer_text_iter():
        for i in range(min(args.tokenizer_stories, len(ds))):
            yield ds[i]["text"]

    print0(f"Training BPE tokenizer (vocab={args.vocab_size}) on {args.tokenizer_stories:,} stories...")
    tokenizer = RustBPETokenizer.train_from_iterator(tokenizer_text_iter(), args.vocab_size)
    tokenizer.save(tokenizer_dir)
    print0(f"Tokenizer trained in {time.time() - t0:.1f}s")

actual_vocab = tokenizer.get_vocab_size()
print0(f"Vocab size: {actual_vocab}")

# Test tokenization
test = "Once upon a time, there was a little girl named Lily."
tokens = tokenizer.encode(test)
decoded = tokenizer.decode(tokens)
print0(f"Test: '{test}' -> {len(tokens)} tokens -> '{decoded}'")
print0(f"Compression ratio: {len(test) / len(tokens):.1f}x")

# =========================================================================
# STAGE 3: Tokenize dataset
# =========================================================================
print0("\n" + "=" * 60)
print0("  Stage 3: Tokenizing dataset")
print0("=" * 60)

t0 = time.time()
bos = tokenizer.get_bos_token_id()

# Tokenize all stories
print0("Tokenizing train set...")
all_tokens = []
for i in range(len(ds)):
    text = ds[i]["text"]
    tokens = tokenizer.encode(text, prepend=bos)
    all_tokens.extend(tokens)
    if (i + 1) % 10000 == 0:
        print0(f"  {i + 1:,}/{len(ds):,} stories...")

train_tokens = np.array(all_tokens, dtype=np.int32)
print0(f"Train tokens: {len(train_tokens):,}")

# Tokenize val set (smaller)
val_tokens_list = []
for i in range(min(5000, len(val_ds))):
    tokens = tokenizer.encode(val_ds[i]["text"], prepend=bos)
    val_tokens_list.extend(tokens)
val_tokens = np.array(val_tokens_list, dtype=np.int32)
print0(f"Val tokens: {len(val_tokens):,}")
print0(f"Tokenized in {time.time() - t0:.1f}s")


def get_batch(data, batch_size, seq_len, key):
    """Get a batch and place it on the mesh for data-parallel training."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    max_start = len(data) - seq_len - 1
    starts = jax.random.randint(key, (batch_size,), 0, max_start)
    inputs = np.zeros((batch_size, seq_len), dtype=np.int32)
    targets = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        s = int(starts[i])
        inputs[i] = data[s:s + seq_len]
        targets[i] = data[s + 1:s + seq_len + 1]
    # Place on mesh — shard batch across 'data' axis for multi-device
    data_sharding = NamedSharding(mesh, P('data'))
    return jax.device_put(jnp.array(inputs), data_sharding), \
           jax.device_put(jnp.array(targets), data_sharding)


# =========================================================================
# STAGE 4: Build model
# =========================================================================
print0("\n" + "=" * 60)
print0("  Stage 4: Building model")
print0("=" * 60)

n_head = max(2, args.n_embd // 32)
# Ensure n_embd is divisible by n_head
n_embd = (args.n_embd // n_head) * n_head

config = GPTConfig(
    sequence_len=args.seq_len,
    vocab_size=actual_vocab,
    n_layer=args.depth,
    n_head=n_head,
    n_kv_head=n_head,
    n_embd=n_embd,
    window_pattern="L",
)

model = GPT(config, rngs=nnx.Rngs(42))
num_params = model.num_params()
print0(f"Model: depth={args.depth}, dim={n_embd}, heads={n_head}, seq_len={args.seq_len}")
print0(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

# Replicate model across all devices in mesh
state = nnx.state(model)
state = replicate_on_mesh(state, mesh)
nnx.update(model, state)
print0(f"Model replicated across {num_devices} devices")

# =========================================================================
# STAGE 5: Train
# =========================================================================
print0("\n" + "=" * 60)
print0("  Stage 5: Pretraining")
print0("=" * 60)

# LR schedule: warmup + cosine decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=args.lr,
    warmup_steps=args.warmup_steps,
    decay_steps=args.steps,
    end_value=args.lr * 0.1,
)
tx = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.1)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, inputs, targets):
    # Shard data across all devices (data-parallel)
    inputs = jax.lax.with_sharding_constraint(inputs, NamedSharding(mesh, P('data')))
    targets = jax.lax.with_sharding_constraint(targets, NamedSharding(mesh, P('data')))

    def loss_fn(model):
        return model(inputs, targets)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, inputs, targets):
    return model(inputs, targets)


print0(f"\nTraining for {args.steps} steps (batch={args.batch_size}, seq={args.seq_len})...")
print0(f"{'step':>6} {'train':>8} {'val':>8} {'dt_ms':>7} {'tok/s':>10} {'eta':>8}")
print0("-" * 55)

best_val = float('inf')
train_losses = []
val_losses = []
t_total = 0.0
t_start = time.time()

for step in range(args.steps):
    t0 = time.time()

    key = jax.random.key(step)
    inputs, targets = get_batch(train_tokens, args.batch_size, args.seq_len, key)
    loss = train_step(model, optimizer, inputs, targets)
    loss_f = float(loss)
    train_losses.append(loss_f)

    dt = time.time() - t0
    if step > 5:
        t_total += dt

    if step % args.eval_every == 0 or step == args.steps - 1:
        # Eval
        val_loss_sum = 0.0
        n_eval = 5
        for e in range(n_eval):
            vi, vt = get_batch(val_tokens, args.batch_size, args.seq_len, jax.random.key(step + 1000 + e))
            val_loss_sum += float(eval_step(model, vi, vt))
        val_f = val_loss_sum / n_eval
        val_losses.append(val_f)
        if val_f < best_val:
            best_val = val_f

        tok_s = int(args.batch_size * args.seq_len / dt) if dt > 0 else 0
        steps_done = max(step - 5, 1)
        avg_dt = t_total / steps_done if steps_done > 0 and t_total > 0 else dt
        eta = avg_dt * (args.steps - step)
        eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta / 60:.1f}m"

        print0(f"{step:6d} {loss_f:8.4f} {val_f:8.4f} {dt*1000:7.1f} {tok_s:10,} {eta_str:>8}")

wall_time = time.time() - t_start
print0(f"\nTraining complete in {wall_time:.1f}s")
print0(f"Best val loss: {best_val:.4f}")
print0(f"Final train loss: {train_losses[-1]:.4f}")

# =========================================================================
# STAGE 6: Generate samples
# =========================================================================
print0("\n" + "=" * 60)
print0("  Stage 6: Generating samples")
print0("=" * 60)

prompts = [
    "Once upon a time",
    "The little dog",
    "A girl named Lily",
    "One day, the sun",
    "There was a big",
]

for i, prompt in enumerate(prompts):
    tokens = tokenizer.encode(prompt, prepend=bos)
    output = generate_with_cache(model, tokens, max_tokens=150, temperature=0.8, top_k=40, seed=i)
    text = tokenizer.decode(output)
    print0(f"\n--- Prompt: '{prompt}' ---")
    print0(text[:400])

# =========================================================================
# STAGE 7: Export
# =========================================================================
print0("\n" + "=" * 60)
print0("  Stage 7: Exporting model")
print0("=" * 60)

# Save checkpoint
model_state = nnx.state(model, nnx.Param)
model_dict = nnx.to_pure_dict(model_state)

ckpt_file = os.path.join(args.export_dir, "model.pkl")
with open(ckpt_file, "wb") as f:
    pickle.dump({
        "params": jax.tree.map(lambda x: np.array(x), model_dict),
        "config": {
            "sequence_len": config.sequence_len,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd,
            "window_pattern": config.window_pattern,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "best_val_loss": best_val,
            "final_train_loss": train_losses[-1],
            "wall_time_seconds": wall_time,
            "num_params": num_params,
        },
    }, f)
print0(f"Checkpoint: {ckpt_file} ({os.path.getsize(ckpt_file) / 1024:.0f} KB)")

# Save numpy weights
weights_path = os.path.join(args.export_dir, "weights.npz")
flat_params = {}
for path, leaf in jax.tree.leaves_with_path(model_dict):
    key = "/".join(str(p) for p in path)
    flat_params[key] = np.array(leaf)
np.savez(weights_path, **flat_params)
print0(f"Weights:    {weights_path} ({os.path.getsize(weights_path) / 1024:.0f} KB)")

# Export StableHLO
try:
    graphdef, state = nnx.split(model)
    def model_fn(state, input_ids):
        m = nnx.merge(graphdef, state)
        return m(input_ids)

    exported = jax.export.export(
        jax.jit(partial(model_fn, state))
    )(jnp.ones((1, args.seq_len), dtype=jnp.int32))

    stablehlo_path = os.path.join(args.export_dir, "model.stablehlo")
    serialized = exported.serialize()
    with open(stablehlo_path, "wb") as f:
        f.write(serialized)
    print0(f"StableHLO:  {stablehlo_path} ({len(serialized) / 1024:.0f} KB)")
except Exception as e:
    print0(f"StableHLO export failed: {e}")

# Copy tokenizer
import shutil
tok_export_dir = os.path.join(args.export_dir, "tokenizer")
if os.path.exists(tok_export_dir):
    shutil.rmtree(tok_export_dir)
shutil.copytree(tokenizer_dir, tok_export_dir)
print0(f"Tokenizer:  {tok_export_dir}/")

# Save training curves
curves_path = os.path.join(args.export_dir, "training_curves.npz")
np.savez(curves_path,
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
    eval_steps=np.arange(0, args.steps, args.eval_every),
)
print0(f"Curves:     {curves_path}")

# =========================================================================
# Summary
# =========================================================================
print0("\n" + "=" * 60)
print0("  Summary")
print0("=" * 60)
print0(f"  Model:        {args.depth}L / {n_embd}d / {n_head}h")
print0(f"  Parameters:   {num_params:,} ({num_params / 1e6:.2f}M)")
print0(f"  Vocab:        {actual_vocab:,}")
print0(f"  Train tokens: {len(train_tokens):,}")
print0(f"  Steps:        {args.steps}")
print0(f"  Best val:     {best_val:.4f}")
print0(f"  Wall time:    {wall_time:.1f}s")
print0(f"  Exports:      {args.export_dir}/")
print0("=" * 60)
