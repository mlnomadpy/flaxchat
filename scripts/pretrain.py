"""
Pretrain base model on TPU pod.

Usage:
    # Single host
    python -m scripts.pretrain --depth=12

    # Multi-host TPU pod (via XLA flags or SLURM)
    python -m scripts.pretrain --depth=24

    # Quick test
    python -m scripts.pretrain --depth=4 --num-iterations=20 --device-batch-size=1
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
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.common import (
    compute_init, setup_mesh, shard_batch, replicate_on_mesh,
    print0, print_banner, get_base_dir, get_peak_flops,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, DummyWandb,
)
from flaxchat.tokenizer import get_tokenizer
from flaxchat.dataloader import data_loader_bos_bestfit
from flaxchat.optim import setup_optimizer, make_lr_schedule, make_weight_decay_schedule
from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
from flaxchat.engine import Engine

print_banner()

# ---------------------------------------------------------------------------
# CLI arguments (mirrors nanochat)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Pretrain base model on TPU")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
parser.add_argument("--depth", type=int, default=20, help="Transformer depth")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-flops", type=float, default=-1.0)
parser.add_argument("--target-param-data-ratio", type=float, default=12)
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1, help="-1 = auto-compute optimal")
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.008)
parser.add_argument("--weight-decay", type=float, default=0.28)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--resume-from-step", type=int, default=-1)
# Evaluation
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
parser.add_argument("--sample-every", type=int, default=2000)
parser.add_argument("--save-every", type=int, default=-1)
# Output
parser.add_argument("--model-tag", type=str, default=None)
args = parser.parse_args()
user_config = vars(args).copy()

# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------
# Compute init: distributed setup + mesh creation over ALL devices
mesh = compute_init()
master_process = jax.process_index() == 0
num_devices = jax.device_count()

# TPU peak FLOPS
peak_flops = get_peak_flops()
print0(f"Peak FLOPS (BF16) per device: {peak_flops:.2e}")

# wandb
import wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="flaxchat", name=args.run, config=user_config)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# ---------------------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------------------
config = FlaxChatConfig.from_depth(
    depth=args.depth,
    aspect_ratio=args.aspect_ratio,
    head_dim=args.head_dim,
    max_seq_len=args.max_seq_len,
    window_pattern=args.window_pattern,
    vocab_size=vocab_size,
)
model_config = config.model
print0(f"Model config:\n{json.dumps(asdict(model_config), indent=2)}")

model = GPT(model_config, rngs=nnx.Rngs(0))
num_params = model.num_params()
num_flops_per_token = model.estimate_flops()
print0(f"Parameters: {num_params:,}")
print0(f"FLOPs per token: {num_flops_per_token:e}")

# Replicate model params across all devices in the mesh
state = nnx.state(model)
state = replicate_on_mesh(state, mesh)
nnx.update(model, state)
print0(f"Model replicated across {num_devices} devices")

# ---------------------------------------------------------------------------
# Scaling laws (same as nanochat)
# ---------------------------------------------------------------------------
def build_model_meta(depth):
    """Build model config for scaling reference."""
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    cfg = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    ref_model = GPT(cfg, rngs=nnx.Rngs(42))
    # Scaling params = block params + lm_head params
    all_p = jax.tree.leaves(nnx.state(ref_model, nnx.Param))
    total = sum(p.size for p in all_p)
    wte_size = ref_model.wte.embedding.value.size
    ve_size = sum(ve.embedding.value.size for ve in ref_model.value_embeds.values())
    scalar_size = (ref_model.resid_lambdas.value.size + ref_model.x0_lambdas.value.size +
                   ref_model.smear_gate.kernel.value.size + ref_model.smear_lambda.value.size +
                   ref_model.backout_lambda.value.size)
    scaling_params = total - wte_size - ve_size - scalar_size
    return scaling_params

num_scaling_params = build_model_meta(args.depth)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

D_REF = args.target_param_data_ratio * build_model_meta(12)
B_REF = 2**19  # ~524K tokens

# Auto-compute batch size
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# LR scaling
batch_lr_scale = (total_batch_size / B_REF) ** 0.5
if batch_lr_scale != 1.0:
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")

# Weight decay scaling (T_epoch framework)
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
config.training.embedding_lr = args.embedding_lr
config.training.unembedding_lr = args.unembedding_lr
config.training.matrix_lr = args.matrix_lr
config.training.scalar_lr = args.scalar_lr

optimizer = setup_optimizer(model, config, batch_lr_scale, weight_decay_scaled,
                           lr_schedule_fn=lr_schedule)

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
train_loader = data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train",
)

# ---------------------------------------------------------------------------
# Training iterations
# ---------------------------------------------------------------------------
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
else:
    num_iterations = target_tokens // total_batch_size
total_tokens = total_batch_size * num_iterations
print0(f"Training iterations: {num_iterations:,}")
print0(f"Total training tokens: {total_tokens:,}")

# Grad accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * num_devices
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")

# LR schedule
lr_schedule = make_lr_schedule(num_iterations, args.warmup_steps, args.warmdown_ratio, args.final_lr_frac)
wd_schedule = make_weight_decay_schedule(num_iterations, weight_decay_scaled)

# ---------------------------------------------------------------------------
# JIT-compiled train step with automatic data sharding
# Data is sharded across the 'data' mesh axis. Gradients are automatically
# averaged across devices by JAX's SPMD — no manual all-reduce needed.
# ---------------------------------------------------------------------------
@partial(nnx.jit, donate_argnames=("optimizer",))
def train_step(model, optimizer, inputs, targets):
    """Single training step with data-parallel sharding across all devices."""
    # Shard inputs across devices (batch dimension along 'data' axis)
    inputs = jax.lax.with_sharding_constraint(inputs, NamedSharding(mesh, P('data')))
    targets = jax.lax.with_sharding_constraint(targets, NamedSharding(mesh, P('data')))

    def loss_fn(model):
        return model(inputs, targets)

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # NaN guard
    grad_finite = jax.tree.reduce(
        lambda x, y: x & jnp.all(jnp.isfinite(y)),
        grads, initializer=True,
    )
    grads = jax.tree.map(
        lambda g: jnp.where(grad_finite, g, jnp.zeros_like(g)),
        grads,
    )

    optimizer.update(model, grads)

    grad_norm = jnp.sqrt(jax.tree.reduce(
        lambda x, y: x + jnp.sum(y ** 2), grads, initializer=0.0
    ))

    return loss, grad_norm, grad_finite


@partial(nnx.jit, donate_argnames=("optimizer",))
def train_step_grad_accum(model, optimizer, all_inputs, all_targets, num_accum_steps):
    """
    Training step with gradient accumulation via jax.lax.scan.
    all_inputs: (num_accum, B, T) — stacked micro-batches
    all_targets: (num_accum, B, T)
    """
    def micro_step(acc_grads, micro_batch):
        micro_inputs, micro_targets = micro_batch

        def loss_fn(model):
            return model(micro_inputs, micro_targets)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        # Accumulate gradients
        new_acc = jax.tree.map(lambda a, g: a + g, acc_grads, grads)
        return new_acc, loss

    # Initialize accumulated grads to zeros
    zero_grads = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))

    # Scan over micro-batches
    acc_grads, losses = jax.lax.scan(
        micro_step, zero_grads, (all_inputs, all_targets)
    )

    # Average gradients
    avg_grads = jax.tree.map(lambda g: g / num_accum_steps, acc_grads)
    avg_loss = jnp.mean(losses)

    # NaN guard
    grad_finite = jax.tree.reduce(
        lambda x, y: x & jnp.all(jnp.isfinite(y)),
        avg_grads, initializer=True,
    )
    avg_grads = jax.tree.map(
        lambda g: jnp.where(grad_finite, g, jnp.zeros_like(g)),
        avg_grads,
    )

    optimizer.update(model, avg_grads)

    grad_norm = jnp.sqrt(jax.tree.reduce(
        lambda x, y: x + jnp.sum(y ** 2), avg_grads, initializer=0.0
    ))

    return avg_loss, grad_norm, grad_finite


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
ckpt_manager = create_checkpoint_manager(checkpoint_dir, max_to_keep=3)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
step = 0
smooth_train_loss = 0.0
total_training_time = 0.0

print0(f"\nStarting training for {num_iterations} steps...")

val_loader = data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val") if args.eval_every > 0 else None
val_loss = None
min_val_loss = float('inf')

while True:
    last_step = step == num_iterations

    # Evaluate val loss
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        val_sum, val_count = 0.0, 0
        eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * num_devices))
        for _ in range(eval_steps):
            vi, vt, _ = next(val_loader) if val_loader else (None, None, None)
            if vi is None:
                break
            vi_j, vt_j = jnp.array(vi), jnp.array(vt)
            vl = model(vi_j, vt_j)
            val_sum += float(vl)
            val_count += 1
        if val_count > 0:
            val_loss = val_sum / val_count
            if val_loss < min_val_loss:
                min_val_loss = val_loss
            print0(f"Step {step:05d} | Val loss: {val_loss:.6f}")

    # Sampling
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "The planets of the solar system are:",
        ]
        engine = Engine(model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            all_tokens, texts = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(texts[0])

    # Save checkpoint
    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(ckpt_manager, step, model, optimizer, {
            "step": step,
            "model_config": asdict(model_config),
            "user_config": user_config,
            "total_batch_size": total_batch_size,
        })

    if last_step:
        break

    # ------- Single training step (with gradient accumulation) -------
    t0 = time.time()

    if grad_accum_steps == 1:
        # Fast path: no accumulation needed
        inputs_np, targets_np, dataloader_state = next(train_loader)
        # For multi-host: each host loads its own shard, create global array
        if jax.process_count() > 1:
            data_sharding = NamedSharding(mesh, P('data'))
            inputs = jax.make_array_from_process_local_data(data_sharding, inputs_np)
            targets = jax.make_array_from_process_local_data(data_sharding, targets_np)
        else:
            inputs = jnp.array(inputs_np)
            targets = jnp.array(targets_np)
        loss, grad_norm, grad_finite = train_step(model, optimizer, inputs, targets)
    else:
        # Gradient accumulation: collect micro-batches then scan
        micro_inputs_list = []
        micro_targets_list = []
        for _ in range(grad_accum_steps):
            inputs_np, targets_np, dataloader_state = next(train_loader)
            micro_inputs_list.append(inputs_np)
            micro_targets_list.append(targets_np)
        all_inputs = jnp.array(np.stack(micro_inputs_list))   # (num_accum, B, T)
        all_targets = jnp.array(np.stack(micro_targets_list))  # (num_accum, B, T)
        loss, grad_norm, grad_finite = train_step_grad_accum(
            model, optimizer, all_inputs, all_targets, grad_accum_steps
        )

    # Force sync for timing
    loss_val = float(loss)
    t1 = time.time()
    dt = t1 - t0

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

    if step > 10:
        total_training_time += dt

    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
    flops_per_sec = num_flops_per_token * total_batch_size / dt if dt > 0 else 0
    mfu = 100 * flops_per_sec / (peak_flops * num_devices) if peak_flops < float('inf') else 0

    steps_done = step - 10
    if steps_done > 0:
        avg_time = total_training_time / steps_done
        eta_seconds = (num_iterations - step) * avg_time
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""

    epoch_info = f"ep:{dataloader_state['epoch']} pq:{dataloader_state['pq_idx']} rg:{dataloader_state['rg_idx']}"
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
        f"loss: {debiased_loss:.6f} | dt: {dt * 1000:.0f}ms | "
        f"tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}% | "
        f"{epoch_info}{eta_str}"
    )

    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased_loss,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/grad_norm": float(grad_norm),
        })

    step += 1

    # GC management (like nanochat)
    if step == 1:
        gc.collect()

# Cleanup
print0(f"\nTraining complete! Total time: {total_training_time / 60:.1f}m")
wandb_run.finish()
