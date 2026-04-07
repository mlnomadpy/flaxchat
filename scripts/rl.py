"""
Reinforcement learning on GSM8K via simplified GRPO/REINFORCE.

Simplified from GRPO:
1. No trust region / KL regularization to reference model
2. On-policy, so no PPO ratio+clip needed
3. DAPO-style token-level normalization
4. Advantage = (r - mu), not z-score

Port of nanochat's chat_rl.py for JAX/TPU.

Usage:
    python -m scripts.rl --model=d12
    python -m scripts.rl --model=d12 --run=rl-gsm8k
"""

import os
import time
import argparse
import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import optax

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig
from flaxchat.common import (
    compute_init, get_mesh, replicate_on_mesh,
    print0, print_banner, get_base_dir, DummyWandb,
)
from flaxchat.tokenizer import get_tokenizer
from flaxchat.engine import generate_with_cache
from flaxchat.checkpoint import (
    create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint,
)
from flaxchat.report import get_report

from tasks.gsm8k import GSM8K

print_banner()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="RL on GSM8K (GRPO-style)")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--model", type=str, default="d12", help="model tag")
parser.add_argument("--model-step", type=int, default=None)
parser.add_argument("--num-epochs", type=int, default=1)
parser.add_argument("--examples-per-step", type=int, default=16)
parser.add_argument("--num-samples", type=int, default=8, help="rollout samples per example")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--init-lr-frac", type=float, default=0.05)
parser.add_argument("--eval-every", type=int, default=60)
parser.add_argument("--eval-examples", type=int, default=100)
parser.add_argument("--save-every", type=int, default=60)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
mesh = compute_init()
master_process = jax.process_index() == 0

import wandb
wandb_run = DummyWandb() if args.run == "dummy" or not master_process else wandb.init(
    project="flaxchat-rl", name=args.run, config=vars(args)
)

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# Load SFT model (or base if no SFT exists)
depth = int(args.model.replace("d", ""))
config = FlaxChatConfig.from_depth(depth=depth, vocab_size=vocab_size)
model = GPT(config.model, rngs=nnx.Rngs(0))

base_dir = get_base_dir()
for ckpt_type in ["sft", "base"]:
    ckpt_dir = os.path.join(base_dir, f"{ckpt_type}_checkpoints", args.model)
    if os.path.exists(ckpt_dir):
        print0(f"Loading {ckpt_type} model from {ckpt_dir}")
        restore_model_from_checkpoint(model, ckpt_dir)
        break

# Replicate on mesh
state = nnx.state(model)
state = replicate_on_mesh(state, mesh)
nnx.update(model, state)
print0(f"Model: {model.num_params():,} params on {jax.device_count()} devices")

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Training: {num_steps} steps, {args.examples_per_step} examples/step, "
       f"{args.num_samples} samples/example")

# ---------------------------------------------------------------------------
# Rollout generator
# ---------------------------------------------------------------------------
def get_batch(step, example_idx):
    """Generate rollout samples for one example, compute rewards and advantages."""
    conversation = train_task[example_idx]
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)

    # Generate samples
    all_sequences = []
    all_rewards = []
    for s in range(args.num_samples):
        seed = hash((step, example_idx, s)) & 0x7FFFFFFF
        output = generate_with_cache(
            model, tokens,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=seed,
        )
        all_sequences.append(output)

        # Compute reward
        generated_text = tokenizer.decode(output[prefix_length:])
        reward = train_task.reward(conversation, generated_text)
        all_rewards.append(reward)

    rewards = np.array(all_rewards)
    advantages = rewards - rewards.mean()  # simple (r - mu) advantage

    # Pad sequences to same length
    max_len = max(len(s) for s in all_sequences)
    pad_token = tokenizer.encode_special("<|assistant_end|>")
    padded = np.full((len(all_sequences), max_len), pad_token, dtype=np.int32)
    masks = np.zeros((len(all_sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(all_sequences):
        padded[i, :len(seq)] = seq
        # Mask: only train on generated tokens (after prefix), not prompt or padding
        masks[i, prefix_length:len(seq)] = 1

    inputs = padded[:, :-1]
    targets = padded[:, 1:].copy()
    targets[masks[:, 1:] == 0] = -1  # ignore index

    return inputs, targets, rewards, advantages


# ---------------------------------------------------------------------------
# RL train step
# ---------------------------------------------------------------------------
replicated = NamedSharding(mesh, P())

@partial(nnx.jit, donate_argnames=("optimizer",))
def rl_train_step(model, optimizer, inputs, targets, advantages):
    """Policy gradient step: minimize -advantage * log_prob."""
    def loss_fn(model):
        logits = model(inputs)  # (B, T, V)
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Gather log probs of target tokens
        B, T, V = logits.shape
        one_hot = jax.nn.one_hot(jnp.maximum(targets, 0), V)
        token_logp = jnp.sum(log_probs * one_hot, axis=-1)  # (B, T)

        # Mask invalid tokens (targets == -1)
        valid = (targets >= 0).astype(jnp.float32)

        # PG objective: sum of advantage-weighted log probs
        pg_obj = jnp.sum(token_logp * advantages[:, None] * valid)
        num_valid = jnp.maximum(jnp.sum(valid), 1.0)
        pg_obj = pg_obj / num_valid

        return -pg_obj  # minimize negative objective = maximize objective

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
lr = args.lr * args.init_lr_frac  # start low
schedule = optax.linear_schedule(init_value=lr, end_value=0.0, transition_steps=num_steps)
tx = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.0)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def run_eval(task, max_examples=100):
    """Evaluate pass@1 on a task."""
    correct = 0
    total = 0
    for idx in range(min(max_examples, len(task))):
        conv = task[idx]
        tokens = tokenizer.render_for_completion(conv)
        output = generate_with_cache(model, tokens, max_tokens=256, temperature=0.0, seed=idx)
        response = tokenizer.decode(output[len(tokens):])
        correct += task.evaluate(conv, response)
        total += 1
        if (idx + 1) % 20 == 0:
            print0(f"  Eval: {correct}/{total} ({100*correct/total:.1f}%)")
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
rl_dir = os.path.join(base_dir, "rl_checkpoints", args.model)
rl_manager = create_checkpoint_manager(rl_dir, max_to_keep=3)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"\nStarting RL for {num_steps} steps...")
example_indices = list(range(len(train_task)))
np.random.seed(42)
np.random.shuffle(example_indices)
example_iter = itertools.cycle(example_indices)

for step in range(num_steps):
    # Eval
    if step % args.eval_every == 0:
        print0(f"\n--- Eval at step {step} ---")
        acc = run_eval(val_task, max_examples=args.eval_examples)
        print0(f"GSM8K pass@1: {acc:.4f}")
        wandb_run.log({"step": step, "pass@1": acc})

    # Collect rollouts for this step
    t0 = time.time()
    all_rewards = []
    for _ in range(args.examples_per_step):
        example_idx = next(example_iter)
        inputs, targets, rewards, advantages = get_batch(step, example_idx)

        # Put on device
        inputs_j = jax.device_put(jnp.array(inputs), replicated)
        targets_j = jax.device_put(jnp.array(targets), replicated)
        advantages_j = jax.device_put(jnp.array(advantages, dtype=jnp.float32), replicated)

        loss = rl_train_step(model, optimizer, inputs_j, targets_j, advantages_j)
        all_rewards.extend(rewards)

    dt = time.time() - t0
    mean_reward = np.mean(all_rewards)
    print0(f"Step {step}/{num_steps} | reward: {mean_reward:.4f} | loss: {float(loss):.6f} | {dt:.1f}s")
    wandb_run.log({"step": step, "reward": mean_reward, "loss": float(loss)})

    # Save
    if master_process and step > 0 and step % args.save_every == 0:
        save_checkpoint(rl_manager, step, model, optimizer, {
            "step": step, "model": args.model, "reward": mean_reward,
        })

# Final save + eval
print0("\n--- Final eval ---")
acc = run_eval(val_task, max_examples=args.eval_examples)
print0(f"Final GSM8K pass@1: {acc:.4f}")

if master_process:
    save_checkpoint(rl_manager, num_steps, model, optimizer, {
        "step": num_steps, "model": args.model, "final_pass1": acc,
    })

get_report(args.run).log("RL Training", {
    "steps": num_steps, "final_pass@1": acc,
    "examples_per_step": args.examples_per_step,
    "num_samples": args.num_samples,
})

print0("RL complete!")
wandb_run.finish()
