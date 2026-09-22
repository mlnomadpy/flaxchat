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
import hashlib
import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import optax

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.common import (
    compute_init, replicate_on_mesh, replicate_optimizer_state,
    print0, print_banner, get_base_dir, DummyWandb,
)
from flaxchat.tokenizer import get_tokenizer, load_tokenizer
from flaxchat.engine import generate_with_cache
from flaxchat.checkpoint import (
    create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint, load_checkpoint_metadata,
)
from flaxchat.report import get_report
from flaxchat.rl import centered_advantages, train_step as rl_train_step
from flaxchat.stages import RequestMixin, StageResult
from flaxchat.checkpoint import validate_checkpoint_tokenizer

from tasks.gsm8k import GSM8K


@dataclass(frozen=True)
class RLRequest(RequestMixin):
    resolved_config: FlaxChatConfig | None = None
    tokenizer_dir: str | None = None
    checkpoint_dir: str | None = None
    run: str = "dummy"
    model: str = "d12"
    model_step: int | None = None
    num_epochs: int = 1
    examples_per_step: int = 16
    num_samples: int = 8
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    lr: float = 1e-5
    init_lr_frac: float = 0.05
    eval_every: int = 60
    eval_examples: int = 100
    save_every: int = 60
    resume_from_step: int = -1


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--resume-from-step", type=int, default=-1)
    parser.add_argument("--tokenizer-dir", help="Tokenizer artifact matching the checkpoint")
    parser.add_argument("--checkpoint-dir", help="Explicit input checkpoint directory")
    return parser


def run(request: RLRequest) -> StageResult:
    print_banner()
    args = request
    if min(
        args.num_epochs,
        args.examples_per_step,
        args.num_samples,
        args.eval_every,
        args.eval_examples,
        args.save_every,
    ) < 1:
        raise ValueError("epoch, batch, sample, evaluation, and save counts must be positive")

    # ---------------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------------
    mesh = compute_init()
    if jax.process_count() > 1:
        raise ValueError("RL multi-host updates are not validated; use one host")
    master_process = jax.process_index() == 0

    if args.run == "dummy" or not master_process:
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(
            project="flaxchat-rl", name=args.run, config=vars(args)
        )

    tokenizer = load_tokenizer(args.tokenizer_dir) if args.tokenizer_dir else get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Load SFT model (or base if no SFT exists)
    base_dir = get_base_dir()
    for ckpt_type in ["sft", "base"]:
        ckpt_dir = args.checkpoint_dir or os.path.join(base_dir, f"{ckpt_type}_checkpoints", args.model)
        if args.checkpoint_dir or os.path.exists(ckpt_dir):
            print0(f"Loading {ckpt_type} model from {ckpt_dir}")
            metadata = load_checkpoint_metadata(ckpt_dir, step=args.model_step)
            validate_checkpoint_tokenizer(metadata, tokenizer,
                tokenizer_path=args.tokenizer_dir or os.path.join(get_base_dir(), 'tokenizer'))
            model_config = GPTConfig(**metadata["model_config"])
            if model_config.vocab_size != vocab_size:
                raise ValueError("RL tokenizer vocabulary differs from checkpoint")
            config = args.resolved_config or FlaxChatConfig(model=model_config)
            if config.model != model_config:
                raise ValueError("RL architecture differs from checkpoint")
            model = GPT(config.model, rngs=nnx.Rngs(0))
            if metadata.get('step') is None:
                raise ValueError('Checkpoint metadata must identify a concrete step')
            restore_model_from_checkpoint(model, ckpt_dir, step=metadata['step'])
            break
    else:
        raise FileNotFoundError(
            f"No SFT or base checkpoint found for model {args.model!r} under {base_dir}"
        )

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
    if num_steps < 1:
        raise ValueError("RL dataset must cover at least one outer step")
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

        if prefix_length + args.max_new_tokens > config.model.sequence_len:
            raise ValueError("RL prompt plus generation exceeds model context")

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
        advantages = np.asarray(centered_advantages(rewards))

        # Pad sequences to same length
        max_len = config.model.sequence_len + 1
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

    # ---------------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------------
    lr = args.lr * args.init_lr_frac  # start low
    schedule = optax.linear_schedule(init_value=lr, end_value=0.0, transition_steps=num_steps * args.examples_per_step)
    tx = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.0)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    replicate_optimizer_state(optimizer, mesh)

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
    try:

        # ---------------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------------
        print0(f"\nStarting RL for {num_steps} steps...")
        example_indices = list(range(len(train_task)))
        np.random.seed(42)
        np.random.shuffle(example_indices)
        example_iter = itertools.cycle(example_indices)

        from flaxchat.dataloader import _tokenizer_identity
        data_digest = hashlib.sha256()
        for index in range(len(train_task)):
            data_digest.update(json.dumps(train_task[index], sort_keys=True).encode() + b'\n')
        resolved = {"model": metadata['model_config'], "steps": num_steps,
                    "examples_per_step": args.examples_per_step, "num_samples": args.num_samples,
                    "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                    "top_k": args.top_k, "learning_rate": lr, "seed": 42,
                    "objective": "centered-reward-token-mean-v1"}
        identity = {"resolved_config": resolved, "tokenizer": _tokenizer_identity(tokenizer),
                    "data_manifest": data_digest.hexdigest()}
        run_metadata = {"model_config": metadata['model_config'], "resolved_config": resolved,
                        "tokenizer_identity": identity['tokenizer'],
                        "data_manifest_identity": identity['data_manifest']}
        start = 0
        if args.resume_from_step >= 0:
            _, cursor = restore_model_from_checkpoint(model, rl_dir, step=args.resume_from_step,
                         optimizer=optimizer, expected_identity=identity, load_training_state=True)
            if cursor is None:
                raise ValueError('Checkpoint lacks training cursor')
            start = int(cursor['outer_step'])
            if start != args.resume_from_step or start > num_steps or int(cursor['optimizer_updates']) != start * args.examples_per_step:
                raise ValueError("Invalid RL resume cursor")
            optimizer.step[...] = start * args.examples_per_step
            for _ in range(start * args.examples_per_step):
                next(example_iter)
        def save(completed, extra):
            save_checkpoint(rl_manager, completed, model, optimizer,
                            {**run_metadata, **extra, "step": completed},
                            training_state=replicate_on_mesh({
                                "outer_step": np.asarray(completed, np.int32),
                                "optimizer_updates": np.asarray(completed * args.examples_per_step, np.int32)}, mesh))

        for step in range(start, num_steps):
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
                if not np.isfinite(float(loss)):
                    raise FloatingPointError("Nonfinite RL update rejected; optimizer state preserved")
                all_rewards.extend(rewards)

            dt = time.time() - t0
            mean_reward = np.mean(all_rewards)
            print0(f"Step {step}/{num_steps} | reward: {mean_reward:.4f} | loss: {float(loss):.6f} | {dt:.1f}s")
            wandb_run.log({"step": step, "reward": mean_reward, "loss": float(loss)})

            # Save
            if master_process and args.save_every > 0 and (step + 1) % args.save_every == 0 and step + 1 < num_steps:
                save(step + 1, {"reward": float(mean_reward)})

        # Final save + eval
        print0("\n--- Final eval ---")
        acc = run_eval(val_task, max_examples=args.eval_examples)
        print0(f"Final GSM8K pass@1: {acc:.4f}")

        if master_process and start < num_steps:
            save(num_steps, {"final_pass1": acc})
    finally:
        rl_manager.close()

    get_report(args.run).log("RL Training", {
        "steps": num_steps, "final_pass@1": acc,
        "examples_per_step": args.examples_per_step,
        "num_samples": args.num_samples,
    })

    print0("RL complete!")
    wandb_run.finish()
    return StageResult(
        stage="rl",
        resolved_config=config.to_dict(),
        metrics={"steps": num_steps, "final_pass1": acc},
        artifact_paths=(rl_dir,),
    )
