"""
Supervised Fine-Tuning (SFT) on conversation data.

Port of nanochat's chat_sft.py for JAX/TPU.

Usage:
    python -m scripts.sft --base-model=d24 --dataset=smoltalk
"""

import os
import time
import argparse
from dataclasses import dataclass

import jax
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig
from flaxchat.common import (
    compute_init, print0, print_banner,
    get_base_dir, DummyWandb, replicate_optimizer_state,
)
from flaxchat.tokenizer import get_tokenizer
from flaxchat.checkpoint import (
    create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint,
)
from flaxchat.sft import load_conversations, make_sft_batch, train_step
from flaxchat.stages import RequestMixin, StageResult


@dataclass(frozen=True)
class SFTRequest(RequestMixin):
    resolved_config: FlaxChatConfig | None = None
    run: str = "dummy"
    base_model: str = "d12"
    dataset: str = "smoltalk"
    num_iterations: int = 500
    batch_size: int = 4
    max_seq_len: int = 2048
    learning_rate: float = 3e-5
    warmup_steps: int = 20
    save_every: int = -1


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def run(request: SFTRequest) -> StageResult:
    print_banner()
    args = request
    if args.num_iterations < 1:
        raise ValueError("num_iterations must be positive")

    # ---------------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------------
    compute_init()
    master_process = jax.process_index() == 0

    # wandb
    if args.run == "dummy" or not master_process:
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(project="flaxchat-sft", name=args.run)

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # ---------------------------------------------------------------------------
    # Load base model
    # ---------------------------------------------------------------------------
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.base_model)
    print0(f"Loading base model from {checkpoint_dir}")

    # For now, reconstruct config from base model tag
    depth = int(args.base_model.replace("d", ""))
    config = args.resolved_config or FlaxChatConfig.from_depth(
        depth=depth, vocab_size=vocab_size
    )
    model = GPT(config.model, rngs=nnx.Rngs(0))
    restore_model_from_checkpoint(model, checkpoint_dir)
    print0(f"Loaded base model: {model.num_params():,} params")

    print0(f"Loading dataset: {args.dataset}")
    conversations = load_conversations(args.dataset)
    print0(f"Loaded {len(conversations)} conversations")


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
    replicate_optimizer_state(optimizer)


    # ---------------------------------------------------------------------------
    # Train step
    # ---------------------------------------------------------------------------
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
            save_checkpoint(
                sft_ckpt_manager,
                step,
                model,
                optimizer,
                {"step": step, "resolved_config": config.to_dict()},
            )

    # Final save
    save_checkpoint(
        sft_ckpt_manager,
        args.num_iterations,
        model,
        optimizer,
        {
            "step": args.num_iterations,
            "base_model": args.base_model,
            "dataset": args.dataset,
            "resolved_config": config.to_dict(),
        },
    )
    sft_ckpt_manager.close()

    print0("SFT complete!")
    wandb_run.finish()
    return StageResult(
        stage="sft",
        resolved_config=config.to_dict(),
        metrics={"iterations": args.num_iterations, "final_loss": loss_val},
        artifact_paths=(sft_checkpoint_dir,),
    )
