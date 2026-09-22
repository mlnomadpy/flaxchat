"""
Supervised Fine-Tuning (SFT) on conversation data.

Port of nanochat's chat_sft.py for JAX/TPU.

Usage:
    python -m scripts.sft --base-model=d24 --dataset=smoltalk
"""

import os
import numpy as np
import hashlib
import json
import jax.numpy as jnp
import time
import argparse
from dataclasses import dataclass

import jax
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.common import (
    compute_init, print0, print_banner,
    get_base_dir, DummyWandb, replicate_optimizer_state, replicate_on_mesh,
)
from flaxchat.tokenizer import get_tokenizer, load_tokenizer
from flaxchat.checkpoint import (
    create_checkpoint_manager, save_checkpoint, restore_model_from_checkpoint, load_checkpoint_metadata,
)
from flaxchat.sft import load_conversations, make_sft_batch, train_step
from flaxchat.stages import RequestMixin, StageResult
from flaxchat.checkpoint import validate_checkpoint_tokenizer


@dataclass(frozen=True)
class SFTRequest(RequestMixin):
    resolved_config: FlaxChatConfig | None = None
    tokenizer_dir: str | None = None
    checkpoint_dir: str | None = None
    run: str = "dummy"
    base_model: str = "d12"
    dataset: str = "smoltalk"
    num_iterations: int = 500
    batch_size: int = 4
    max_seq_len: int = 2048
    learning_rate: float = 3e-5
    warmup_steps: int = 20
    save_every: int = -1
    resume_from_step: int = -1


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
    parser.add_argument("--resume-from-step", type=int, default=-1)
    parser.add_argument("--tokenizer-dir", help="Tokenizer artifact matching the checkpoint")
    parser.add_argument("--checkpoint-dir", help="Explicit input checkpoint directory")
    return parser


def run(request: SFTRequest) -> StageResult:
    print_banner()
    args = request
    if args.num_iterations < 1:
        raise ValueError("num_iterations must be positive")

    # ---------------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------------
    mesh = compute_init()
    if jax.process_count() > 1:
        raise ValueError("SFT multi-host training is not validated; use one host")
    if args.batch_size < 1 or args.batch_size % jax.device_count():
        raise ValueError("SFT global batch must be positive and divisible by device count")
    master_process = jax.process_index() == 0

    # wandb
    if args.run == "dummy" or not master_process:
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(project="flaxchat-sft", name=args.run)

    # Tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir) if args.tokenizer_dir else get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # ---------------------------------------------------------------------------
    # Load base model
    # ---------------------------------------------------------------------------
    base_dir = get_base_dir()
    checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "base_checkpoints", args.base_model)
    print0(f"Loading base model from {checkpoint_dir}")

    base_metadata = load_checkpoint_metadata(checkpoint_dir)
    validate_checkpoint_tokenizer(base_metadata, tokenizer,
        tokenizer_path=args.tokenizer_dir or os.path.join(get_base_dir(), 'tokenizer'))
    model_config = GPTConfig(**base_metadata['model_config'])
    if model_config.vocab_size != vocab_size:
        raise ValueError("Base checkpoint tokenizer vocabulary mismatch")
    if args.max_seq_len > model_config.sequence_len:
        raise ValueError("SFT sequence length exceeds the base model context")
    config = args.resolved_config or FlaxChatConfig(model=model_config)
    if config.model != model_config:
        raise ValueError("SFT model configuration must match the base checkpoint")
    model = GPT(model_config, rngs=nnx.Rngs(0))
    if base_metadata.get('step') is None:
        raise ValueError('Base checkpoint metadata must identify a concrete step')
    restore_model_from_checkpoint(model, checkpoint_dir, step=base_metadata['step'])
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
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
        warmup_steps=min(args.warmup_steps, args.num_iterations - 1),
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
    try:

        # ---------------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------------
        print0(f"\nStarting SFT for {args.num_iterations} steps...")

        from flaxchat.dataloader import _tokenizer_identity
        identity = {
            "resolved_config": {"model": base_metadata['model_config'], "iterations": args.num_iterations,
                                "batch_size": args.batch_size, "max_seq_len": args.max_seq_len,
                                "learning_rate": args.learning_rate, "warmup_steps": args.warmup_steps, "seed": 0},
            "data_manifest": hashlib.sha256(json.dumps(conversations, sort_keys=True).encode()).hexdigest(),
            "tokenizer": _tokenizer_identity(tokenizer),
        }
        metadata = {"model_config": base_metadata['model_config'],
                    "resolved_config": identity['resolved_config'],
                    "data_manifest_identity": identity['data_manifest'],
                    "tokenizer_identity": identity['tokenizer'], "base_model": args.base_model}
        start = 0
        if args.resume_from_step >= 0:
            _, cursor = restore_model_from_checkpoint(model, sft_checkpoint_dir,
                          step=args.resume_from_step, optimizer=optimizer,
                          expected_identity=identity, load_training_state=True)
            if cursor is None:
                raise ValueError('Checkpoint lacks training cursor')
            start = int(cursor['update_step'])
            if start != args.resume_from_step or start > args.num_iterations:
                raise ValueError("Invalid SFT resume cursor")
            optimizer.step[...] = start
        def save(completed):
            save_checkpoint(sft_ckpt_manager, completed, model, optimizer,
                            {**metadata, "step": completed},
                            training_state=replicate_on_mesh({"update_step": jnp.asarray(completed, jnp.int32)}, mesh))
        loss_val = None
        for step in range(start, args.num_iterations):
            t0 = time.time()

            key = jax.random.key(step)
            inputs, targets = make_sft_batch(conversations, tokenizer, args.batch_size, args.max_seq_len, key)
            loss = train_step(model, optimizer, inputs, targets)
            if not np.isfinite(float(loss)):
                raise FloatingPointError("Nonfinite SFT update rejected; optimizer state preserved")

            loss_val = float(loss)
            dt = time.time() - t0

            if step % 10 == 0:
                print0(f"step {step:04d}/{args.num_iterations} | loss: {loss_val:.4f} | dt: {dt*1000:.0f}ms")
                wandb_run.log({"step": step, "sft/loss": loss_val, "sft/dt": dt})

            completed = step + 1
            if args.save_every > 0 and completed % args.save_every == 0 and completed < args.num_iterations:
                save(completed)

        if start < args.num_iterations:
            save(args.num_iterations)
    finally:
        sft_ckpt_manager.close()

    print0("SFT complete!")
    wandb_run.finish()
    return StageResult(
        stage="sft",
        resolved_config=config.to_dict(),
        metrics={"iterations": args.num_iterations, "final_loss": loss_val},
        artifact_paths=(sft_checkpoint_dir,),
    )
