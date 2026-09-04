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
import subprocess
from functools import partial
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx

from flaxchat.gpt import GPT, attention_backend_metadata
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.common import (
    compute_init, replicate_on_mesh,
    replicate_optimizer_state,
    print0, print_banner, get_base_dir, get_peak_flops,
    DummyWandb,
)
from flaxchat.tokenizer import get_tokenizer
from flaxchat.dataloader import data_loader_bos_bestfit
from flaxchat.optim import setup_optimizer, make_lr_schedule
from flaxchat.checkpoint import (
    create_checkpoint_manager,
    restore_model_from_checkpoint,
    save_checkpoint,
)
from flaxchat.engine import Engine
from flaxchat.training import (
    accumulation_dtype,
    apply_gradients_if_finite,
    gradients_for_microbatches,
    tree_all_finite,
)

def run(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--gradient-accumulation-dtype", type=str, default="float32",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--resume-from-step", type=int, default=-1)
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
    parser.add_argument("--sample-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=-1)
    # Output
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="run two deterministic optimizer steps without external data")
    args = parser.parse_args(argv)
    if args.cpu_smoke:
        args.depth = 2
        args.aspect_ratio = 16
        args.head_dim = 16
        args.max_seq_len = 16
        args.device_batch_size = 2
        args.total_batch_size = 32
        args.num_iterations = 2
        args.eval_every = 0
        args.sample_every = 0
        args.save_every = -1
        args.model_tag = args.model_tag or "cpu-smoke"
    user_config = vars(args).copy()
    try:
        source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        source_revision = "unavailable"

    # ---------------------------------------------------------------------------
    # Distributed init
    # ---------------------------------------------------------------------------
    # Compute init: distributed setup + mesh creation over ALL devices
    mesh = compute_init()
    master_process = jax.process_index() == 0
    num_devices = jax.device_count()
    if args.cpu_smoke:
        # Keep the smoke run to one forward/backward pass per update on any device
        # count. The batch dimension itself must also divide the data mesh.
        args.device_batch_size = max(args.device_batch_size, num_devices)
        args.total_batch_size = args.device_batch_size * args.max_seq_len * num_devices
        user_config["device_batch_size"] = args.device_batch_size
        user_config["total_batch_size"] = args.total_batch_size

    # TPU peak FLOPS
    peak_flops = get_peak_flops()
    print0(f"Peak FLOPS (BF16) per device: {peak_flops:.2e}")

    # wandb
    use_dummy_wandb = args.run == "dummy" or not master_process
    if use_dummy_wandb:
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(project="flaxchat", name=args.run, config=user_config)

    # ---------------------------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------------------------
    if args.cpu_smoke:
        class _SmokeTokenizer:
            def get_vocab_size(self):
                return 256

            def get_bos_token_id(self):
                return 0

        tokenizer = _SmokeTokenizer()
    else:
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
        wte_size = ref_model.wte.embedding[...].size
        ve_size = sum(ve.embedding[...].size for ve in ref_model.value_embeds.values())
        scalar_size = (ref_model.resid_lambdas[...].size + ref_model.x0_lambdas[...].size +
                       ref_model.smear_gate.kernel[...].size + ref_model.smear_lambda[...].size +
                       ref_model.backout_lambda[...].size)
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

    # Resolve the complete training horizon before constructing any schedule or
    # optimizer. This ordering is part of the serialized run contract.
    assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    elif args.target_flops > 0:
        num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    else:
        num_iterations = target_tokens // total_batch_size
    total_tokens = total_batch_size * num_iterations

    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * num_devices
    assert total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

    lr_schedule = make_lr_schedule(
        num_iterations, args.warmup_steps, args.warmdown_ratio, args.final_lr_frac
    )
    user_config.update({
        "effective_global_batch_tokens": total_batch_size,
        "total_update_count": num_iterations,
        "warmup_count": args.warmup_steps,
        "gradient_accumulation_steps": grad_accum_steps,
        "weight_decay_scaled": weight_decay_scaled,
    })

    print0(f"Training iterations: {num_iterations:,}")
    print0(f"Total training tokens: {total_tokens:,}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    # ---------------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------------
    config.training.embedding_lr = args.embedding_lr
    config.training.unembedding_lr = args.unembedding_lr
    config.training.matrix_lr = args.matrix_lr
    config.training.scalar_lr = args.scalar_lr
    config.training.gradient_accumulation_dtype = args.gradient_accumulation_dtype

    optimizer = setup_optimizer(model, config, batch_lr_scale, weight_decay_scaled,
                               lr_schedule_fn=lr_schedule)
    replicate_optimizer_state(optimizer, mesh)

    # ---------------------------------------------------------------------------
    # Checkpoint restore (must happen before constructing the resumable loader)
    # ---------------------------------------------------------------------------
    base_dir = get_base_dir()
    output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
    step = 0
    microbatches_processed = 0
    successful_updates = 0
    skipped_updates = 0
    resume_dataloader_state = None
    if args.resume_from_step >= 0:
        restored_metadata, restored_training_state = restore_model_from_checkpoint(
            model,
            checkpoint_dir,
            step=args.resume_from_step,
            optimizer=optimizer,
            load_training_state=True,
        )
        step = int(restored_training_state["update_step"])
        if step != args.resume_from_step:
            raise ValueError(
                f"Checkpoint step mismatch: requested {args.resume_from_step}, restored {step}"
            )
        resume_dataloader_state = restored_metadata.get("dataloader_state")
        microbatches_processed = int(restored_metadata.get("microbatches_processed", 0))
        successful_updates = int(restored_metadata.get("successful_updates", step))
        skipped_updates = int(restored_metadata.get("skipped_updates", 0))
        print0(f"Resumed model, optimizer, and loader state from step {step}")
    ckpt_manager = create_checkpoint_manager(checkpoint_dir, max_to_keep=3)

    # ---------------------------------------------------------------------------
    # Dataloader
    # ---------------------------------------------------------------------------
    if args.cpu_smoke:
        def _smoke_loader(consumed_batches=0):
            rng = np.random.default_rng(1234)
            for _ in range(consumed_batches):
                rng.integers(
                    0, vocab_size,
                    size=(args.device_batch_size, args.max_seq_len + 1),
                    dtype=np.int32,
                )
            while True:
                tokens = rng.integers(
                    0, vocab_size,
                    size=(args.device_batch_size, args.max_seq_len + 1),
                    dtype=np.int32,
                )
                yield tokens[:, :-1], tokens[:, 1:], {
                    "epoch": 1, "pq_idx": 0, "rg_idx": 0,
                }
        train_loader = _smoke_loader(microbatches_processed)
    else:
        train_loader = data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="train",
            resume_state_dict=resume_dataloader_state,
        )

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

        grads = jax.tree.map(
            lambda g: g.astype(accumulation_dtype(args.gradient_accumulation_dtype)),
            grads,
        )
        grad_finite = jnp.isfinite(loss) & tree_all_finite(grads)
        apply_gradients_if_finite(model, optimizer, grads, loss)

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
        del num_accum_steps
        avg_loss, avg_grads = gradients_for_microbatches(
            model,
            all_inputs,
            all_targets,
            dtype=accumulation_dtype(args.gradient_accumulation_dtype),
        )
        grad_finite = jnp.isfinite(avg_loss) & tree_all_finite(avg_grads)
        apply_gradients_if_finite(model, optimizer, avg_grads, avg_loss)

        grad_norm = jnp.sqrt(jax.tree.reduce(
            lambda x, y: x + jnp.sum(y ** 2), avg_grads, initializer=0.0
        ))

        return avg_loss, grad_norm, grad_finite


    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    smooth_train_loss = 0.0
    total_training_time = 0.0
    dataloader_state = resume_dataloader_state

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
            checkpoint_metadata = {
                "step": step,
                "model_config": asdict(model_config),
                "resolved_config": config.to_dict(),
                "user_config": user_config,
                "total_batch_size": total_batch_size,
                "tokenizer_identity": {
                    "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                    "vocab_size": vocab_size,
                },
                "data_manifest_identity": (
                    dataloader_state.get("dataset_manifest", {}).get("sha256", "unavailable")
                    if dataloader_state else "unavailable"
                ),
                "dataloader_state": dataloader_state,
                "source_revision": source_revision,
                "attention_backend": attention_backend_metadata(
                    model_config.attention_backend, model_config.sequence_len
                ),
                "microbatches_processed": microbatches_processed,
                "successful_updates": successful_updates,
                "skipped_updates": skipped_updates,
            }
            save_checkpoint(
                ckpt_manager, step, model, optimizer, checkpoint_metadata,
                training_state={
                    "update_step": jnp.asarray(step, dtype=jnp.int32),
                    "microbatches_processed": jnp.asarray(
                        microbatches_processed, dtype=jnp.int32
                    ),
                },
            )

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
            microbatches_processed += 1
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
            microbatches_processed += grad_accum_steps

        # Force sync for timing
        loss_val = float(loss)
        update_succeeded = bool(grad_finite)
        successful_updates += int(update_succeeded)
        skipped_updates += int(not update_succeeded)
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
                "train/microbatches_processed": microbatches_processed,
                "train/successful_updates": successful_updates,
                "train/skipped_updates": skipped_updates,
            })

        step += 1

        # GC management (like nanochat)
        if step == 1:
            gc.collect()

    # Cleanup
    print0(f"\nTraining complete! Total time: {total_training_time / 60:.1f}m")
    ckpt_manager.wait_until_finished()
    ckpt_manager.close()
    wandb_run.finish()
    return 0
