"""
Train a GPT-2-sized causal language model on FineWeb-Edu using HF streaming.

Self-contained — no pre-existing tokenizer or data files are needed.  The
default tokenizer is the open GPT-2 byte-level BPE (50,257 tokens), which is
small enough for the embedding table to remain practical while preserving a
standard GPT vocabulary contract.

Usage:
    # Single host (v6e-8)
    python scripts/train_gpt2.py --batch-per-device=512

    # Multi-host (v6e-32, 4 workers)
    python scripts/train_gpt2.py --batch-per-device=512 --tokens=10B

    # Quick test
    python scripts/train_gpt2.py --batch-per-device=4 --tokens=1M --eval-every=10
"""

import os
import sys
import time
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flaxchat.gpt import GPT, GPTConfig
from flaxchat.common import compute_init, print0


def parse_tokens(s):
    """Parse '10B', '1M', '500K' etc."""
    s = s.upper().strip()
    if s.endswith('B'):
        return int(float(s[:-1]) * 1e9)
    if s.endswith('M'):
        return int(float(s[:-1]) * 1e6)
    if s.endswith('K'):
        return int(float(s[:-1]) * 1e3)
    return int(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a GPT-2-sized model on FineWeb-Edu")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--batch-per-device", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--tokens", type=str, default="1B")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-subset", type=str, default="sample-10BT")
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2",
        help="Hugging Face tokenizer ID; must have at most --max-vocab-size tokens",
    )
    parser.add_argument(
        "--max-vocab-size", type=int, default=60_000,
        help="hard upper bound for the tokenizer vocabulary",
    )
    parser.add_argument("--run-name", type=str, default="gpt2-base")
    parser.add_argument(
        "--ckpt-dir", type=str, default="artifacts/fineweb-gpt/checkpoints",
        help="local or mounted directory for resumable Orbax checkpoints",
    )
    parser.add_argument(
        "--artifact-dir", type=str, default="artifacts/fineweb-gpt",
        help="directory for the immutable resolved training summary",
    )
    embedding_tying = parser.add_mutually_exclusive_group()
    embedding_tying.add_argument(
        "--tie-embeddings", action="store_true", dest="tie_embeddings", default=True,
        help="share token embedding and output projection weights (GPT-2 default)",
    )
    embedding_tying.add_argument(
        "--untie-embeddings", action="store_false", dest="tie_embeddings",
        help="use separate token embedding and output projection weights",
    )
    args = parser.parse_args(argv)

    # ── Distributed init ──
    mesh = compute_init()
    rank = jax.process_index()
    n_devices = jax.device_count()
    total_tokens = parse_tokens(args.tokens)

    if args.max_vocab_size <= 0:
        parser.error("--max-vocab-size must be positive")

    # ── Tokenizer ──
    print0("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    if vocab_size > args.max_vocab_size:
        parser.error(
            f"tokenizer {args.tokenizer!r} has {vocab_size:,} tokens; "
            f"the configured maximum is {args.max_vocab_size:,}"
        )
    print0(f"Tokenizer: {args.tokenizer} ({vocab_size:,} tokens)")

    # ── Model ──
    aspect_ratio = 64
    # Match the canonical GPT-2 small head layout: 12 layers, 768 hidden
    # dimensions, and 12 full-attention heads of width 64 by default.
    head_dim = 64
    base_dim = args.depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_heads = model_dim // head_dim
    n_kv_heads = n_heads

    config = GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=n_heads,
        n_kv_head=n_kv_heads,
        n_embd=model_dim,
        window_pattern="L",
        tie_embeddings=args.tie_embeddings,
        standard_gpt=True,
    )

    print0(f"Model: {config.n_layer}L/{config.n_embd}d/{config.n_head}h (GQA: {config.n_kv_head}kv)")
    model = GPT(config, rngs=nnx.Rngs(42))
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print0(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Replicate model across devices
    from flaxchat.common import replicate_on_mesh
    state = nnx.state(model)
    state = replicate_on_mesh(state, mesh)
    nnx.update(model, state)

    # ── Training config ──
    global_batch = args.batch_per_device * n_devices
    tok_per_step = global_batch * args.seq_len
    num_steps = total_tokens // tok_per_step
    actual_tokens = num_steps * tok_per_step
    if num_steps <= 0:
        parser.error(
            "--tokens must cover at least one global training step "
            f"({tok_per_step:,} tokens)"
        )

    print0(f"Batch: {args.batch_per_device}/dev × {n_devices} = {global_batch} global")
    print0(f"Tokens/step: {tok_per_step:,} ({tok_per_step/1e6:.1f}M)")
    print0(f"Steps: {num_steps:,} ({actual_tokens/1e9:.1f}B tokens)")

    # ── MFU computation (cached — only computed once) ──
    from flaxchat.common import get_peak_flops
    flops_per_token = model.estimate_flops()
    peak_flops = get_peak_flops() * n_devices  # total peak across all devices
    print0(f"FLOPs/token: {flops_per_token:.2e}")
    print0(f"Peak FLOPS: {peak_flops:.2e} ({peak_flops/1e12:.0f} TFLOPS)")

    # ── Optimizer (AdamW with cosine schedule) ──
    warmup_steps = args.warmup_steps
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=args.lr * 0.05,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    print0(f"Optimizer: AdamW lr={args.lr} warmup={warmup_steps} cosine→{args.lr*0.05}")

    # ── Dataset (HF streaming) ──
    print0(f"Loading dataset: {args.dataset}/{args.dataset_subset} (streaming)...")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_subset, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)
    ds_iter = iter(ds)

    def get_batch():
        """Get a batch of tokenized text from the streaming dataset."""
        texts = []
        for _ in range(global_batch):
            try:
                item = next(ds_iter)
                texts.append(item["text"][:args.seq_len * 6])  # rough char limit
            except StopIteration:
                # Restart stream
                texts.append("The ")

        encoded = tokenizer(
            texts, truncation=True, max_length=args.seq_len + 1,
            padding="max_length", return_tensors="np",
        )
        ids = encoded["input_ids"]
        inputs = ids[:, :-1]
        targets = ids[:, 1:]
        return inputs, targets

    # ── Train step ──
    @partial(nnx.jit, donate_argnames=("optimizer",))
    def train_step(model, optimizer, inputs, targets):
        inputs = jax.lax.with_sharding_constraint(inputs, NamedSharding(mesh, P('data')))
        targets = jax.lax.with_sharding_constraint(targets, NamedSharding(mesh, P('data')))

        def loss_fn(model):
            logits = model(inputs).astype(jnp.float32)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            mask = (targets != tokenizer.pad_token_id).astype(jnp.float32)
            return (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # ── W&B ──
    if rank == 0:
        try:
            import wandb
            wandb.init(project="flaxchat", name=args.run_name, config={
                "depth": args.depth, "n_embd": config.n_embd, "n_params": n_params,
                "batch_per_device": args.batch_per_device, "global_batch": global_batch,
                "total_tokens": actual_tokens, "num_steps": num_steps,
                "lr": args.lr, "dataset": args.dataset,
                "dataset_subset": args.dataset_subset, "tokenizer": args.tokenizer,
                "vocab_size": vocab_size,
            })
            use_wandb = True
        except Exception:
            use_wandb = False
    else:
        use_wandb = False

    # ── Training loop ──
    print0(f"\n{'='*60}")
    print0(f"Starting training: {num_steps:,} steps, {actual_tokens/1e9:.1f}B tokens")
    print0(f"{'='*60}\n")

    t0 = time.time()
    smooth_loss = None
    batch_sharding = NamedSharding(mesh, P('data'))
    for step in range(num_steps):
        # HF streaming uses PyArrow C extensions. Keep it and device placement
        # on the main process: Kaggle's TPU runtime can abort when a daemon
        # prefetch thread finalizes while PyArrow is reading a remote parquet.
        inputs, targets = get_batch()
        inputs = jax.device_put(jnp.asarray(inputs), batch_sharding)
        targets = jax.device_put(jnp.asarray(targets), batch_sharding)

        loss = train_step(model, optimizer, inputs, targets)
        loss_val = float(loss)

        smooth_loss = loss_val if smooth_loss is None else 0.95 * smooth_loss + 0.05 * loss_val

        if step % 20 == 0:
            elapsed = time.time() - t0
            tok_s = (step + 1) * tok_per_step / elapsed if elapsed > 0 else 0
            eta = (num_steps - step) * elapsed / max(step + 1, 1)
            lr_now = float(schedule(step))
            mfu = (flops_per_token * tok_s) / peak_flops if peak_flops > 0 else 0
            print0(
                f"step {step:5d}/{num_steps} | loss {loss_val:.4f} (smooth {smooth_loss:.4f}) | "
                f"lr {lr_now:.2e} | {tok_s/1e6:.2f}M tok/s | mfu {mfu:.1%} | "
                f"eta {eta/3600:.1f}h"
            )
            if use_wandb:
                wandb.log({
                    "loss": loss_val, "smooth_loss": smooth_loss,
                    "lr": lr_now, "tok_per_sec": tok_s, "mfu": mfu, "step": step,
                    "tokens_seen": (step + 1) * tok_per_step,
                })

        # Checkpoint
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            if rank == 0:
                from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
                os.makedirs(args.ckpt_dir, exist_ok=True)
                mgr = create_checkpoint_manager(args.ckpt_dir, async_checkpointing=False)
                save_checkpoint(mgr, step, model, optimizer, {
                    "step": step, "loss": loss_val, "tokens": (step + 1) * tok_per_step,
                    "resolved_config": vars(args),
                    "tokenizer_identity": {"id": args.tokenizer, "vocab_size": vocab_size},
                    "data_manifest_identity": {
                        "dataset": args.dataset, "subset": args.dataset_subset,
                    },
                })
                mgr.wait_until_finished()
                print0(f"  Saved checkpoint at step {step}")

    # ── Final ──
    elapsed = time.time() - t0
    final_tok_s = num_steps * tok_per_step / elapsed
    print0(f"\n{'='*60}")
    print0("Training complete!")
    print0(f"  Steps: {num_steps:,}")
    print0(f"  Tokens: {actual_tokens/1e9:.1f}B")
    print0(f"  Time: {elapsed/3600:.2f}h")
    print0(f"  Throughput: {final_tok_s/1e6:.2f}M tok/s")
    print0(f"  Final loss: {loss_val:.4f}")
    print0(f"{'='*60}")

    if rank == 0:
        # Save final checkpoint
        from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
        os.makedirs(args.ckpt_dir, exist_ok=True)
        mgr = create_checkpoint_manager(args.ckpt_dir, async_checkpointing=False)
        final_metadata = {
            "step": num_steps, "loss": loss_val, "tokens": actual_tokens,
            "elapsed_h": elapsed / 3600, "tok_per_sec": final_tok_s,
            "resolved_config": vars(args),
            "tokenizer_identity": {"id": args.tokenizer, "vocab_size": vocab_size},
            "data_manifest_identity": {
                "dataset": args.dataset, "subset": args.dataset_subset,
            },
        }
        save_checkpoint(mgr, num_steps, model, optimizer, final_metadata)
        mgr.wait_until_finished()
        artifact_dir = os.path.abspath(args.artifact_dir)
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
            import json
            json.dump(final_metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print0("Final checkpoint saved!")

    if use_wandb:
        wandb.finish()

    print0("=== TRAINING COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
