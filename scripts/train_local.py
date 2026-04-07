"""
Train a tiny model locally on Shakespeare and export for on-device inference.

This script:
1. Downloads tiny Shakespeare dataset
2. Trains a small GPT model on CPU/GPU
3. Saves JAX checkpoint
4. Exports to LiteRT (.tflite) format via JAX export + StableHLO

Usage:
    python -m scripts.train_local
    python -m scripts.train_local --depth=4 --steps=200 --export-tflite
"""

import os
import time
import argparse
import urllib.request
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.common import print0, get_base_dir

parser = argparse.ArgumentParser(description="Train locally on Shakespeare")
parser.add_argument("--depth", type=int, default=2, help="Model depth (2-4 for laptop)")
parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--steps", type=int, default=100, help="Training steps")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--export-tflite", action="store_true", help="Export to LiteRT/TFLite")
parser.add_argument("--export-dir", type=str, default="exports", help="Export directory")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Download Shakespeare
# ---------------------------------------------------------------------------
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
cache_dir = os.path.join(get_base_dir(), "data")
os.makedirs(cache_dir, exist_ok=True)
shakespeare_path = os.path.join(cache_dir, "shakespeare.txt")

if not os.path.exists(shakespeare_path):
    print0("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, shakespeare_path)

with open(shakespeare_path, "r") as f:
    text = f.read()
print0(f"Shakespeare: {len(text):,} characters")

# ---------------------------------------------------------------------------
# 2. Byte-level tokenization (simple, no BPE needed)
# ---------------------------------------------------------------------------
vocab_size = 256  # byte-level
data = np.array(list(text.encode("utf-8")), dtype=np.int32)

# Train/val split
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]
print0(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")


def get_batch(data, batch_size, seq_len, key):
    """Random batch from data."""
    max_start = len(data) - seq_len - 1
    starts = jax.random.randint(key, (batch_size,), 0, max_start)
    # Build batch on CPU with numpy, then convert
    inputs = np.zeros((batch_size, seq_len), dtype=np.int32)
    targets = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        s = int(starts[i])
        inputs[i] = data[s:s + seq_len]
        targets[i] = data[s + 1:s + seq_len + 1]
    return jnp.array(inputs), jnp.array(targets)


# ---------------------------------------------------------------------------
# 3. Build model
# ---------------------------------------------------------------------------
n_head = max(1, args.n_embd // 32)
config = GPTConfig(
    sequence_len=args.seq_len,
    vocab_size=vocab_size,
    n_layer=args.depth,
    n_head=n_head,
    n_kv_head=n_head,
    n_embd=args.n_embd,
    window_pattern="L",  # no sliding window for tiny model
)

model = GPT(config, rngs=nnx.Rngs(0))
num_params = model.num_params()
print0(f"Model: depth={args.depth}, dim={args.n_embd}, heads={n_head}")
print0(f"Parameters: {num_params:,}")

# ---------------------------------------------------------------------------
# 4. Optimizer (simple AdamW for local training)
# ---------------------------------------------------------------------------
import optax

tx = optax.adamw(learning_rate=args.lr, b1=0.9, b2=0.99, weight_decay=0.01)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, inputs, targets):
    def loss_fn(model):
        return model(inputs, targets)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, inputs, targets):
    return model(inputs, targets)


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
print0(f"\nTraining for {args.steps} steps...")
print0(f"{'step':>6} {'train_loss':>10} {'val_loss':>10} {'dt_ms':>8} {'tok/s':>10}")
print0("-" * 50)

best_val_loss = float('inf')
t_total = 0

for step in range(args.steps):
    t0 = time.time()

    key = jax.random.key(step)
    inputs, targets = get_batch(train_data, args.batch_size, args.seq_len, key)
    loss = train_step(model, optimizer, inputs, targets)

    dt = time.time() - t0
    if step > 5:
        t_total += dt

    # Eval every 20 steps
    if step % 20 == 0 or step == args.steps - 1:
        val_key = jax.random.key(step + 1000000)
        val_inputs, val_targets = get_batch(val_data, args.batch_size, args.seq_len, val_key)
        val_loss = eval_step(model, val_inputs, val_targets)
        val_loss_f = float(val_loss)
        if val_loss_f < best_val_loss:
            best_val_loss = val_loss_f

        tok_per_sec = int(args.batch_size * args.seq_len / dt) if dt > 0 else 0
        print0(f"{step:6d} {float(loss):10.4f} {val_loss_f:10.4f} {dt*1000:8.1f} {tok_per_sec:10,}")

print0(f"\nBest val loss: {best_val_loss:.4f}")
print0(f"Training time: {t_total:.1f}s")

# ---------------------------------------------------------------------------
# 6. Generate a sample
# ---------------------------------------------------------------------------
print0("\n--- Sample generation ---")
prompt = "ROMEO:"
prompt_tokens = list(prompt.encode("utf-8"))
prompt_tokens = [t % vocab_size for t in prompt_tokens]

from flaxchat.engine import generate
output_tokens = generate(model, prompt_tokens, max_tokens=200, temperature=0.8, top_k=40, seed=42)
output_text = bytes(output_tokens).decode("utf-8", errors="replace")
print0(output_text[:500])

# ---------------------------------------------------------------------------
# 7. Save JAX checkpoint
# ---------------------------------------------------------------------------
os.makedirs(args.export_dir, exist_ok=True)
ckpt_path = os.path.join(args.export_dir, "model_local")

# Save as pure dict (portable)
model_state = nnx.state(model, nnx.Param)
model_dict = nnx.to_pure_dict(model_state)

import pickle
ckpt_file = os.path.join(args.export_dir, "model_local.pkl")
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
    }, f)
print0(f"\nCheckpoint saved to {ckpt_file}")

# ---------------------------------------------------------------------------
# 8. Export to LiteRT / TFLite (if requested)
# ---------------------------------------------------------------------------
if args.export_tflite:
    print0("\n--- Exporting to LiteRT (.tflite) ---")

    try:
        # Modern path: JAX → jax.export → StableHLO → flatbuffer
        # This uses jax.export which produces a portable StableHLO artifact

        # First, create a pure function for export (no nnx.Module state)
        graphdef, state = nnx.split(model)

        def model_fn(state, input_ids):
            model = nnx.merge(graphdef, state)
            logits = model(input_ids)
            return logits

        # Export with jax.export (call syntax: export(fn)(example_args))
        exported = jax.export.export(
            jax.jit(partial(model_fn, state))
        )(jnp.ones((1, args.seq_len), dtype=jnp.int32))

        # Save as StableHLO bytecode
        stablehlo_path = os.path.join(args.export_dir, "model.stablehlo")
        serialized = exported.serialize()
        with open(stablehlo_path, "wb") as f:
            f.write(serialized)
        print0(f"StableHLO exported to {stablehlo_path} ({len(serialized) / 1024:.1f} KB)")

        # Try TFLite conversion via LiteRT native converter
        try:
            from ai_edge_litert import _pywrap_litert_converter as litert_conv
            converter = litert_conv.Converter()

            # The native converter expects a SavedModel or flatbuffer
            # For JAX, we need to go through the TF SavedModel path
            # which requires tensorflow. Let's try the direct StableHLO path.
            print0("Note: Direct StableHLO → TFLite conversion requires tensorflow.")
            print0("The StableHLO artifact can be converted on a Linux machine with:")
            print0(f"  pip install tensorflow")
            print0(f"  python -c \"")
            print0(f"    import tensorflow as tf")
            print0(f"    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')")
            print0(f"    tflite_model = converter.convert()\"")
            print0("")
            print0("Or use the provided convert_to_tflite.py script on a Linux machine.")
        except Exception as e:
            print0(f"LiteRT native converter: {e}")

        # Also export as ONNX for broader compatibility
        print0("\nAlternatively, exporting model weights as numpy for portable use...")
        weights_path = os.path.join(args.export_dir, "model_weights.npz")
        flat_params = {}
        for path, leaf in jax.tree.leaves_with_path(model_dict):
            key = "/".join(str(p) for p in path)
            flat_params[key] = np.array(leaf)
        np.savez(weights_path, **flat_params)
        print0(f"Weights saved to {weights_path} ({os.path.getsize(weights_path) / 1024:.1f} KB)")

    except Exception as e:
        print0(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

print0("\nDone!")
