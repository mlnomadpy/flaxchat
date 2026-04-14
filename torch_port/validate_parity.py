"""
End-to-end parity test: load the gelu-d12-chinchilla Flax checkpoint into BOTH
the original Flax model and the PyTorch port, run the same token sequence
through both in fp32, and print the max |diff| between their logits.

Acceptance: max |diff| <= 1e-4.

Usage:
    python validate_parity.py \\
        --flax-ckpt models/gelu-d12-chinchilla-seed0/19920 \\
        --torch-ckpt gelu_d12.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from torch_port.torch_gpt import GELU_GPT, GPTConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Flax forward (uses JAX/Flax)
# ---------------------------------------------------------------------------
def run_flax(flax_ckpt: str, ids_np: np.ndarray) -> np.ndarray:
    """Load the flaxchat GPT with the GELU MLP patch, restore weights, run
    forward in fp32, return logits as numpy (B, T, vocab_size)."""
    # Force CPU + fp32 for Flax to match the torch-CPU-fp32 reference exactly.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")  # fp32 is fine
    os.environ.setdefault("FLAXCHAT_COMPUTE_DTYPE", "float32")

    import jax
    import jax.numpy as jnp
    from flax import nnx

    # Monkey-patch compute dtype to fp32 BEFORE importing flaxchat.gpt.
    # flaxchat.common reads COMPUTE_DTYPE at import time.
    import flaxchat.common as fc_common
    fc_common.COMPUTE_DTYPE = jnp.float32

    # Now import flaxchat modules (they read COMPUTE_DTYPE once at import time).
    from flaxchat.gpt import GPT, GPTConfig as FlaxGPTConfig, MLP as FlaxMLP
    import flaxchat.gpt as fc_gpt
    fc_gpt.COMPUTE_DTYPE = jnp.float32  # override in-module too

    # Apply the GELU patch from train_d12_chinchilla.py.
    def _gelu_call(self, x):
        x = self.c_fc(x)
        x = jax.nn.gelu(x)  # default approximate=False? see below
        x = self.c_proj(x)
        return x
    FlaxMLP.__call__ = _gelu_call

    # Build model. We need the same config used in training.
    # Read config.json from the checkpoint's parent.
    import json
    ckpt_path = Path(flax_ckpt).resolve()
    for cand in [ckpt_path / "config.json",
                 ckpt_path.parent / "config.json",
                 ckpt_path.parent.parent / "config.json"]:
        if cand.exists():
            with open(cand) as f:
                cfg_data = json.load(f)["model"]
            break
    else:
        raise RuntimeError("No config.json found near checkpoint")

    config = FlaxGPTConfig(
        sequence_len=cfg_data["sequence_len"],
        vocab_size=cfg_data["vocab_size"],
        n_layer=cfg_data["n_layer"],
        n_head=cfg_data["n_head"],
        n_kv_head=cfg_data["n_kv_head"],
        n_embd=cfg_data["n_embd"],
        window_pattern=cfg_data["window_pattern"],
        tie_embeddings=cfg_data.get("tie_embeddings", True),
    )
    print(f"[flax] building model with {config}")
    model = GPT(config, rngs=nnx.Rngs(0))

    # Restore Orbax weights — bypass flaxchat.checkpoint.load_checkpoint
    # because on CPU it hits an Orbax "sharding=None" error. Build a concrete
    # abstract target that asks for plain numpy arrays, then update the model.
    import orbax.checkpoint as ocp
    step = int(ckpt_path.name)
    manager_dir = str(ckpt_path.parent)
    print(f"[flax] restoring step {step} from {manager_dir}")

    pure_abstract = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    restore_args = jax.tree.map(
        lambda a: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=a.dtype),
        pure_abstract,
    )
    pure_abstract_sd = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), pure_abstract,
    )
    manager = ocp.CheckpointManager(
        directory=manager_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=999),
    )
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeRestore(item=pure_abstract_sd, restore_args=restore_args),
        ),
    )
    loaded = restored["model"]
    # Copy loaded params back into the nnx model in-place via state.
    current_state = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(current_state, loaded)
    nnx.update(model, current_state)
    print(f"[flax] restored")

    ids_jax = jnp.asarray(ids_np)
    logits = model(ids_jax)
    logits = np.asarray(logits)
    return logits


# ---------------------------------------------------------------------------
# Torch forward (pure PyTorch)
# ---------------------------------------------------------------------------
def run_torch(torch_ckpt: str, ids_np: np.ndarray) -> np.ndarray:
    model = GELU_GPT.from_pretrained(torch_ckpt, map_location="cpu")
    model = model.to(torch.float32).eval()
    ids = torch.from_numpy(ids_np).to(torch.long)
    with torch.no_grad():
        logits = model(ids)
    return logits.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flax-ckpt", required=True)
    ap.add_argument("--torch-ckpt", required=True)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--flax-only", action="store_true",
                    help="Only run flax side (useful for debugging)")
    ap.add_argument("--torch-only", action="store_true",
                    help="Only run torch side (useful for debugging)")
    args = ap.parse_args()

    # Fixed token sequence [1..seq_len]
    ids_np = np.arange(1, args.seq_len + 1, dtype=np.int32)[None, :]

    if args.flax_only:
        flax_logits = run_flax(args.flax_ckpt, ids_np)
        print(f"flax logits: shape={flax_logits.shape} "
              f"min={flax_logits.min():.4f} max={flax_logits.max():.4f}")
        return
    if args.torch_only:
        torch_logits = run_torch(args.torch_ckpt, ids_np)
        print(f"torch logits: shape={torch_logits.shape} "
              f"min={torch_logits.min():.4f} max={torch_logits.max():.4f}")
        return

    # Run flax first (imports JAX which may fight torch for BLAS on CPU).
    flax_logits = run_flax(args.flax_ckpt, ids_np)
    torch_logits = run_torch(args.torch_ckpt, ids_np)

    assert flax_logits.shape == torch_logits.shape, (flax_logits.shape, torch_logits.shape)

    diff = np.abs(flax_logits - torch_logits)
    max_diff = diff.max()
    mean_diff = diff.mean()
    # also compare argmax for a sanity vibe check
    flax_top = flax_logits[0, -1].argmax()
    torch_top = torch_logits[0, -1].argmax()

    print(f"\nFlax logits   shape={flax_logits.shape}  range=[{flax_logits.min():.4f}, {flax_logits.max():.4f}]")
    print(f"Torch logits  shape={torch_logits.shape}  range=[{torch_logits.min():.4f}, {torch_logits.max():.4f}]")
    print(f"max |diff|  = {max_diff:.6e}")
    print(f"mean |diff| = {mean_diff:.6e}")
    print(f"flax argmax(last token) = {flax_top} | torch argmax(last token) = {torch_top}")

    if max_diff <= 1e-4:
        print("\nPARITY OK (<= 1e-4)")
    else:
        print(f"\nPARITY FAIL: max |diff| = {max_diff:.2e} > 1e-4")
        # Give a tighter-grained breakdown over the time axis
        per_t = diff.reshape(diff.shape[0], diff.shape[1], -1).max(axis=-1)
        print(f"per-position max diff: {per_t[0].tolist()}")


if __name__ == "__main__":
    main()
