"""Parity test: load YatNMN-Softplus Orbax weights into BOTH Flax and PyTorch,
forward the same token sequence, report max |diff|. Target ≤ 1e-4 on CPU/fp32."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from torch_port.yatnmn_gpt import Yat_GPT  # noqa: E402


def run_flax(flax_ckpt: str, ids_np: np.ndarray, scalar_bias: bool, constant_alpha: bool):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("FLAXCHAT_COMPUTE_DTYPE", "float32")

    import jax, jax.numpy as jnp
    from flax import nnx
    import orbax.checkpoint as ocp
    from nmn.nnx.layers import YatNMN

    import flaxchat.common as fc_common
    fc_common.COMPUTE_DTYPE = jnp.float32
    import flaxchat.gpt as fc_gpt
    fc_gpt.COMPUTE_DTYPE = jnp.float32

    _orig_init = fc_gpt.Block.__init__
    def _patched(self, cfg, layer_idx, *, rngs, use_remat=False):
        _orig_init(self, cfg, layer_idx, rngs=rngs, use_remat=use_remat)
        class YatFFN(nnx.Module):
            def __init__(self, n, ff, *, rngs):
                kw = dict(use_bias=True, softplus_bias=True, scalar_bias=scalar_bias,
                          learnable_epsilon=True, epsilon=1e-3)
                if constant_alpha:
                    kw["use_alpha"] = True
                    kw["constant_alpha"] = True
                    kw["alpha_init"] = nnx.initializers.constant(1.0)
                self.c_fc = YatNMN(n, ff, rngs=rngs, **kw)
                self.c_proj = nnx.Linear(ff, n, use_bias=False, rngs=rngs)
            def __call__(self, x): return self.c_proj(self.c_fc(x))
        self.mlp = YatFFN(cfg.n_embd, 4 * cfg.n_embd, rngs=rngs)
    fc_gpt.Block.__init__ = _patched

    from flaxchat.gpt import GPT, GPTConfig as FlaxGPTConfig

    ckpt_path = Path(flax_ckpt).resolve()
    for c in [ckpt_path / "config.json", ckpt_path.parent / "config.json", ckpt_path.parent.parent / "config.json"]:
        if c.exists():
            with open(c) as f:
                cfg_data = json.load(f)["model"]
            break
    else:
        raise RuntimeError("No config.json found")

    config = FlaxGPTConfig(
        sequence_len=cfg_data["sequence_len"], vocab_size=cfg_data["vocab_size"],
        n_layer=cfg_data["n_layer"], n_head=cfg_data["n_head"],
        n_kv_head=cfg_data["n_kv_head"], n_embd=cfg_data["n_embd"],
        window_pattern=cfg_data["window_pattern"],
        tie_embeddings=cfg_data.get("tie_embeddings", True),
    )
    print(f"[flax] building {config}")
    model = GPT(config, rngs=nnx.Rngs(0))

    step = int(ckpt_path.name)
    manager_dir = str(ckpt_path.parent)
    pure_abstract = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    restore_args = jax.tree.map(
        lambda a: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=a.dtype),
        pure_abstract,
    )
    pure_abstract_sd = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), pure_abstract,
    )
    mgr = ocp.CheckpointManager(directory=manager_dir,
                                 options=ocp.CheckpointManagerOptions(max_to_keep=999))
    restored = mgr.restore(step, args=ocp.args.Composite(
        model=ocp.args.PyTreeRestore(item=pure_abstract_sd, restore_args=restore_args)
    ))
    current_state = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(current_state, restored["model"])
    nnx.update(model, current_state)

    ids_jax = jnp.asarray(ids_np)
    logits = np.asarray(model(ids_jax))
    return logits


def run_torch(torch_ckpt: str, ids_np: np.ndarray):
    model = Yat_GPT.from_pretrained(torch_ckpt, map_location="cpu").to(torch.float32).eval()
    ids = torch.from_numpy(ids_np).to(torch.long)
    with torch.no_grad():
        logits = model(ids)
    return logits.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flax-ckpt", required=True)
    ap.add_argument("--torch-ckpt", required=True)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--scalar-bias", action="store_true")
    ap.add_argument("--constant-alpha", action="store_true")
    args = ap.parse_args()

    ids_np = np.arange(1, args.seq_len + 1, dtype=np.int32)[None, :]

    print(">>> Flax forward")
    flax_logits = run_flax(args.flax_ckpt, ids_np, args.scalar_bias, args.constant_alpha)
    print(">>> Torch forward")
    torch_logits = run_torch(args.torch_ckpt, ids_np)

    diff = np.abs(flax_logits - torch_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    flax_top = int(flax_logits[0, -1].argmax())
    torch_top = int(torch_logits[0, -1].argmax())

    print(f"\nFlax  range=[{flax_logits.min():.4f}, {flax_logits.max():.4f}]")
    print(f"Torch range=[{torch_logits.min():.4f}, {torch_logits.max():.4f}]")
    print(f"max |diff|  = {max_diff:.4e}")
    print(f"mean |diff| = {mean_diff:.4e}")
    print(f"argmax(last) flax={flax_top} torch={torch_top} {'MATCH' if flax_top == torch_top else 'MISMATCH'}")
    if max_diff <= 1e-4:
        print("\nPARITY OK (<= 1e-4)")
    else:
        print(f"\nPARITY FAIL: {max_diff:.2e} > 1e-4")
        per_t = diff.reshape(diff.shape[0], diff.shape[1], -1).max(axis=-1)
        print(f"per-position max diff: {per_t[0].tolist()[:16]}")


if __name__ == "__main__":
    main()
