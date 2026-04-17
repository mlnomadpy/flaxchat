"""
Convert a flaxchat YatNMN-Softplus Orbax checkpoint into a PyTorch state_dict.

YatNMN param mapping (Flax → Torch):
    blocks.{i}.mlp.c_fc.kernel         -> blocks.{i}.mlp.c_fc.kernel         (NO transpose; nmn stores (in, out))
    blocks.{i}.mlp.c_fc.bias           -> blocks.{i}.mlp.c_fc.bias           (shape (1,) or (ff,), no transpose)
    blocks.{i}.mlp.c_fc.epsilon_param  -> blocks.{i}.mlp.c_fc.epsilon_param  (shape (1,))
    blocks.{i}.mlp.c_fc.alpha          -> blocks.{i}.mlp.c_fc.alpha          (shape (1,)); omitted if constant_alpha
    blocks.{i}.mlp.c_proj.kernel       -> blocks.{i}.mlp.c_proj.weight       (TRANSPOSE — standard Linear)

Everything else matches convert_flax_to_torch.py (wte/attn/ve/smear/backout/...).

Usage:
    pixi run python torch_port/convert_yatnmn_flax_to_torch.py \\
        --flax-ckpt models/yatnmn-softplus-d12-seed0/19922 \\
        --out yatnmn_softplus_d12.pt
"""
from __future__ import annotations

import argparse, json, os, sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from torch_port.yatnmn_gpt import YatGPTConfig, Yat_GPT  # noqa: E402
from torch_port.torch_gpt import has_ve  # noqa: E402


def _restore_flax_state(ckpt_dir: str, config: YatGPTConfig) -> Dict[str, Any]:
    """Rebuild the live Flax model with matching YatNMN config, restore Orbax."""
    import jax, jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import nnx
    from nmn.nnx.layers import YatNMN

    # Patch Block BEFORE importing GPT so the model has the right MLP.
    import flaxchat.gpt as _gpt_mod

    _orig_block_init = _gpt_mod.Block.__init__
    def _patched(self, cfg, layer_idx, *, rngs, use_remat=False):
        _orig_block_init(self, cfg, layer_idx, rngs=rngs, use_remat=use_remat)
        class YatFFN(nnx.Module):
            def __init__(self, n, ff, *, rngs):
                kw = dict(
                    use_bias=True,
                    softplus_bias=config.softplus_bias,
                    scalar_bias=config.scalar_bias,
                    learnable_epsilon=config.learnable_epsilon,
                    epsilon=config.epsilon_init,
                )
                if config.constant_alpha:
                    kw["use_alpha"] = True
                    kw["constant_alpha"] = True
                    kw["alpha_init"] = nnx.initializers.constant(1.0)
                self.c_fc = YatNMN(n, ff, rngs=rngs, **kw)
                self.c_proj = nnx.Linear(ff, n, use_bias=False, rngs=rngs)
            def __call__(self, x): return self.c_proj(self.c_fc(x))
        self.mlp = YatFFN(cfg.n_embd, 4 * cfg.n_embd, rngs=rngs)
    _gpt_mod.Block.__init__ = _patched

    from flaxchat.gpt import GPT, GPTConfig as FlaxGPTConfig

    flax_cfg = FlaxGPTConfig(
        sequence_len=config.sequence_len,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        n_embd=config.n_embd,
        window_pattern=config.window_pattern,
        tie_embeddings=config.tie_embeddings,
    )
    model = GPT(flax_cfg, rngs=nnx.Rngs(0))

    pure_abstract = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    restore_args = jax.tree.map(
        lambda a: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=a.dtype),
        pure_abstract,
    )
    pure_abstract_sd = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), pure_abstract,
    )

    abs_ckpt = os.path.abspath(ckpt_dir) if not ckpt_dir.startswith("gs://") else ckpt_dir
    manager_dir = os.path.dirname(abs_ckpt)
    step = int(os.path.basename(abs_ckpt))
    mgr = ocp.CheckpointManager(directory=manager_dir,
                                 options=ocp.CheckpointManagerOptions(max_to_keep=999))
    restored = mgr.restore(step, args=ocp.args.Composite(
        model=ocp.args.PyTreeRestore(item=pure_abstract_sd, restore_args=restore_args),
    ))
    _gpt_mod.Block.__init__ = _orig_block_init
    # Convert all jnp arrays to numpy for downstream torch use
    def _tonp(t):
        if isinstance(t, dict): return {k: _tonp(v) for k, v in t.items()}
        if isinstance(t, list): return [_tonp(v) for v in t]
        return np.asarray(t)
    return _tonp(restored["model"])


def _to_numpy(x):
    if isinstance(x, np.ndarray): return x
    if hasattr(x, "value"): x = x.value
    return np.asarray(x)


def build_torch_state_dict(flax_model, config: YatGPTConfig):
    sd = {}
    # wte
    sd["wte.weight"] = torch.from_numpy(_to_numpy(flax_model["wte"]["embedding"]))

    blocks_c = flax_model["blocks"]
    def _get_block(i: int):
        if isinstance(blocks_c, list): return blocks_c[i]
        return blocks_c[i] if i in blocks_c else blocks_c[str(i)]

    for i in range(config.n_layer):
        blk = _get_block(i)
        # Attention Q/K/V/proj — transpose
        for name in ("c_q", "c_k", "c_v", "c_proj"):
            k = _to_numpy(blk["attn"][name]["kernel"])
            sd[f"blocks.{i}.attn.{name}.weight"] = torch.from_numpy(k.T.copy())

        if has_ve(i, config.n_layer):
            k = _to_numpy(blk["attn"]["ve_gate"]["kernel"])
            sd[f"blocks.{i}.attn.ve_gate.weight"] = torch.from_numpy(k.T.copy())

        # MLP YatNMN — c_fc keeps native (in, out); c_proj transposes
        c_fc = blk["mlp"]["c_fc"]
        sd[f"blocks.{i}.mlp.c_fc.kernel"] = torch.from_numpy(_to_numpy(c_fc["kernel"]).copy())
        if "bias" in c_fc:
            sd[f"blocks.{i}.mlp.c_fc.bias"] = torch.from_numpy(_to_numpy(c_fc["bias"]).copy())
        if "epsilon_param" in c_fc:
            sd[f"blocks.{i}.mlp.c_fc.epsilon_param"] = torch.from_numpy(_to_numpy(c_fc["epsilon_param"]).copy())
        if "alpha" in c_fc and not config.constant_alpha:
            sd[f"blocks.{i}.mlp.c_fc.alpha"] = torch.from_numpy(_to_numpy(c_fc["alpha"]).copy())

        cp = _to_numpy(blk["mlp"]["c_proj"]["kernel"])
        sd[f"blocks.{i}.mlp.c_proj.weight"] = torch.from_numpy(cp.T.copy())

    # Per-layer scalars, smear, backout
    sd["resid_lambdas"] = torch.from_numpy(_to_numpy(flax_model["resid_lambdas"]))
    sd["x0_lambdas"] = torch.from_numpy(_to_numpy(flax_model["x0_lambdas"]))
    sd["smear_gate.weight"] = torch.from_numpy(_to_numpy(flax_model["smear_gate"]["kernel"]).T.copy())
    sd["smear_lambda"] = torch.from_numpy(_to_numpy(flax_model["smear_lambda"]))
    sd["backout_lambda"] = torch.from_numpy(_to_numpy(flax_model["backout_lambda"]))

    # Value embeddings
    ve = flax_model["value_embeds"]
    for i in range(config.n_layer):
        if not has_ve(i, config.n_layer): continue
        entry = ve[i] if i in ve else ve[str(i)]
        sd[f"value_embeds.{i}.weight"] = torch.from_numpy(_to_numpy(entry["embedding"]).copy())

    # lm_head only if untied
    if not config.tie_embeddings:
        k = _to_numpy(flax_model["lm_head"]["kernel"])
        sd["lm_head.weight"] = torch.from_numpy(k.T.copy())

    # Cast to fp32 for parity testing
    for k, t in list(sd.items()):
        sd[k] = t.to(torch.float32).contiguous()
    return sd


def load_config(ckpt_dir: str) -> YatGPTConfig:
    # Find sibling config.json
    p = Path(ckpt_dir).resolve() if not ckpt_dir.startswith("gs://") else Path(ckpt_dir)
    candidates = [p / "config.json", p.parent / "config.json", p.parent.parent / "config.json"]
    cfg_file = next((c for c in candidates if str(c).startswith("gs://") or c.exists()), None)

    if cfg_file is None:
        print(f"[warn] no config.json found near {ckpt_dir}; using d=12 defaults")
        return YatGPTConfig()

    with open(cfg_file) as f:
        data = json.load(f)
    m = data.get("model", {})
    feat = data.get("features", {}) or {}
    # Detect scalar_bias/constant_alpha from description/mlp_details if present
    mlp = str(data.get("architecture", "")).lower() + " " + str(feat.get("mlp", "")).lower()
    sb = "scalar_bias" in mlp or "scalar bias" in mlp
    # Can't detect constant_alpha from legacy config; use default False
    return YatGPTConfig(
        sequence_len=m.get("sequence_len", 1024),
        vocab_size=m.get("vocab_size", 32768),
        n_layer=m.get("n_layer", 12),
        n_head=m.get("n_head", 12),
        n_kv_head=m.get("n_kv_head", 12),
        n_embd=m.get("n_embd", 768),
        window_pattern=m.get("window_pattern", "SSSL"),
        tie_embeddings=m.get("tie_embeddings", True),
        scalar_bias=sb,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flax-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scalar-bias", action="store_true")
    ap.add_argument("--constant-alpha", action="store_true")
    args = ap.parse_args()

    print(f"Loading config from {args.flax_ckpt}")
    config = load_config(args.flax_ckpt)
    # CLI overrides (useful when config.json can't detect variant)
    if args.scalar_bias:
        config.scalar_bias = True
    if args.constant_alpha:
        config.constant_alpha = True
    print(f"  config = {config}")

    print(f"Restoring Flax Orbax from {args.flax_ckpt}")
    flax_model = _restore_flax_state(args.flax_ckpt, config=config)

    print("Converting weights to torch state_dict...")
    sd = build_torch_state_dict(flax_model, config)
    print(f"  {len(sd)} tensors, {sum(t.numel() for t in sd.values()):,} params")

    payload = {"config": asdict(config), "state_dict": sd}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(payload, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
