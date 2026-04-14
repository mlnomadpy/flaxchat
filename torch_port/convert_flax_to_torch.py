"""
Convert a flaxchat Orbax checkpoint into a PyTorch state_dict.

Usage:
    python convert_flax_to_torch.py \\
        --flax-ckpt models/gelu-d12-chinchilla-seed0/19920 \\
        --out gelu_d12.pt

The output `.pt` is a single file containing both the GPTConfig (as a dict)
and the flattened state_dict, ready to be loaded with
`GELU_GPT.from_pretrained(out_path)`.

This script DOES import JAX/Flax at conversion time (to restore the Orbax
tree), but the produced `.pt` file is consumable by pure PyTorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Add parent so we can "from torch_port.torch_gpt import ..." when run as a script.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from torch_port.torch_gpt import GPTConfig, GELU_GPT, has_ve  # noqa: E402


# ---------------------------------------------------------------------------
# Orbax restore (requires JAX/Flax only at conversion time)
# ---------------------------------------------------------------------------
def _restore_flax_state(ckpt_dir: str, config: "GPTConfig | None" = None) -> Dict[str, Any]:
    """Restore the Flax model state by reconstructing the live GPT model and
    using the repo's own restore helper. This avoids the "sharding=None"
    error that a schema-less ``PyTreeRestore`` hits on newer orbax.

    Returns an `nnx.to_pure_dict`-flavored nested dict of numpy arrays.
    """
    # Patch the MLP to GELU BEFORE importing flaxchat.gpt so the module-level
    # class is in the GELU configuration that matches the checkpoint.
    import jax
    from flaxchat import gpt as _gpt_mod
    _orig_mlp_call = _gpt_mod.MLP.__call__

    def _gelu_call(self, x):
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)
        return x

    _gpt_mod.MLP.__call__ = _gelu_call

    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import numpy as _np_jax
    from flaxchat.gpt import GPT, GPTConfig as FlaxGPTConfig

    if config is None:
        config = load_config(ckpt_dir)

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

    # Build an abstract target with concrete sharding (replicated on single
    # device) so Orbax can deserialize without guessing shardings.
    devices = jax.devices()
    mesh = Mesh(_np_jax.array(devices).reshape(len(devices)), axis_names=('data',))
    rep = NamedSharding(mesh, P())

    param_state = nnx.state(model, nnx.Param)
    pure_abstract = nnx.to_pure_dict(param_state)

    # Ask Orbax to materialize every leaf as a plain numpy array — this
    # sidesteps the "sharding must be concrete" requirement entirely.
    restore_args = jax.tree.map(
        lambda a: ocp.ArrayRestoreArgs(
            restore_type=_np_jax.ndarray,
            dtype=a.dtype,
        ),
        pure_abstract,
    )
    pure_abstract = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype),
        pure_abstract,
    )

    abs_ckpt = os.path.abspath(ckpt_dir)
    if "_CHECKPOINT_METADATA" in os.listdir(abs_ckpt):
        manager_dir = os.path.dirname(abs_ckpt)
        step = int(os.path.basename(abs_ckpt))
    else:
        manager_dir = abs_ckpt
        step = None

    manager = ocp.CheckpointManager(
        directory=manager_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=999),
    )
    if step is None:
        step = manager.latest_step()
        if step is None:
            raise RuntimeError(f"No checkpoints found in {manager_dir}")

    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeRestore(
                item=pure_abstract,
                restore_args=restore_args,
            ),
        ),
    )
    pure = restored["model"]

    def _tonp(tree):
        if isinstance(tree, dict):
            return {k: _tonp(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [_tonp(v) for v in tree]
        return np.asarray(tree)

    return _tonp(pure)


# ---------------------------------------------------------------------------
# Flax -> Torch key/tensor mapping
# ---------------------------------------------------------------------------
def _to_numpy(x: Any) -> np.ndarray:
    """Convert a JAX-or-numpy leaf to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "value"):  # nnx.Param-ish wrapper if any slipped through
        x = x.value
    return np.asarray(x)


def build_torch_state_dict(flax_model: Dict[str, Any], config: GPTConfig) -> Dict[str, torch.Tensor]:
    """Walk the Flax nested-dict model state and produce a torch state_dict.

    Key mapping (Flax -> Torch), derived by reading flaxchat/gpt.py:

      wte.embedding                        -> wte.weight                                  (no transpose)
      blocks[i].attn.c_q.kernel            -> blocks.{i}.attn.c_q.weight                  (TRANSPOSE)
      blocks[i].attn.c_k.kernel            -> blocks.{i}.attn.c_k.weight                  (TRANSPOSE)
      blocks[i].attn.c_v.kernel            -> blocks.{i}.attn.c_v.weight                  (TRANSPOSE)
      blocks[i].attn.c_proj.kernel         -> blocks.{i}.attn.c_proj.weight               (TRANSPOSE)
      blocks[i].attn.ve_gate.kernel        -> blocks.{i}.attn.ve_gate.weight              (TRANSPOSE, only if has_ve(i))
      blocks[i].mlp.c_fc.kernel            -> blocks.{i}.mlp.c_fc.weight                  (TRANSPOSE)
      blocks[i].mlp.c_proj.kernel          -> blocks.{i}.mlp.c_proj.weight                (TRANSPOSE)
      resid_lambdas                        -> resid_lambdas                               (vec, no transpose)
      x0_lambdas                           -> x0_lambdas                                  (vec, no transpose)
      smear_gate.kernel                    -> smear_gate.weight                           (TRANSPOSE)
      smear_lambda                         -> smear_lambda                                (scalar vec)
      backout_lambda                       -> backout_lambda                              (scalar vec)
      value_embeds.{i}.embedding           -> value_embeds.{i}.weight                     (no transpose)
      lm_head.kernel (if not tied)         -> lm_head.weight                              (TRANSPOSE)

    JAX Linear kernel shape is (in, out); PyTorch expects (out, in). Hence
    the transposes marked above.
    """
    sd: Dict[str, torch.Tensor] = {}

    def _pick(d: Dict[str, Any], *path: str) -> Any:
        cur = d
        for p in path:
            cur = cur[p]
        return cur

    # Embedding
    sd["wte.weight"] = torch.from_numpy(_to_numpy(_pick(flax_model, "wte", "embedding")))

    # Blocks — Orbax restore may produce either a list, a dict keyed by int,
    # or a dict keyed by str. Handle all three.
    blocks_container = flax_model["blocks"]
    def _get_block(idx: int):
        if isinstance(blocks_container, list):
            return blocks_container[idx]
        if idx in blocks_container:
            return blocks_container[idx]
        return blocks_container[str(idx)]

    for i in range(config.n_layer):
        block = _get_block(i)

        # Attention Q/K/V/proj — all transposed.
        sd[f"blocks.{i}.attn.c_q.weight"] = torch.from_numpy(
            _to_numpy(_pick(block, "attn", "c_q", "kernel")).T.copy()
        )
        sd[f"blocks.{i}.attn.c_k.weight"] = torch.from_numpy(
            _to_numpy(_pick(block, "attn", "c_k", "kernel")).T.copy()
        )
        sd[f"blocks.{i}.attn.c_v.weight"] = torch.from_numpy(
            _to_numpy(_pick(block, "attn", "c_v", "kernel")).T.copy()
        )
        sd[f"blocks.{i}.attn.c_proj.weight"] = torch.from_numpy(
            _to_numpy(_pick(block, "attn", "c_proj", "kernel")).T.copy()
        )

        if has_ve(i, config.n_layer):
            sd[f"blocks.{i}.attn.ve_gate.weight"] = torch.from_numpy(
                _to_numpy(_pick(block, "attn", "ve_gate", "kernel")).T.copy()
            )

        sd[f"blocks.{i}.mlp.c_fc.weight"] = torch.from_numpy(
            _to_numpy(_pick(block, "mlp", "c_fc", "kernel")).T.copy()
        )
        sd[f"blocks.{i}.mlp.c_proj.weight"] = torch.from_numpy(
            _to_numpy(_pick(block, "mlp", "c_proj", "kernel")).T.copy()
        )

    # Per-layer scalars
    sd["resid_lambdas"] = torch.from_numpy(_to_numpy(flax_model["resid_lambdas"]))
    sd["x0_lambdas"] = torch.from_numpy(_to_numpy(flax_model["x0_lambdas"]))

    # Smear / backout
    sd["smear_gate.weight"] = torch.from_numpy(
        _to_numpy(_pick(flax_model, "smear_gate", "kernel")).T.copy()
    )
    sd["smear_lambda"] = torch.from_numpy(_to_numpy(flax_model["smear_lambda"]))
    sd["backout_lambda"] = torch.from_numpy(_to_numpy(flax_model["backout_lambda"]))

    # Value embeddings — handle dict-int-keyed or dict-str-keyed
    ve_dict = flax_model["value_embeds"]
    for i in range(config.n_layer):
        if not has_ve(i, config.n_layer):
            continue
        if i in ve_dict:
            ve = ve_dict[i]
        elif str(i) in ve_dict:
            ve = ve_dict[str(i)]
        else:
            raise KeyError(f"Missing value_embeds[{i}] in flax checkpoint; keys={list(ve_dict.keys())}")
        sd[f"value_embeds.{i}.weight"] = torch.from_numpy(
            _to_numpy(ve["embedding"])
        )

    # Untied lm_head (not used for d12-chinchilla since tied, but handle it).
    if not config.tie_embeddings:
        sd["lm_head.weight"] = torch.from_numpy(
            _to_numpy(_pick(flax_model, "lm_head", "kernel")).T.copy()
        )

    # Normalize dtypes to fp32 for parity testing; caller can .half() later.
    for k, t in list(sd.items()):
        sd[k] = t.to(torch.float32).contiguous()

    return sd


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(ckpt_dir: str, override_mlp: str = "gelu") -> GPTConfig:
    """Look for a sibling config.json and build a GPTConfig from it.

    The d12-chinchilla checkpoint dir has a `config.json` one level up
    (e.g. models/gelu-d12-chinchilla-seed0/config.json). Try that first,
    falling back to defaults if missing.
    """
    ckpt_path = Path(ckpt_dir).resolve()
    candidates = [
        ckpt_path / "config.json",
        ckpt_path.parent / "config.json",
        ckpt_path.parent.parent / "config.json",
    ]
    cfg_file = next((p for p in candidates if p.exists()), None)

    if cfg_file is None:
        print(f"[warn] no config.json found near {ckpt_dir}; using defaults for GELU d12", flush=True)
        return GPTConfig(mlp=override_mlp)

    with open(cfg_file) as f:
        data = json.load(f)
    model = data.get("model", {})
    return GPTConfig(
        sequence_len=model.get("sequence_len", 1024),
        vocab_size=model.get("vocab_size", 32768),
        n_layer=model.get("n_layer", 12),
        n_head=model.get("n_head", 12),
        n_kv_head=model.get("n_kv_head", 12),
        n_embd=model.get("n_embd", 768),
        window_pattern=model.get("window_pattern", "SSSL"),
        tie_embeddings=model.get("tie_embeddings", True),
        mlp=override_mlp,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flax-ckpt", required=True, help="Path to the Orbax step directory "
                    "(e.g. models/gelu-d12-chinchilla-seed0/19920) or a manager dir.")
    ap.add_argument("--out", required=True, help="Output path for the torch .pt file")
    ap.add_argument("--mlp", default="gelu", choices=["gelu", "relu2"])
    args = ap.parse_args()

    print("Building GPTConfig from config.json")
    config = load_config(args.flax_ckpt, override_mlp=args.mlp)
    print(f"  config = {config}")

    print(f"Loading Flax Orbax checkpoint from: {args.flax_ckpt}")
    flax_model = _restore_flax_state(args.flax_ckpt, config=config)

    print("Converting weights to torch state_dict...")
    sd = build_torch_state_dict(flax_model, config)
    total = sum(t.numel() for t in sd.values())
    print(f"  total converted params = {total:,}")

    payload = {"config": asdict(config), "state_dict": sd}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(payload, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
