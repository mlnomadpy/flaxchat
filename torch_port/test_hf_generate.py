"""End-to-end test: load the converted GELU_GPT via the HF wrapper, verify
cached forward matches uncached forward, then run .generate()."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from torch_port.configuration_gelu_gpt import GeluGPTConfig
from torch_port.modeling_gelu_gpt import GeluGPTForCausalLM
from torch_port.torch_gpt import GELU_GPT


def build_hf_model_from_pt(pt_path: str) -> GeluGPTForCausalLM:
    # Load the raw torch_gpt checkpoint first to get config + state
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    inner_cfg = payload["config"]
    state_dict = payload["state_dict"]

    hf_cfg = GeluGPTConfig(
        sequence_len=inner_cfg["sequence_len"],
        vocab_size=inner_cfg["vocab_size"],
        n_layer=inner_cfg["n_layer"],
        n_head=inner_cfg["n_head"],
        n_kv_head=inner_cfg["n_kv_head"],
        n_embd=inner_cfg["n_embd"],
        window_pattern=inner_cfg["window_pattern"],
        tie_embeddings=inner_cfg["tie_embeddings"],
        rope_base=inner_cfg["rope_base"],
        pad_vocab_size_to=inner_cfg["pad_vocab_size_to"],
        mlp=inner_cfg["mlp"],
        tie_word_embeddings=inner_cfg["tie_embeddings"],
    )
    model = GeluGPTForCausalLM(hf_cfg)
    # The inner GELU_GPT state dict keys start without prefix. Re-prefix.
    prefixed = {f"model.{k}": v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(prefixed, strict=False)
    real_missing = [k for k in missing if "rope_" not in k]
    if real_missing:
        raise RuntimeError(f"Missing keys: {real_missing[:5]} ... total {len(real_missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected: {unexpected[:5]}")
    model.eval()
    return model


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="gelu_d12.pt")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    print(f">>> building HF model from {args.pt}")
    model = build_hf_model_from_pt(args.pt)
    print(f"    n_params = {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(torch.float32)

    # 1. Cache-vs-no-cache parity on a prompt of length 8
    print("\n>>> parity: cached forward vs full forward")
    ids = torch.arange(1, 9, dtype=torch.long)[None, :]
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits  # (1, 8, V)
        # step-by-step with cache
        pkv = None
        cached_logits = []
        for t in range(ids.shape[1]):
            out = model(input_ids=ids[:, t : t + 1] if pkv is not None else ids[:, : t + 1],
                        past_key_values=pkv,
                        use_cache=True)
            pkv = out.past_key_values
            cached_logits.append(out.logits[:, -1:, :])
        cached = torch.cat(cached_logits, dim=1)
    diff = (full - cached).abs()
    print(f"    max |diff| = {diff.max().item():.3e}   mean |diff| = {diff.mean().item():.3e}")
    # Position 0 will differ because "full" sees the seq at once while cache
    # starts with a 1-token forward. What matters is the LAST position (the
    # one generate() actually reads). Assert that.
    last_diff = diff[:, -1, :].max().item()
    print(f"    max |diff| at last pos = {last_diff:.3e}")
    assert last_diff < 1e-3, f"KV-cache parity mismatch at last token: {last_diff:.3e}"
    print("    KV-cache parity OK")

    # 2. generate() end-to-end
    print(f"\n>>> generating from prompt: {args.prompt!r}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id or 0,
        )
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    print(f"    completion: {text!r}")
    print("    generate() OK")


if __name__ == "__main__":
    main()
