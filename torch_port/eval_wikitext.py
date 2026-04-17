"""
Evaluate models on wikitext-103 test set — reports cross-entropy loss and perplexity.

Supports:
  - GELU model:      --model-type gelu --weights gelu_d12.pt
  - YatNMN model:    --model-type yatnmn --weights yatnmn_softplus_d12.pt
  - HF repo:         --model-type hf --weights mlnomad/gelu-d12-chinchilla-261M-pytorch

Usage:
    pixi run python torch_port/eval_wikitext.py --model-type gelu --weights gelu_d12.pt
    pixi run python torch_port/eval_wikitext.py --model-type yatnmn --weights yatnmn_softplus_d12.pt --tag "pn"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))


def load_model(model_type: str, weights: str):
    if model_type == "gelu":
        from torch_port.torch_gpt import GELU_GPT
        return GELU_GPT.from_pretrained(weights, map_location="cpu").to(torch.float32).eval()
    elif model_type == "yatnmn":
        from torch_port.yatnmn_gpt import Yat_GPT
        return Yat_GPT.from_pretrained(weights, map_location="cpu").to(torch.float32).eval()
    elif model_type == "hf":
        return AutoModelForCausalLM.from_pretrained(
            weights, trust_remote_code=True, dtype=torch.float32
        ).eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def eval_wikitext(model, tokenizer, seq_len=1024, max_tokens=2_000_000, model_type="gelu"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Concatenate all text into one long string, then tokenize in chunks
    all_text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(all_text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]  # (total_tokens,)

    total_tokens = min(len(input_ids), max_tokens + seq_len)
    stride = seq_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    total_loss = 0.0
    total_count = 0

    print(f"Evaluating on wikitext-103 test ({total_tokens:,} tokens, stride={stride})")

    for start in range(0, total_tokens - 1, stride):
        end = min(start + seq_len + 1, total_tokens)
        chunk = input_ids[start:end].unsqueeze(0)  # (1, chunk_len)
        inp = chunk[:, :-1]
        tgt = chunk[:, 1:]

        if inp.shape[1] == 0:
            continue

        with torch.no_grad():
            if model_type == "hf":
                logits = model(input_ids=inp, use_cache=False).logits
            else:
                logits = model(inp)

        # Cross-entropy per token
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt.reshape(-1).to(torch.long)

        # Mask padding tokens
        mask = (tgt_flat != pad_id).float()
        loss_per_tok = F.cross_entropy(logits_flat, tgt_flat, reduction="none")
        total_loss += (loss_per_tok * mask).sum().item()
        total_count += mask.sum().item()

        if total_count > max_tokens:
            break

        if start > 0 and start % (stride * 100) == 0:
            avg = total_loss / total_count
            print(f"  {total_count:,} tokens | loss {avg:.4f} | ppl {np.exp(avg):.2f}")

    avg_loss = total_loss / total_count
    ppl = np.exp(avg_loss)
    return avg_loss, ppl, int(total_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, choices=["gelu", "yatnmn", "hf"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", default="", help="Label for output")
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    args = ap.parse_args()

    print(f"[{args.tag}] Loading model from {args.weights}")
    model = load_model(args.model_type, args.weights)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loss, ppl, n_tok = eval_wikitext(
        model, tokenizer, max_tokens=args.max_tokens, model_type=args.model_type,
    )
    print(f"\n[{args.tag}] wikitext-103 test | loss={loss:.4f} | ppl={ppl:.2f} | tokens={n_tok:,}")


if __name__ == "__main__":
    main()
