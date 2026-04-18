"""
Quantization sensitivity test: compare GELU vs YatNMN degradation under
naive weight-only quantization (INT8, INT4).

Applies round-to-nearest symmetric quantization on all Linear/embedding weights,
then evaluates wikitext-103 loss. Reports Δ = quantized_loss - fp32_loss.
"""
import sys, math, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))


def quantize_weights(model, bits=8):
    """In-place naive symmetric weight quantization to `bits` precision."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    n_quantized = 0
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        w = param.data.float()
        scale = w.abs().max() / qmax
        if scale == 0:
            continue
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
        param.data.copy_(w_q.to(param.dtype))
        n_quantized += 1
    return n_quantized


def eval_wikitext_loss(model, tokenizer, model_type, max_tokens=500_000, seq_len=1024):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    all_text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(all_text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    pad_id = tokenizer.pad_token_id

    total_loss, total_count = 0.0, 0
    for start in range(0, min(len(input_ids), max_tokens + seq_len), seq_len):
        end = min(start + seq_len + 1, len(input_ids))
        chunk = input_ids[start:end].unsqueeze(0)
        inp, tgt = chunk[:, :-1], chunk[:, 1:]
        if inp.shape[1] == 0:
            continue
        with torch.no_grad():
            if model_type == "hf":
                logits = model(input_ids=inp, use_cache=False).logits
            else:
                logits = model(inp)
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt.reshape(-1).to(torch.long)
        mask = (tgt_flat != pad_id).float()
        loss = F.cross_entropy(logits_flat, tgt_flat, reduction="none")
        total_loss += (loss * mask).sum().item()
        total_count += mask.sum().item()
        if total_count > max_tokens:
            break
    return total_loss / total_count


def run_quant_test(model_type, weights_path, tag):
    if model_type == "gelu":
        from torch_port.torch_gpt import GELU_GPT
        load_fn = lambda: GELU_GPT.from_pretrained(weights_path, map_location="cpu").to(torch.float32).eval()
    elif model_type == "yatnmn":
        from torch_port.yatnmn_gpt import Yat_GPT
        load_fn = lambda: Yat_GPT.from_pretrained(weights_path, map_location="cpu").to(torch.float32).eval()
    else:
        raise ValueError(model_type)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[{tag}] FP32 eval...")
    model_fp32 = load_fn()
    loss_fp32 = eval_wikitext_loss(model_fp32, tokenizer, model_type)
    ppl_fp32 = math.exp(loss_fp32)
    print(f"  FP32: loss={loss_fp32:.4f}  ppl={ppl_fp32:.2f}")
    del model_fp32

    for bits in [8, 4]:
        model_q = load_fn()
        n = quantize_weights(model_q, bits=bits)
        print(f"  INT{bits}: quantized {n} weight tensors, evaluating...")
        loss_q = eval_wikitext_loss(model_q, tokenizer, model_type)
        ppl_q = math.exp(loss_q)
        delta_loss = loss_q - loss_fp32
        delta_ppl = ppl_q - ppl_fp32
        print(f"  INT{bits}: loss={loss_q:.4f}  ppl={ppl_q:.2f}  "
              f"Δloss={delta_loss:+.4f}  Δppl={delta_ppl:+.2f}")
        del model_q

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Quantization sensitivity: GELU vs YatNMN vs YatNMN-sb")
    print("=" * 60)

    tests = [
        ("gelu", "gelu_d12.pt", "GELU"),
        ("yatnmn", "yatnmn_softplus_d12.pt", "YatNMN pn+α"),
        ("yatnmn", "yatnmn_softplus_sb_d12.pt", "YatNMN sb+α"),
    ]

    for model_type, weights, tag in tests:
        run_quant_test(model_type, weights, tag)
