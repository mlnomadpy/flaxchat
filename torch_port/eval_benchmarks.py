"""
Lightweight eval suite for small LMs: LAMBADA, HellaSwag, ARC-Easy, PIQA.
All use log-likelihood ranking (no generation needed).

Usage:
    pixi run python torch_port/eval_benchmarks.py --model-type gelu --weights gelu_d12.pt --tag GELU
    pixi run python torch_port/eval_benchmarks.py --model-type yatnmn --weights yatnmn_softplus_d12.pt --tag "pn+α"
"""
from __future__ import annotations

import argparse, sys, math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
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
    else:
        raise ValueError(f"Unknown: {model_type}")


def log_likelihood(model, tokenizer, text: str, seq_len: int = 1024) -> float:
    """Compute total log-likelihood of `text` under the model."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    ids = enc["input_ids"]  # (1, T)
    if ids.shape[1] < 2:
        return -100.0
    with torch.no_grad():
        logits = model(ids[:, :-1])  # (1, T-1, V)
    targets = ids[:, 1:]  # (1, T-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_lps = log_probs[0].gather(1, targets[0].unsqueeze(-1)).squeeze(-1)
    return float(token_lps.sum())


def conditional_log_likelihood(model, tokenizer, context: str, continuation: str,
                                seq_len: int = 1024) -> float:
    """Log-likelihood of `continuation` conditioned on `context`."""
    ctx_enc = tokenizer(context, add_special_tokens=False)
    cont_enc = tokenizer(continuation, add_special_tokens=False)
    ctx_ids = ctx_enc["input_ids"]
    cont_ids = cont_enc["input_ids"]
    full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)
    if full_ids.shape[1] > seq_len:
        full_ids = full_ids[:, :seq_len]
    n_ctx = len(ctx_ids)
    if n_ctx >= full_ids.shape[1]:
        return -1e9
    with torch.no_grad():
        logits = model(full_ids[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    targets = full_ids[:, 1:]
    token_lps = log_probs[0].gather(1, targets[0].unsqueeze(-1)).squeeze(-1)
    # Only score the continuation tokens (positions n_ctx-1 onward in the shifted sequence)
    cont_lps = token_lps[max(n_ctx - 1, 0):]
    return float(cont_lps.sum())


# --- LAMBADA ---
def eval_lambada(model, tokenizer, max_examples: int = 5000):
    ds = load_dataset("lambada", split="test")
    correct = 0
    total = 0
    total_ll = 0.0
    total_toks = 0

    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        text = ex["text"]
        words = text.strip().split()
        if len(words) < 2:
            continue
        last_word = words[-1]
        context = " ".join(words[:-1])

        # Score the actual last word
        ctx_enc = tokenizer(context, add_special_tokens=False)["input_ids"]
        full_enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        n_ctx = len(ctx_enc)
        n_last = len(full_enc) - n_ctx

        ids = torch.tensor([full_enc], dtype=torch.long)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(ids[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        targets = ids[:, 1:]
        token_lps = log_probs[0].gather(1, targets[0].unsqueeze(-1)).squeeze(-1)

        # Check if greedy decode of the last n_last tokens matches
        greedy_preds = logits[0, max(n_ctx - 1, 0):].argmax(dim=-1)
        actual = ids[0, n_ctx:]
        if greedy_preds.shape[0] >= actual.shape[0]:
            if torch.equal(greedy_preds[:actual.shape[0]], actual):
                correct += 1

        last_lps = token_lps[max(n_ctx - 1, 0):]
        total_ll += float(last_lps.sum())
        total_toks += last_lps.shape[0]
        total += 1

    acc = correct / max(total, 1)
    ppl = math.exp(-total_ll / max(total_toks, 1))
    return {"acc": acc, "ppl": ppl, "n": total}


# --- HellaSwag ---
def eval_hellaswag(model, tokenizer, max_examples: int = 5000):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    correct = 0
    total = 0

    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])

        scores = []
        for end in endings:
            ll = conditional_log_likelihood(model, tokenizer, ctx, " " + end)
            scores.append(ll)

        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1
        total += 1

        if total % 500 == 0:
            print(f"  hellaswag: {total} done, acc={correct/total:.3f}")

    return {"acc": correct / max(total, 1), "n": total}


# --- ARC-Easy ---
def eval_arc_easy(model, tokenizer, max_examples: int = 5000):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    correct = 0
    total = 0

    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        question = ex["question"]
        choices = ex["choices"]
        labels = choices["label"]
        texts = choices["text"]
        answer_key = ex["answerKey"]

        ctx = f"Question: {question}\nAnswer:"
        scores = []
        for t in texts:
            ll = conditional_log_likelihood(model, tokenizer, ctx, " " + t)
            scores.append(ll)

        pred_idx = int(np.argmax(scores))
        pred_label = labels[pred_idx]
        if pred_label == answer_key:
            correct += 1
        total += 1

    return {"acc": correct / max(total, 1), "n": total}


# --- PIQA ---
def eval_piqa(model, tokenizer, max_examples: int = 5000):
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    correct = 0
    total = 0

    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        goal = ex["goal"]
        sol1 = ex["sol1"]
        sol2 = ex["sol2"]
        label = int(ex["label"])

        s1 = conditional_log_likelihood(model, tokenizer, goal, " " + sol1)
        s2 = conditional_log_likelihood(model, tokenizer, goal, " " + sol2)
        pred = 0 if s1 >= s2 else 1

        if pred == label:
            correct += 1
        total += 1

    return {"acc": correct / max(total, 1), "n": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, choices=["gelu", "yatnmn"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--tasks", default="lambada,hellaswag,arc_easy,piqa",
                    help="Comma-separated list of tasks")
    ap.add_argument("--max-examples", type=int, default=5000)
    args = ap.parse_args()

    print(f"[{args.tag}] Loading {args.weights}")
    model = load_model(args.model_type, args.weights)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = [t.strip() for t in args.tasks.split(",")]
    results = {}

    for task in tasks:
        print(f"\n[{args.tag}] === {task} ===")
        if task == "lambada":
            r = eval_lambada(model, tokenizer, args.max_examples)
            print(f"  acc={r['acc']:.4f}  ppl={r['ppl']:.2f}  n={r['n']}")
        elif task == "hellaswag":
            r = eval_hellaswag(model, tokenizer, args.max_examples)
            print(f"  acc={r['acc']:.4f}  n={r['n']}")
        elif task == "arc_easy":
            r = eval_arc_easy(model, tokenizer, args.max_examples)
            print(f"  acc={r['acc']:.4f}  n={r['n']}")
        elif task == "piqa":
            r = eval_piqa(model, tokenizer, args.max_examples)
            print(f"  acc={r['acc']:.4f}  n={r['n']}")
        else:
            print(f"  unknown task: {task}")
            continue
        results[task] = r

    print(f"\n[{args.tag}] SUMMARY:")
    for task, r in results.items():
        acc = r.get("acc", 0)
        extra = f"  ppl={r['ppl']:.1f}" if "ppl" in r else ""
        print(f"  {task:15s}  acc={acc:.4f}{extra}")


if __name__ == "__main__":
    main()
