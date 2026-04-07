"""
Evaluate a trained model.

Usage:
    python -m scripts.eval --model=d12 --checkpoint-type=base
    python -m scripts.eval --model=d12 --checkpoint-type=sft --tasks=mmlu,gsm8k
"""

import argparse
import json

import jax
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig
from flaxchat.common import compute_init, print0, print_banner, get_base_dir
from flaxchat.tokenizer import get_tokenizer
from flaxchat.eval import evaluate_core, evaluate_bpb
from flaxchat.engine import Engine, generate_with_cache
from flaxchat.checkpoint import restore_model_from_checkpoint

print_banner()

parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--model", type=str, default="d12")
parser.add_argument("--checkpoint-type", type=str, default="base", choices=["base", "sft", "rl"])
parser.add_argument("--tasks", type=str, default="core", help="core | mmlu | gsm8k | arc | all")
parser.add_argument("--max-per-task", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

# Init
compute_init()
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# Load model
depth = int(args.model.replace("d", ""))
config = FlaxChatConfig.from_depth(depth=depth, vocab_size=vocab_size)
model = GPT(config.model, rngs=nnx.Rngs(0))

base_dir = get_base_dir()
ckpt_dir = f"{args.checkpoint_type}_checkpoints"
checkpoint_dir = f"{base_dir}/{ckpt_dir}/{args.model}"
print0(f"Loading model from {checkpoint_dir}")
restore_model_from_checkpoint(model, checkpoint_dir)
print0(f"Model loaded: {model.num_params():,} params")

# Evaluate
results = {}
task_list = args.tasks.split(",")

if "core" in task_list or "all" in task_list:
    print0("\n=== CORE Metric ===")
    core_results = evaluate_core(model, tokenizer, max_per_task=args.max_per_task)
    results["core"] = core_results
    print0(f"CORE: {core_results['core_metric']:.4f}")

if "mmlu" in task_list or "all" in task_list:
    print0("\n=== MMLU ===")
    from tasks.mmlu import MMLU
    mmlu = MMLU(subset="all", split="validation", stop=args.max_per_task)
    engine = Engine(model, tokenizer)

    correct = 0
    total = len(mmlu)
    for i in range(total):
        conv = mmlu[i]
        prompt_tokens = tokenizer.encode(conv['messages'][0]['content'], prepend="<|bos|>")
        output = generate_with_cache(model, prompt_tokens, max_tokens=1, temperature=0)
        pred_token = output[-1]
        pred_text = tokenizer.decode([pred_token]).strip()

        if pred_text in conv.get('letters', ('A', 'B', 'C', 'D')):
            correct += int(mmlu.evaluate(conv, pred_text))

    accuracy = correct / total if total > 0 else 0
    results["mmlu"] = {"accuracy": accuracy, "correct": correct, "total": total}
    print0(f"MMLU: {accuracy:.4f} ({correct}/{total})")

if "gsm8k" in task_list or "all" in task_list:
    print0("\n=== GSM8K ===")
    from tasks.gsm8k import GSM8K
    gsm = GSM8K(subset="main", split="test", stop=args.max_per_task)
    engine = Engine(model, tokenizer)

    correct = 0
    total = len(gsm)
    for i in range(total):
        conv = gsm[i]
        prompt_tokens = tokenizer.encode(conv['messages'][0]['content'], prepend="<|bos|>")
        output = generate_with_cache(model, prompt_tokens, max_tokens=args.max_tokens, temperature=args.temperature)
        response_text = tokenizer.decode(output[len(prompt_tokens):])
        correct += gsm.evaluate(conv, response_text)

    accuracy = correct / total if total > 0 else 0
    results["gsm8k"] = {"accuracy": accuracy, "correct": correct, "total": total}
    print0(f"GSM8K: {accuracy:.4f} ({correct}/{total})")

if "arc" in task_list or "all" in task_list:
    print0("\n=== ARC-Challenge ===")
    from tasks.arc import ARC
    arc = ARC(subset="ARC-Challenge", split="test", stop=args.max_per_task)

    correct = 0
    total = len(arc)
    for i in range(total):
        conv = arc[i]
        prompt_tokens = tokenizer.encode(conv['messages'][0]['content'], prepend="<|bos|>")
        output = generate_with_cache(model, prompt_tokens, max_tokens=1, temperature=0)
        pred_token = output[-1]
        pred_text = tokenizer.decode([pred_token]).strip()

        if pred_text in conv.get('letters', []):
            correct += int(arc.evaluate(conv, pred_text))

    accuracy = correct / total if total > 0 else 0
    results["arc"] = {"accuracy": accuracy, "correct": correct, "total": total}
    print0(f"ARC-Challenge: {accuracy:.4f} ({correct}/{total})")

# Summary
print0("\n=== Summary ===")
print0(json.dumps(results, indent=2, default=str))
