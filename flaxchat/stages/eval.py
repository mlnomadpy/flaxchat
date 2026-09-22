"""
Evaluate a trained model.

Usage:
    python -m scripts.eval --model=d12 --checkpoint-type=base
    python -m scripts.eval --model=d12 --checkpoint-type=sft --tasks=mmlu,gsm8k
"""

import argparse
import json
import os
from dataclasses import dataclass

from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig, GPTConfig
from flaxchat.common import compute_init, print0, print_banner, get_base_dir
from flaxchat.tokenizer import get_tokenizer, load_tokenizer
from flaxchat.eval import evaluate_core
from flaxchat.engine import generate_with_cache
from flaxchat.checkpoint import restore_model_from_checkpoint, load_checkpoint_metadata, validate_checkpoint_tokenizer
from flaxchat.stages import RequestMixin, StageResult


@dataclass(frozen=True)
class EvalRequest(RequestMixin):
    resolved_config: FlaxChatConfig | None = None
    tokenizer_dir: str | None = None
    checkpoint_dir: str | None = None
    model: str = "d12"
    checkpoint_type: str = "base"
    tasks: str = "core"
    max_per_task: int = 0
    manifest_path: str = "core_eval_manifest.json"
    temperature: float = 0.0
    max_tokens: int = 512


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", type=str, default="d12")
    parser.add_argument("--checkpoint-type", type=str, default="base", choices=["base", "sft", "rl"])
    parser.add_argument("--tasks", type=str, default="core", help="core | mmlu | gsm8k | arc | all")
    parser.add_argument("--max-per-task", type=int, default=0, help="0 evaluates the full pinned split")
    parser.add_argument("--manifest-path", type=str, default="core_eval_manifest.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tokenizer-dir", help="Tokenizer artifact matching the checkpoint")
    parser.add_argument("--checkpoint-dir", help="Explicit input checkpoint directory")
    return parser


def run(request: EvalRequest) -> StageResult:
    print_banner()
    args = request
    if args.max_per_task < 0 or args.max_tokens < 1:
        raise ValueError("max_per_task must be non-negative and max_tokens positive")

    task_list = [name.strip() for name in args.tasks.split(",")]
    if not task_list or any(name not in {"core", "all", "mmlu", "gsm8k", "arc"} for name in task_list):
        raise ValueError("Unknown or empty evaluation task")
    limit = args.max_per_task or None

    # Init
    compute_init()
    tokenizer = load_tokenizer(args.tokenizer_dir) if args.tokenizer_dir else get_tokenizer()
    base_dir = get_base_dir()
    checkpoint_dir = args.checkpoint_dir or f"{base_dir}/{args.checkpoint_type}_checkpoints/{args.model}"
    metadata = load_checkpoint_metadata(checkpoint_dir)
    validate_checkpoint_tokenizer(metadata, tokenizer,
        tokenizer_path=args.tokenizer_dir or os.path.join(get_base_dir(), 'tokenizer'))
    model_config = GPTConfig(**metadata["model_config"])
    config = args.resolved_config or FlaxChatConfig(model=model_config)
    if config.model != model_config:
        raise ValueError("Evaluation architecture differs from checkpoint")
    model = GPT(model_config, rngs=nnx.Rngs(0))
    step = metadata.get("step")
    if step is None:
        raise ValueError("Checkpoint metadata lacks a concrete step")
    restore_model_from_checkpoint(model, checkpoint_dir, step=step)
    print0(f"Model loaded: {model.num_params():,} params")

    # Evaluate
    results = {}

    if "core" in task_list or "all" in task_list:
        print0("\n=== CORE Metric ===")
        core_results = evaluate_core(
            model, tokenizer, max_per_task=args.max_per_task,
            manifest_path=args.manifest_path,
            model_checkpoint_identity=json.dumps({"path": checkpoint_dir, "step": step, "metadata": metadata}, sort_keys=True),
        )
        results["core"] = core_results
        if core_results['core_metric'] is None:
            print0("CORE: INCOMPLETE (see task errors and manifest)")
        else:
            print0(f"CORE: {core_results['core_metric']:.4f}")

    if "mmlu" in task_list or "all" in task_list:
        print0("\n=== MMLU ===")
        from tasks.mmlu import MMLU
        mmlu = MMLU(subset="all", split="validation", stop=limit)
        correct = 0
        total = len(mmlu)
        for i in range(total):
            conv = mmlu[i]
            prompt_tokens = tokenizer.encode(conv['messages'][0]['content'], prepend=tokenizer.get_bos_token_id())
            output = generate_with_cache(model, prompt_tokens, max_tokens=1, temperature=0)
            pred_token = output[-1]
            pred_text = tokenizer.decode([pred_token]).strip()

            if pred_text in conv.get('letters', ('A', 'B', 'C', 'D')):
                correct += int(mmlu.evaluate(conv, pred_text))

        if total == 0:
            raise ValueError("Evaluation dataset is empty")
        accuracy = correct / total
        results["mmlu"] = {"accuracy": accuracy, "correct": correct, "total": total}
        print0(f"MMLU: {accuracy:.4f} ({correct}/{total})")

    if "gsm8k" in task_list or "all" in task_list:
        print0("\n=== GSM8K ===")
        from tasks.gsm8k import GSM8K
        gsm = GSM8K(subset="main", split="test", stop=limit)
        correct = 0
        total = len(gsm)
        for i in range(total):
            conv = gsm[i]
            prompt_tokens = tokenizer.encode(conv['messages'][0]['content'], prepend=tokenizer.get_bos_token_id())
            output = generate_with_cache(model, prompt_tokens, max_tokens=args.max_tokens, temperature=args.temperature)
            response_text = tokenizer.decode(output[len(prompt_tokens):])
            correct += gsm.evaluate(conv, response_text)

        if total == 0:
            raise ValueError("Evaluation dataset is empty")
        accuracy = correct / total
        results["gsm8k"] = {"accuracy": accuracy, "correct": correct, "total": total}
        print0(f"GSM8K: {accuracy:.4f} ({correct}/{total})")

    if "arc" in task_list or "all" in task_list:
        print0("\n=== ARC-Challenge ===")
        from tasks.arc import ARC
        arc = ARC(subset="ARC-Challenge", split="test", stop=limit)

        correct = 0
        total = len(arc)
        for i in range(total):
            conv = arc[i]
            prompt_tokens = tokenizer.encode(conv['messages'][0]['content'], prepend=tokenizer.get_bos_token_id())
            output = generate_with_cache(model, prompt_tokens, max_tokens=1, temperature=0)
            pred_token = output[-1]
            pred_text = tokenizer.decode([pred_token]).strip()

            if pred_text in conv.get('letters', []):
                correct += int(arc.evaluate(conv, pred_text))

        if total == 0:
            raise ValueError("Evaluation dataset is empty")
        accuracy = correct / total
        results["arc"] = {"accuracy": accuracy, "correct": correct, "total": total}
        print0(f"ARC-Challenge: {accuracy:.4f} ({correct}/{total})")

    # Summary
    print0("\n=== Summary ===")
    print0(json.dumps(results, indent=2, default=str))
    return StageResult(
        stage="eval",
        exit_code=int(any(value.get("status") == "incomplete" for value in results.values())),
        resolved_config=config.to_dict(),
        metrics=results,
        artifact_paths=(args.manifest_path,),
    )
