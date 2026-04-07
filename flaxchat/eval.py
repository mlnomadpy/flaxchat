"""
Evaluation utilities for flaxchat.

CORE metric (DCLM paper): In-Context Learning benchmark.
BPB (Bits Per Byte): Tokenization-agnostic loss metric.

Port of nanochat's core_eval.py and loss_eval.py for JAX.
"""

import math
import random

import jax
import jax.numpy as jnp
import numpy as np

from flaxchat.common import print0


# ---------------------------------------------------------------------------
# Prompt rendering for CORE metric
# ---------------------------------------------------------------------------
def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for multiple-choice questions."""
    fewshot_examples = fewshot_examples or []
    parts = []
    for ex in fewshot_examples:
        parts.append(f"{ex['query']}{continuation_delimiter}{ex['choices'][ex['gold']]}\n")
    prompts = []
    for choice in item['choices']:
        prompt = "".join(parts) + f"{item['query']}{continuation_delimiter}{choice}"
        prompts.append(prompt)
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for language modeling tasks."""
    fewshot_examples = fewshot_examples or []
    parts = []
    for ex in fewshot_examples:
        parts.append(f"{ex['context'].strip()}{continuation_delimiter}{ex['continuation']}\n")
    prefix = "".join(parts) + f"{item['context'].strip()}{continuation_delimiter}"
    prompt_without = prefix
    prompt_with = prefix + item['continuation']
    return [prompt_without.strip(), prompt_with]


def find_common_length(token_sequences, direction='left'):
    """Find length of common prefix/suffix across token sequences."""
    min_len = min(len(seq) for seq in token_sequences)
    indices = range(min_len) if direction == 'left' else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


# ---------------------------------------------------------------------------
# Forward model and evaluate
# ---------------------------------------------------------------------------
def forward_model(model, input_ids):
    """
    Forward pass returning per-position losses and predictions.
    input_ids: (B, T) int32
    Returns: (losses (B, T), predictions (B, T))
    """
    logits = model(input_ids)  # (B, T, vocab_size)
    B, T, V = logits.shape

    # Autoregressive targets: shift left by 1
    target_ids = jnp.roll(input_ids, shift=-1, axis=1)

    # Cross-entropy at each position
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(target_ids, V)
    losses = -jnp.sum(one_hot * log_probs, axis=-1)  # (B, T)

    # Last column is invalid (no target)
    losses = losses.at[:, -1].set(float('nan'))

    predictions = jnp.argmax(logits, axis=-1)  # (B, T)
    return losses, predictions


def evaluate_example_mc(model, tokenizer, item, fewshot_examples, continuation_delimiter):
    """Evaluate a single multiple-choice example."""
    prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer(prompts, prepend=bos)

    # Find common prefix
    answer_start = find_common_length(tokens, 'left')
    start_indices = [answer_start] * len(prompts)
    end_indices = [len(t) for t in tokens]

    # Pad and stack
    max_len = max(len(t) for t in tokens)
    padded = np.full((len(tokens), max_len), bos, dtype=np.int32)
    for i, t in enumerate(tokens):
        padded[i, :len(t)] = t

    input_ids = jnp.array(padded)
    losses, _ = forward_model(model, input_ids)

    # Find option with lowest average loss
    mean_losses = []
    for i, (si, ei) in enumerate(zip(start_indices, end_indices)):
        mean_losses.append(float(jnp.nanmean(losses[i, si - 1:ei - 1])))
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == item['gold']


def evaluate_example_lm(model, tokenizer, item, fewshot_examples, continuation_delimiter):
    """Evaluate a single language modeling example."""
    prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer(prompts, prepend=bos)
    tokens_without, tokens_with = tokens

    start_idx = len(tokens_without)
    end_idx = len(tokens_with)
    assert start_idx < end_idx

    input_ids = jnp.array([tokens_with], dtype=jnp.int32)
    _, predictions = forward_model(model, input_ids)

    predicted = predictions[0, start_idx - 1:end_idx - 1]
    actual = jnp.array(tokens_with[start_idx:end_idx])
    return bool(jnp.all(predicted == actual))


# ---------------------------------------------------------------------------
# CORE metric evaluation
# ---------------------------------------------------------------------------
# CORE task definitions (from DCLM paper)
CORE_TASKS = {
    'hellaswag': {'task_type': 'multiple_choice', 'num_fewshot': 10, 'continuation_delimiter': '\n', 'dataset': 'Rowan/hellaswag', 'split': 'validation', 'baseline': 0.2534},
    'arc_easy': {'task_type': 'multiple_choice', 'num_fewshot': 25, 'continuation_delimiter': '\n', 'dataset': 'allenai/ai2_arc:ARC-Easy', 'split': 'test', 'baseline': 0.2527},
    'arc_challenge': {'task_type': 'multiple_choice', 'num_fewshot': 25, 'continuation_delimiter': '\n', 'dataset': 'allenai/ai2_arc:ARC-Challenge', 'split': 'test', 'baseline': 0.2099},
    'piqa': {'task_type': 'multiple_choice', 'num_fewshot': 5, 'continuation_delimiter': '\n', 'dataset': 'ybisk/piqa', 'split': 'validation', 'baseline': 0.5},
    'winogrande': {'task_type': 'multiple_choice', 'num_fewshot': 5, 'continuation_delimiter': '\n', 'dataset': 'allenai/winogrande:winogrande_xl', 'split': 'validation', 'baseline': 0.5},
    'mmlu': {'task_type': 'multiple_choice', 'num_fewshot': 5, 'continuation_delimiter': ' ', 'dataset': 'cais/mmlu:all', 'split': 'validation', 'baseline': 0.25},
}


def evaluate_core(model, tokenizer, max_per_task=500):
    """
    Evaluate the CORE metric (DCLM paper).

    Returns dict with:
    - core_metric: float — the aggregate CORE score
    - centered_results: dict — per-task centered accuracy
    """
    print0("Evaluating CORE metric...")
    results = {}

    for task_name, meta in CORE_TASKS.items():
        print0(f"  {task_name}...")
        try:
            from datasets import load_dataset
            ds_parts = meta['dataset'].split(':')
            if len(ds_parts) == 2:
                data = load_dataset(ds_parts[0], ds_parts[1], split=meta['split'])
            else:
                data = load_dataset(ds_parts[0], split=meta['split'])

            # Limit examples
            n = min(max_per_task, len(data))
            indices = list(range(n))

            correct = 0
            for idx in indices:
                item = data[idx]
                # Build item in CORE format
                if meta['task_type'] == 'multiple_choice':
                    # Need to format into query/choices/gold format
                    if 'endings' in item:
                        # HellaSwag format
                        core_item = {
                            'query': item.get('ctx', item.get('activity_label', '')),
                            'choices': item['endings'],
                            'gold': int(item['label']) if item['label'] != '' else 0,
                        }
                    elif 'choices' in item and isinstance(item['choices'], dict):
                        # ARC format
                        core_item = {
                            'query': item['question'],
                            'choices': item['choices']['text'],
                            'gold': item['choices']['label'].index(item['answerKey']) if item['answerKey'] in item['choices']['label'] else 0,
                        }
                    elif 'choices' in item and isinstance(item['choices'], list):
                        # MMLU format
                        core_item = {
                            'query': item['question'],
                            'choices': item['choices'],
                            'gold': item['answer'],
                        }
                    elif 'sol1' in item:
                        # PIQA format
                        core_item = {
                            'query': item['goal'],
                            'choices': [item['sol1'], item['sol2']],
                            'gold': int(item['label']),
                        }
                    elif 'sentence' in item:
                        # Winogrande format
                        core_item = {
                            'query': item['sentence'],
                            'choices': [item['option1'], item['option2']],
                            'gold': int(item['answer']) - 1 if item['answer'] != '' else 0,
                        }
                    else:
                        continue

                    fewshot = []
                    if meta['num_fewshot'] > 0:
                        rng = random.Random(1234 + idx)
                        avail = [i for i in range(len(data)) if i != idx]
                        fewshot_idx = rng.sample(avail, min(meta['num_fewshot'], len(avail)))
                        # Build fewshot items (simplified)

                    is_correct = evaluate_example_mc(
                        model, tokenizer, core_item, fewshot,
                        meta['continuation_delimiter']
                    )
                    correct += int(is_correct)

            accuracy = correct / n if n > 0 else 0.0
            centered = accuracy - meta['baseline']
            results[task_name] = {'accuracy': accuracy, 'centered': centered}
            print0(f"    {task_name}: {accuracy:.4f} (centered: {centered:+.4f})")

        except Exception as e:
            print0(f"    {task_name}: FAILED ({e})")
            results[task_name] = {'accuracy': 0.0, 'centered': -meta['baseline']}

    # Aggregate CORE metric = mean of centered accuracies
    centered_values = [r['centered'] for r in results.values()]
    core_metric = sum(centered_values) / len(centered_values) if centered_values else 0.0

    print0(f"CORE metric: {core_metric:.4f}")
    return {
        'core_metric': core_metric,
        'centered_results': {k: v['centered'] for k, v in results.items()},
        'raw_results': results,
    }


# ---------------------------------------------------------------------------
# BPB (Bits Per Byte) evaluation — tokenization-agnostic metric
# ---------------------------------------------------------------------------
def compute_token_bytes(tokenizer) -> np.ndarray:
    """
    Build a tensor mapping each token ID to its byte length.
    Special tokens get 0 bytes (excluded from BPB calculation).
    """
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = np.zeros(vocab_size, dtype=np.int32)
    special_tokens = tokenizer.get_special_tokens()

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_tokens:
            token_bytes[token_id] = 0  # exclude special tokens
        else:
            token_bytes[token_id] = len(token_str.encode('utf-8'))

    return token_bytes


def evaluate_bpb(model, val_loader, eval_steps, token_bytes=None, tokenizer=None):
    """
    Evaluate Bits Per Byte (BPB) — tokenization-agnostic loss metric.

    Unlike simple average loss, BPB normalizes by the actual byte count of each
    target token. This makes the metric comparable across different tokenizers
    and vocab sizes.

    Port of nanochat's loss_eval.evaluate_bpb.

    Args:
        model: GPT model
        val_loader: Iterator yielding (inputs, targets) numpy arrays
        eval_steps: Number of batches to evaluate
        token_bytes: np.array of shape (vocab_size,) with byte count per token.
                     If None and tokenizer is provided, computed automatically.
        tokenizer: Used to compute token_bytes if not provided

    Returns:
        bpb: float — bits per byte (lower is better)
    """
    if token_bytes is None and tokenizer is not None:
        token_bytes = compute_token_bytes(tokenizer)

    total_nats = 0.0
    total_bytes = 0

    for step in range(eval_steps):
        inputs_np, targets_np = next(val_loader)
        inputs = jnp.array(inputs_np)
        targets = jnp.array(targets_np)

        # Get per-position loss (not reduced)
        logits = model(inputs)  # (B, T, V)
        B, T, V = logits.shape
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(targets, V)
        per_token_loss = -jnp.sum(one_hot * log_probs, axis=-1)  # (B, T)

        if token_bytes is not None:
            # Proper BPB: weight by actual byte count
            targets_flat = targets_np.flatten()
            valid = targets_flat >= 0
            targets_safe = np.where(valid, targets_flat, 0)
            num_bytes_flat = np.where(valid, token_bytes[targets_safe], 0)
            num_bytes_2d = num_bytes_flat.reshape(B, T)

            # Only count tokens with bytes > 0 (excludes special tokens)
            byte_mask = jnp.array(num_bytes_2d > 0, dtype=jnp.float32)
            total_nats += float(jnp.sum(per_token_loss * byte_mask))
            total_bytes += int(np.sum(num_bytes_2d))
        else:
            # Fallback: approximate BPB (assume ~4 bytes per token)
            mask = (targets >= 0).astype(jnp.float32)
            total_nats += float(jnp.sum(per_token_loss * mask))
            total_bytes += int(jnp.sum(mask)) * 4  # approximate

    if total_bytes == 0:
        return float('inf')

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
