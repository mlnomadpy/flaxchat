"""Auditable CORE and bits-per-byte evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random

import jax
import jax.numpy as jnp
import numpy as np

from flaxchat.common import print0


PROMPT_TEMPLATE_VERSION = "core-v2"


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    demonstrations = "".join(
        f"{example['query']}{continuation_delimiter}"
        f"{example['choices'][example['gold']]}\n"
        for example in fewshot_examples
    )
    return [
        f"{demonstrations}{item['query']}{continuation_delimiter}{choice}"
        for choice in item['choices']
    ]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    demonstrations = "".join(
        f"{example['context'].strip()}{continuation_delimiter}"
        f"{example['continuation']}\n"
        for example in fewshot_examples
    )
    prefix = demonstrations + f"{item['context'].strip()}{continuation_delimiter}"
    return [prefix.strip(), prefix + item['continuation']]


def find_common_length(token_sequences, direction='left'):
    min_len = min(len(sequence) for sequence in token_sequences)
    indices = range(min_len) if direction == 'left' else range(-1, -min_len - 1, -1)
    for offset, index in enumerate(indices):
        token = token_sequences[0][index]
        if not all(sequence[index] == token for sequence in token_sequences):
            return offset
    return min_len


def forward_model(model, input_ids):
    """Return autoregressive token losses and predictions without one-hot targets."""
    logits = model(input_ids)
    target_ids = jnp.roll(input_ids, shift=-1, axis=1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]
    losses = losses.at[:, -1].set(float('nan'))
    return losses, jnp.argmax(logits, axis=-1)


def evaluate_example_mc(model, tokenizer, item, fewshot_examples,
                        continuation_delimiter, return_record=False):
    prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer(prompts, prepend=bos)
    answer_start = find_common_length(tokens, 'left')
    max_len = max(map(len, tokens))
    padded = np.full((len(tokens), max_len), bos, dtype=np.int32)
    for index, sequence in enumerate(tokens):
        padded[index, :len(sequence)] = sequence
    losses, _ = forward_model(model, jnp.array(padded))
    scores = [
        float(jnp.nanmean(losses[index, answer_start - 1:len(sequence) - 1]))
        for index, sequence in enumerate(tokens)
    ]
    prediction = scores.index(min(scores))
    correct = prediction == item['gold']
    record = {'prediction': prediction, 'gold': item['gold'], 'scores': scores}
    return (correct, record) if return_record else correct


def evaluate_example_lm(model, tokenizer, item, fewshot_examples,
                        continuation_delimiter, return_record=False):
    without, with_continuation = tokenizer(
        render_prompts_lm(item, continuation_delimiter, fewshot_examples),
        prepend=tokenizer.get_bos_token_id(),
    )
    start, end = len(without), len(with_continuation)
    if start >= end:
        raise ValueError("Continuation must add at least one token")
    _, predictions = forward_model(
        model, jnp.array([with_continuation], dtype=jnp.int32)
    )
    predicted = predictions[0, start - 1:end - 1]
    actual = jnp.array(with_continuation[start:end])
    correct = bool(jnp.all(predicted == actual))
    record = {
        'prediction_tokens': np.asarray(predicted).tolist(),
        'gold_tokens': np.asarray(actual).tolist(),
    }
    return (correct, record) if return_record else correct


# Dataset commits were resolved on 2026-09-02. Few-shot examples come from a
# dedicated training/dev split, never from the scored example set.
CORE_TASKS = {
    'hellaswag': {'task_type': 'multiple_choice', 'num_fewshot': 10, 'continuation_delimiter': '\n', 'dataset': 'Rowan/hellaswag', 'revision': '218ec52e09a7e7462a5400043bb9a69a41d06b76', 'split': 'validation', 'fewshot_split': 'train', 'baseline': 0.2534},
    'arc_easy': {'task_type': 'multiple_choice', 'num_fewshot': 25, 'continuation_delimiter': '\n', 'dataset': 'allenai/ai2_arc:ARC-Easy', 'revision': '210d026faf9955653af8916fad021475a3f00453', 'split': 'test', 'fewshot_split': 'train', 'baseline': 0.2527},
    'arc_challenge': {'task_type': 'multiple_choice', 'num_fewshot': 25, 'continuation_delimiter': '\n', 'dataset': 'allenai/ai2_arc:ARC-Challenge', 'revision': '210d026faf9955653af8916fad021475a3f00453', 'split': 'test', 'fewshot_split': 'train', 'baseline': 0.2099},
    'piqa': {'task_type': 'multiple_choice', 'num_fewshot': 5, 'continuation_delimiter': '\n', 'dataset': 'ybisk/piqa', 'revision': '2e8ac2dffd59bac8c3c6714948f4c551a0848bb0', 'split': 'validation', 'fewshot_split': 'train', 'baseline': 0.5},
    'winogrande': {'task_type': 'multiple_choice', 'num_fewshot': 5, 'continuation_delimiter': '\n', 'dataset': 'allenai/winogrande:winogrande_xl', 'revision': '01e74176c63542e6b0bcb004dcdea22d94fb67b5', 'split': 'validation', 'fewshot_split': 'train', 'baseline': 0.5},
    'mmlu': {'task_type': 'multiple_choice', 'num_fewshot': 5, 'continuation_delimiter': ' ', 'dataset': 'cais/mmlu:all', 'revision': 'c30699e8356da336a370243923dbaf21066bb9fe', 'split': 'validation', 'fewshot_split': 'dev', 'baseline': 0.25},
}


def normalize_core_item(item):
    if 'endings' in item:
        return {'query': item.get('ctx', item.get('activity_label', '')), 'choices': item['endings'], 'gold': int(item['label'])}
    if 'choices' in item and isinstance(item['choices'], dict):
        labels = item['choices']['label']
        if item['answerKey'] not in labels:
            raise ValueError(f"Answer key {item['answerKey']!r} is missing")
        return {'query': item['question'], 'choices': item['choices']['text'], 'gold': labels.index(item['answerKey'])}
    if 'choices' in item and isinstance(item['choices'], list):
        return {'query': item['question'], 'choices': item['choices'], 'gold': int(item['answer'])}
    if 'sol1' in item:
        return {'query': item['goal'], 'choices': [item['sol1'], item['sol2']], 'gold': int(item['label'])}
    if 'sentence' in item:
        return {'query': item['sentence'], 'choices': [item['option1'], item['option2']], 'gold': int(item['answer']) - 1}
    if 'context' in item and 'continuation' in item:
        return {'context': item['context'], 'continuation': item['continuation']}
    raise ValueError(f"Unsupported CORE fields: {sorted(item)}")


def _dataset_args(spec):
    parts = spec['dataset'].split(':', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0],)


def _wilson_interval(correct, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _protocol_hash(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def evaluate_core(model, tokenizer, max_per_task=None, *, seed=1234,
                  manifest_path="core_eval_manifest.json",
                  model_checkpoint_identity="unavailable"):
    """Evaluate pinned CORE tasks, failing the aggregate closed on any task error."""
    print0("Evaluating CORE metric...")
    results, audit_records = {}, {}
    manifest = {
        'version': 2,
        'seed': seed,
        'prompt_template_version': PROMPT_TEMPLATE_VERSION,
        'tokenizer_identity': {
            'class': f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            'vocab_size': tokenizer.get_vocab_size(),
        },
        'model_checkpoint_identity': model_checkpoint_identity,
        'tasks': {},
    }
    for task_name, spec in CORE_TASKS.items():
        print0(f"  {task_name}...")
        try:
            from datasets import load_dataset
            args = _dataset_args(spec)
            data = load_dataset(*args, split=spec['split'], revision=spec['revision'])
            fewshot_data = load_dataset(
                *args, split=spec['fewshot_split'], revision=spec['revision']
            )
            if not max_per_task or max_per_task >= len(data):
                indices = list(range(len(data)))
            else:
                indices = sorted(random.Random(seed).sample(range(len(data)), max_per_task))
            task_manifest = {
                'dataset': spec['dataset'], 'revision': spec['revision'],
                'split': spec['split'], 'fewshot_split': spec['fewshot_split'],
                'example_indices': indices, 'fewshot_indices': {},
                'num_fewshot': spec['num_fewshot'],
            }
            manifest['tasks'][task_name] = task_manifest
            correct, records = 0, []
            for index in indices:
                item = normalize_core_item(data[index])
                available = list(range(len(fewshot_data)))
                if spec['split'] == spec['fewshot_split'] and index in available:
                    available.remove(index)
                if len(available) < spec['num_fewshot']:
                    raise ValueError(f"{task_name} has insufficient few-shot examples")
                fewshot_indices = random.Random(seed + index).sample(
                    available, spec['num_fewshot']
                )
                task_manifest['fewshot_indices'][str(index)] = fewshot_indices
                fewshot = [normalize_core_item(fewshot_data[i]) for i in fewshot_indices]
                evaluator = evaluate_example_mc if spec['task_type'] == 'multiple_choice' else evaluate_example_lm
                is_correct, record = evaluator(
                    model, tokenizer, item, fewshot, spec['continuation_delimiter'],
                    return_record=True,
                )
                correct += int(is_correct)
                record.update({
                    'example_index': index,
                    'fewshot_indices': fewshot_indices,
                    'correct': bool(is_correct),
                })
                records.append(record)
            total = len(indices)
            accuracy = correct / total
            results[task_name] = {
                'status': 'complete', 'accuracy': accuracy,
                'centered': accuracy - spec['baseline'], 'correct': correct,
                'sample_count': total,
                'confidence_interval_95': _wilson_interval(correct, total),
            }
            audit_records[task_name] = records
            print0(f"    {task_name}: {accuracy:.4f}")
        except Exception as error:
            print0(f"    {task_name}: FAILED ({error})")
            results[task_name] = {
                'status': 'failed', 'error': str(error), 'sample_count': 0,
            }

    complete = [result for result in results.values() if result['status'] == 'complete']
    run_complete = len(complete) == len(CORE_TASKS)
    core_metric = (
        sum(result['centered'] for result in complete) / len(complete)
        if run_complete else None
    )
    manifest['status'] = 'complete' if run_complete else 'incomplete'
    manifest['protocol_hash'] = _protocol_hash(manifest)
    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    print0(
        f"CORE metric: {core_metric:.4f}"
        if core_metric is not None else "CORE metric: INCOMPLETE"
    )
    return {
        'core_metric': core_metric,
        'status': manifest['status'],
        'protocol_hash': manifest['protocol_hash'],
        'manifest': manifest,
        'centered_results': {
            name: result['centered'] for name, result in results.items()
            if result['status'] == 'complete'
        },
        'raw_results': results,
        'audit_records': audit_records,
    }


def compute_token_bytes(tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = np.zeros(vocab_size, dtype=np.int32)
    special_tokens = tokenizer.get_special_tokens()
    for token_id in range(vocab_size):
        token = tokenizer.decode([token_id])
        if token not in special_tokens:
            token_bytes[token_id] = len(token.encode('utf-8'))
    return token_bytes


def evaluate_bpb(model, val_loader, eval_steps, token_bytes=None, tokenizer=None):
    if token_bytes is None and tokenizer is not None:
        token_bytes = compute_token_bytes(tokenizer)
    total_nats, total_bytes = 0.0, 0
    for _ in range(eval_steps):
        batch = next(val_loader)
        inputs_np, targets_np = batch[:2]
        inputs, targets = jnp.array(inputs_np), jnp.array(targets_np)
        logits = model(inputs)
        batch_size, sequence_length = targets.shape
        safe_targets = jnp.where(targets >= 0, targets, 0)
        per_token_loss = -jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), safe_targets[..., None], axis=-1
        )[..., 0]
        if token_bytes is not None:
            flat = targets_np.flatten()
            valid = flat >= 0
            safe = np.where(valid, flat, 0)
            bytes_2d = np.where(valid, token_bytes[safe], 0).reshape(
                batch_size, sequence_length
            )
            total_nats += float(jnp.sum(per_token_loss * jnp.array(bytes_2d > 0)))
            total_bytes += int(np.sum(bytes_2d))
        else:
            mask = targets >= 0
            total_nats += float(jnp.sum(per_token_loss * mask))
            total_bytes += int(jnp.sum(mask)) * 4
    return total_nats / (math.log(2) * total_bytes) if total_bytes else float('inf')
