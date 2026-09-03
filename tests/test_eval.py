"""
Tests for evaluation utilities.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from flaxchat.eval import (
    evaluate_core,
    evaluate_example_mc,
    find_common_length,
    forward_model,
    render_prompts_lm,
    render_prompts_mc,
)


REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "core_reference.json").read_text()
)


def test_multiple_choice_prompt_matches_golden_reference():
    fixture = REFERENCE["multiple_choice"]
    assert render_prompts_mc(
        fixture["item"], fixture["delimiter"], fixture["fewshot"]
    ) == fixture["expected"]


def test_language_model_prompt_matches_golden_reference():
    fixture = REFERENCE["language_model"]
    assert render_prompts_lm(
        fixture["item"], fixture["delimiter"], fixture["fewshot"]
    ) == fixture["expected"]


def test_multiple_choice_label_and_scores_match_golden_reference(monkeypatch):
    class ReferenceTokenizer:
        def get_bos_token_id(self):
            return 0

        def __call__(self, prompts, prepend=None):
            assert prompts == REFERENCE["multiple_choice"]["expected"]
            return [[prepend, 10, 20], [prepend, 10, 21]]

    losses = jax.numpy.asarray([[9.0, 2.0, jax.numpy.nan], [9.0, 1.0, jax.numpy.nan]])
    monkeypatch.setattr(
        "flaxchat.eval.forward_model",
        lambda model, inputs: (losses, jax.numpy.zeros_like(inputs)),
    )
    fixture = REFERENCE["multiple_choice"]
    correct, record = evaluate_example_mc(
        object(), ReferenceTokenizer(), fixture["item"], fixture["fewshot"],
        fixture["delimiter"], return_record=True,
    )
    assert correct
    assert record == {"prediction": 1, "gold": 1, "scores": [2.0, 1.0]}


class TestForwardModel:
    def test_output_shapes(self, tiny_model, tiny_config):
        B, T = 2, tiny_config.sequence_len
        input_ids = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        losses, predictions = forward_model(tiny_model, input_ids)
        assert losses.shape == (B, T)
        assert predictions.shape == (B, T)

    def test_last_position_nan(self, tiny_model, tiny_config):
        B, T = 1, tiny_config.sequence_len
        input_ids = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        losses, _ = forward_model(tiny_model, input_ids)
        assert jnp.isnan(losses[0, -1])

    def test_losses_positive(self, tiny_model, tiny_config):
        B, T = 1, tiny_config.sequence_len
        input_ids = jax.random.randint(jax.random.key(0), (B, T), 0, tiny_config.vocab_size)
        losses, _ = forward_model(tiny_model, input_ids)
        # All losses except last should be positive
        assert jnp.all(losses[0, :-1] > 0)


class TestFindCommonLength:
    def test_common_prefix(self):
        seqs = [[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 7, 8]]
        assert find_common_length(seqs, 'left') == 2

    def test_no_common_prefix(self):
        seqs = [[1, 2], [3, 4]]
        assert find_common_length(seqs, 'left') == 0

    def test_common_suffix(self):
        seqs = [[1, 2, 3], [4, 2, 3], [5, 2, 3]]
        assert find_common_length(seqs, 'right') == 2

    def test_identical_sequences(self):
        seqs = [[1, 2, 3], [1, 2, 3]]
        assert find_common_length(seqs, 'left') == 3


class TestCoreProtocol:
    class Tokenizer:
        def get_vocab_size(self):
            return 32

    @staticmethod
    def _items(count):
        return [
            {'question': f'q{index}', 'choices': ['a', 'b'], 'answer': index % 2}
            for index in range(count)
        ]

    def test_declared_fewshot_count_and_manifest(self, monkeypatch, tmp_path):
        import datasets
        import flaxchat.eval as module
        spec = {
            'task_type': 'multiple_choice', 'num_fewshot': 2,
            'continuation_delimiter': '\n', 'dataset': 'owner/task',
            'revision': 'abc123', 'split': 'validation',
            'fewshot_split': 'train', 'baseline': 0.5,
        }
        monkeypatch.setattr(module, 'CORE_TASKS', {'fixture': spec})
        monkeypatch.setattr(
            datasets, 'load_dataset',
            lambda *args, split, revision: self._items(3 if split == 'validation' else 5),
        )
        counts = []
        def evaluator(model, tokenizer, item, fewshot, delimiter, return_record=False):
            counts.append(len(fewshot))
            return True, {'prediction': item['gold'], 'gold': item['gold'], 'scores': [0.0, 1.0]}
        monkeypatch.setattr(module, 'evaluate_example_mc', evaluator)
        path = tmp_path / 'manifest.json'
        result = evaluate_core(object(), self.Tokenizer(), manifest_path=str(path))
        assert result['status'] == 'complete'
        assert result['raw_results']['fixture']['sample_count'] == 3
        assert counts == [2, 2, 2]
        saved = __import__('json').loads(path.read_text())
        assert saved['tasks']['fixture']['revision'] == 'abc123'
        assert saved['protocol_hash'] == result['protocol_hash']

    def test_task_failure_invalidates_aggregate(self, monkeypatch, tmp_path):
        import datasets
        import flaxchat.eval as module
        base = {
            'task_type': 'multiple_choice', 'num_fewshot': 0,
            'continuation_delimiter': '\n', 'revision': 'abc123',
            'split': 'validation', 'fewshot_split': 'train', 'baseline': 0.5,
        }
        monkeypatch.setattr(module, 'CORE_TASKS', {
            'good': dict(base, dataset='owner/good'),
            'bad': dict(base, dataset='owner/bad'),
        })
        def load(name, *args, split, revision):
            if name == 'owner/bad':
                raise RuntimeError('fixture failure')
            return self._items(2)
        monkeypatch.setattr(datasets, 'load_dataset', load)
        monkeypatch.setattr(
            module, 'evaluate_example_mc',
            lambda *args, **kwargs: (True, {'prediction': 0, 'gold': 0, 'scores': [0.0]}),
        )
        result = evaluate_core(
            object(), self.Tokenizer(), manifest_path=str(tmp_path / 'manifest.json')
        )
        assert result['status'] == 'incomplete'
        assert result['core_metric'] is None
        assert result['raw_results']['bad']['status'] == 'failed'


class TestRenderMC:
    def test_basic(self):
        from tasks.common import render_mc
        q = "What is 2+2?"
        letters = ('A', 'B', 'C', 'D')
        choices = ['3', '4', '5', '6']
        rendered = render_mc(q, letters, choices)
        assert "What is 2+2?" in rendered
        assert "=A" in rendered
        assert "=B" in rendered
        assert "4=B" in rendered
