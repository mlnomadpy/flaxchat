import importlib
import json

import jax
import jax.numpy as jnp
import pytest

from flaxchat.rl import centered_advantages, preference_loss
from flaxchat.sft import load_conversations, make_sft_batch
from flaxchat.config import FlaxChatConfig
from flaxchat.stages import StageResult
from flaxchat.stages.eval import EvalRequest, build_parser as build_eval_parser
from flaxchat.stages.pretrain import (
    PretrainRequest,
    build_parser as build_pretrain_parser,
)
from flaxchat.stages.rl import RLRequest, build_parser as build_rl_parser
from flaxchat.stages.sft import SFTRequest, build_parser as build_sft_parser


class ConversationTokenizer:
    def render_conversation(self, conversation, max_tokens):
        del conversation
        ids = [1, 2, 3, 4, 5][:max_tokens]
        return ids, [0, 0, 0, 1, 1][:max_tokens]


def test_sft_batch_supervises_only_assistant_tokens():
    inputs, targets = make_sft_batch(
        [{"messages": []}], ConversationTokenizer(), 2, 4, jax.random.key(0)
    )
    assert inputs.shape == targets.shape == (2, 4)
    assert jnp.array_equal(targets[0], jnp.asarray([-1, -1, 4, 5]))


def test_sft_batch_fails_without_examples():
    with pytest.raises(ValueError, match="at least one"):
        make_sft_batch([], ConversationTokenizer(), 1, 4, jax.random.key(0))


def test_sft_batch_rejects_examples_without_supervised_tokens():
    class NoTargets(ConversationTokenizer):
        def render_conversation(self, conversation, max_tokens):
            del conversation, max_tokens
            return [1, 2, 3], [0, 0, 0]

    with pytest.raises(ValueError, match="no supervised"):
        make_sft_batch([{"messages": []}], NoTargets(), 1, 4, jax.random.key(0))


def test_load_conversations_from_jsonl_obeys_limit(tmp_path):
    path = tmp_path / "conversations.jsonl"
    path.write_text("\n".join(json.dumps({"messages": [index]}) for index in range(3)))
    assert load_conversations(str(path), limit=2) == [
        {"messages": [0]}, {"messages": [1]}
    ]


def test_preference_objective_is_finite_and_centered():
    logits = jnp.zeros((2, 3, 8))
    targets = jnp.asarray([[1, 2, -1], [1, 3, 4]])
    advantages = centered_advantages([1.0, 0.0])
    assert jnp.array_equal(advantages, jnp.asarray([0.5, -0.5]))
    assert jnp.isfinite(preference_loss(logits, targets, advantages))


def test_cli_modules_are_import_safe_and_expose_main(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("import attempted to parse arguments")

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", forbidden)
    modules = (
        "scripts.pretrain", "scripts.sft", "scripts.rl", "scripts.eval",
        "scripts.chat_cli", "scripts.chat_web", "scripts.train_kaggle",
        "scripts.train_tpu", "scripts.train_local", "scripts.train_gpt2",
        "scripts.tok_train", "scripts.pretokenize_data",
        "scripts.convert_to_tflite",
    )
    for name in modules:
        assert callable(importlib.reload(importlib.import_module(name)).main)


@pytest.mark.parametrize(
    ("request_type", "parser", "argv", "field", "expected"),
    (
        (PretrainRequest, build_pretrain_parser, ["--depth", "3"], "depth", 3),
        (SFTRequest, build_sft_parser, ["--batch-size", "2"], "batch_size", 2),
        (RLRequest, build_rl_parser, ["--num-samples", "4"], "num_samples", 4),
        (EvalRequest, build_eval_parser, ["--tasks", "core,mmlu"], "tasks", "core,mmlu"),
    ),
)
def test_stage_cli_options_resolve_to_typed_requests(
    request_type, parser, argv, field, expected
):
    request = request_type.from_namespace(parser().parse_args(argv))
    assert isinstance(request, request_type)
    assert getattr(request, field) == expected


def test_stage_result_is_machine_readable():
    result = StageResult(
        stage="eval",
        resolved_config={"model": {"n_layer": 1}},
        metrics={"loss": 1.0},
        artifact_paths=("manifest.json",),
    )
    assert result.exit_code == 0
    assert result.metrics["loss"] == 1.0


def test_all_stage_requests_accept_one_validated_resolved_config():
    config = FlaxChatConfig.from_depth(
        depth=1, aspect_ratio=16, head_dim=16, max_seq_len=16, vocab_size=256
    )
    for request_type in (PretrainRequest, SFTRequest, RLRequest, EvalRequest):
        request = request_type(resolved_config=config)
        assert request.resolved_config is config
