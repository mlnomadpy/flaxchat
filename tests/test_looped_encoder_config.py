"""Model-free recurrence, lineage and inference admission checks."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from flaxchat.encoder import EncoderConfig
from scripts.train_encoder import initialization_metadata, parser, pretrained_shapes


def test_legacy_defaults_and_json_roundtrip():
    legacy = EncoderConfig()
    assert legacy.loop_core_layers == 0
    assert legacy.loop_steps == 1
    assert legacy.effective_depth == legacy.num_hidden_layers
    config = replace(
        legacy,
        num_hidden_layers=8,
        loop_core_start=2,
        loop_core_layers=3,
        loop_steps=6,
        attention_score="goat_input",
        ffn_type="yat_glu",
    )
    assert config.effective_depth == 23
    assert EncoderConfig(**json.loads(json.dumps(asdict(config)))) == config
    assert (config.yat_bias, config.yat_epsilon, config.yat_alpha_trainable) == (
        1.0,
        0.01,
        True,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"loop_steps": 0},
        {"loop_steps": -1},
        {"loop_steps": True},
        {"loop_steps": 1.5},
        {"loop_core_layers": -1},
        {"loop_core_layers": True},
        {"loop_core_start": 0},
        {"loop_core_start": -1},
        {"loop_core_start": 1.5},
        {"loop_steps": 2},
        {"loop_core_start": 21, "loop_core_layers": 2},
        {"loop_core_layers": 22},
    ],
)
def test_invalid_architecture(overrides):
    with pytest.raises(ValueError):
        EncoderConfig(**overrides)


@pytest.mark.parametrize(
    "steps,strength",
    [
        (0, 0),
        (True, 0),
        (1.5, 0),
        (2, -1),
        (2, float("nan")),
        (2, float("inf")),
        (2, True),
    ],
)
def test_invalid_inference(steps, strength):
    config = EncoderConfig(loop_core_layers=3, loop_steps=4)
    with pytest.raises(ValueError):
        config.loop_inference_options(steps, strength)


def test_inference_overrides_do_not_change_training_identity():
    config = EncoderConfig(loop_core_layers=3, loop_steps=4)
    before = asdict(config)
    assert config.loop_inference_options(2, 1.5) == (2, 1.5)
    assert config.loop_inference_options() == (4, 0)
    assert asdict(config) == before
    for steps, strength in [(2, 0), (1, 1)]:
        with pytest.raises(ValueError):
            EncoderConfig().loop_inference_options(steps, strength)


def test_cli_and_hf_migration_rejection():
    args = parser().parse_args(
        [
            "--config",
            "config.json",
            "--data",
            "data",
            "--output",
            "fresh",
            "--loop-core-start",
            "2",
            "--loop-core-layers",
            "3",
            "--loop-steps",
            "6",
        ]
    )
    assert (args.loop_core_start, args.loop_core_layers, args.loop_steps) == (2, 3, 6)
    with pytest.raises(ValueError, match="native checkpoints"):
        pretrained_shapes(EncoderConfig(loop_core_layers=3))


def test_weights_only_stage_cannot_silently_change_recurrence(monkeypatch):
    parent = EncoderConfig(loop_core_layers=3, loop_steps=4)
    metadata = {
        "model_family": "modernbert",
        "step": 100,
        "tokenizer_identity": "tokenizer",
        "resolved_config": {"encoder": asdict(parent)},
    }
    monkeypatch.setattr(
        "scripts.train_encoder.load_checkpoint_metadata", lambda *_: metadata
    )
    for child in (
        replace(parent, loop_steps=2),
        replace(parent, loop_core_layers=2),
        replace(parent, loop_core_start=2),
    ):
        with pytest.raises(ValueError, match="configuration differs"):
            initialization_metadata("checkpoint", 100, child, "tokenizer")
    _, receipt = initialization_metadata("checkpoint", 100, parent, "tokenizer")
    assert receipt["step"] == 100


@pytest.mark.parametrize(
    "parent_fields",
    [
        None,
        {},
        {"loop_core_layers": 3, "loop_steps": 2},
        {"loop_core_start": 2, "loop_core_layers": 3, "loop_steps": 4},
    ],
)
def test_direct_restore_rejects_loop_identity_before_model_mutation(
    monkeypatch, parent_fields
):
    from flaxchat import checkpoint

    closed = []
    manager = SimpleNamespace(close=lambda: closed.append(True))
    metadata = {"resolved_config": {}}
    if parent_fields is not None:
        metadata["resolved_config"]["encoder"] = parent_fields
    monkeypatch.setattr(
        checkpoint, "create_checkpoint_manager", lambda *a, **kw: manager
    )
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *a, **kw: ({}, metadata))
    model = SimpleNamespace(config=EncoderConfig(loop_core_layers=3, loop_steps=4))
    with pytest.raises(checkpoint.CheckpointCompatibilityError, match="[Ll]oop"):
        checkpoint.restore_model_from_checkpoint(model, "unused")
    assert closed == [True]


def test_example_configuration_is_valid():
    from pathlib import Path

    path = Path(__file__).parents[1] / "configs/encoder/looped_goat_input.json"
    config = EncoderConfig(**json.loads(path.read_text()))
    assert config.effective_depth == 26
    assert config.loop_core_layers % config.global_attn_every_n_layers == 0
