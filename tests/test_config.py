"""
Tests for the configuration system.
"""

import os
import json
import tempfile

import pytest

from flaxchat.config import FlaxChatConfig, GPTConfig, TrainingConfig, TPUConfig


class TestGPTConfig:
    def test_defaults(self):
        config = GPTConfig()
        assert config.sequence_len == 2048
        assert config.vocab_size == 32768
        assert config.n_layer == 12
        assert config.n_embd == 768
        assert config.window_pattern == "SSSL"


class TestFlaxChatConfig:
    def test_defaults(self):
        config = FlaxChatConfig()
        assert isinstance(config.model, GPTConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.tpu, TPUConfig)

    def test_from_depth_12(self):
        config = FlaxChatConfig.from_depth(depth=12, aspect_ratio=64, head_dim=128)
        assert config.model.n_layer == 12
        assert config.model.n_embd == 768
        assert config.model.n_head == 6  # 768 / 128
        assert config.model.n_kv_head == 6

    def test_from_depth_24(self):
        config = FlaxChatConfig.from_depth(depth=24, aspect_ratio=64, head_dim=128)
        assert config.model.n_layer == 24
        assert config.model.n_embd == 1536
        assert config.model.n_head == 12

    def test_from_depth_with_overrides(self):
        config = FlaxChatConfig.from_depth(
            depth=12,
            device_batch_size=8,
            warmup_steps=100,
        )
        assert config.training.device_batch_size == 8
        assert config.training.warmup_steps == 100

    def test_from_dict(self):
        data = {
            "model": {"n_layer": 6, "n_embd": 384},
            "training": {"device_batch_size": 4},
        }
        config = FlaxChatConfig.from_dict(data)
        assert config.model.n_layer == 6
        assert config.model.n_embd == 384
        assert config.training.device_batch_size == 4

    def test_from_dict_with_depth(self):
        data = {"depth": 8}
        config = FlaxChatConfig.from_dict(data)
        assert config.model.n_layer == 8

    def test_from_yaml(self):
        import yaml
        data = {
            "model": {"n_layer": 4, "n_embd": 256, "vocab_size": 512},
            "training": {"num_iterations": 100},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            yaml_path = f.name

        try:
            config = FlaxChatConfig.from_yaml(yaml_path)
            assert config.model.n_layer == 4
            assert config.training.num_iterations == 100
        finally:
            os.unlink(yaml_path)

    def test_from_json(self):
        data = {
            "model": {"n_layer": 4},
            "tpu": {"precision": "f32"},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            json_path = f.name

        try:
            config = FlaxChatConfig.from_json(json_path)
            assert config.model.n_layer == 4
            assert config.tpu.precision == "f32"
        finally:
            os.unlink(json_path)

    def test_to_dict(self):
        config = FlaxChatConfig.from_depth(depth=12)
        d = config.to_dict()
        assert "model" in d
        assert "training" in d
        assert "tpu" in d
        assert d["model"]["n_layer"] == 12

    def test_roundtrip(self):
        config = FlaxChatConfig.from_depth(depth=12)
        d = config.to_dict()
        config2 = FlaxChatConfig.from_dict(d)
        assert config.model.n_layer == config2.model.n_layer
        assert config.model.n_embd == config2.model.n_embd
