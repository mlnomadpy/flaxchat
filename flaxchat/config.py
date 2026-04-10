"""
FlaxChat configuration system.

Depth-based auto-scaling config for flaxchat.
All hyperparameters auto-derive from a single "depth" dial.
"""

import math
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import jax


@dataclass(frozen=True)
class GPTConfig:
    """Model architecture config — mirrors nanochat exactly."""
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    tie_embeddings: bool = False
    use_scan: bool = False


# Register GPTConfig as a JAX pytree with all-static fields
# This lets nnx.data(GPTConfig(...)) work inside JIT
jax.tree_util.register_static(GPTConfig)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Training horizon (precedence: num_iterations > target_flops > target_param_data_ratio)
    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 12.0
    # Batch sizes
    device_batch_size: int = 32
    total_batch_size: int = -1  # -1 = auto-compute optimal
    # Learning rates (base, scaled by batch size)
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28
    # Schedule
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    # Evaluation
    eval_every: int = 250
    eval_tokens: int = 80 * 524288
    core_metric_every: int = 2000
    sample_every: int = 2000
    save_every: int = -1
    # Resume
    resume_from_step: int = -1


@dataclass
class TPUConfig:
    """TPU/distributed training config."""
    precision: str = "bf16"  # bf16 | f32
    data_parallel: int = -1  # -1 = auto
    fsdp: int = 1
    tensor_parallel: int = 1


@dataclass
class CheckpointConfig:
    """Checkpoint config (Orbax-based)."""
    dir: str = ""  # auto-derived if empty
    max_to_keep: int = 3
    async_checkpointing: bool = True


@dataclass
class LoggingConfig:
    """Logging and metrics config."""
    run_name: str = "dummy"  # "dummy" disables wandb
    wandb_project: str = "flaxchat"
    log_interval: int = 100
    use_wandb: bool = True


@dataclass
class FlaxChatConfig:
    """
    Top-level config.

    Usage:
        config = FlaxChatConfig.from_depth(depth=12)
        config = FlaxChatConfig.from_yaml("config.yaml")
        config = FlaxChatConfig.from_dict({...})
    """
    model: GPTConfig = field(default_factory=GPTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tpu: TPUConfig = field(default_factory=TPUConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_depth(
        cls,
        depth: int = 12,
        aspect_ratio: int = 64,
        head_dim: int = 128,
        max_seq_len: int = 2048,
        window_pattern: str = "SSSL",
        vocab_size: int = 32768,
        **overrides,
    ) -> "FlaxChatConfig":
        """
        Create config from a single depth dial.
        Model dim, heads, etc. auto-derive from depth.
        Matches nanochat's build_model_meta logic.
        """
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim

        model = GPTConfig(
            sequence_len=max_seq_len,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            window_pattern=window_pattern,
        )

        config = cls(model=model)

        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.tpu, key):
                setattr(config.tpu, key, value)
            elif hasattr(config.checkpoint, key):
                setattr(config.checkpoint, key, value)
            elif hasattr(config.logging, key):
                setattr(config.logging, key, value)

        return config

    @classmethod
    def from_yaml(cls, path: str) -> "FlaxChatConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "FlaxChatConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "FlaxChatConfig":
        # Also support depth= at top level for convenience
        if "depth" in data:
            return cls.from_depth(depth=data["depth"], **{
                k: v for k, v in data.items()
                if k not in ("depth", "model", "training", "tpu", "checkpoint", "logging")
            })

        # Build GPTConfig from dict (frozen dataclass — construct, don't mutate)
        model_kwargs = data.get("model", {})
        defaults = GPTConfig()
        model = GPTConfig(**{
            f.name: model_kwargs.get(f.name, getattr(defaults, f.name))
            for f in defaults.__dataclass_fields__.values()
        })

        config = cls(model=model)

        # Mutable configs can be set via setattr
        if "training" in data:
            for k, v in data["training"].items():
                setattr(config.training, k, v)
        if "tpu" in data:
            for k, v in data["tpu"].items():
                setattr(config.tpu, k, v)
        if "checkpoint" in data:
            for k, v in data["checkpoint"].items():
                setattr(config.checkpoint, k, v)
        if "logging" in data:
            for k, v in data["logging"].items():
                setattr(config.logging, k, v)
        return config

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "tpu": asdict(self.tpu),
            "checkpoint": asdict(self.checkpoint),
            "logging": asdict(self.logging),
        }
