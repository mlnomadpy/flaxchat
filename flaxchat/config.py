"""
FlaxChat configuration system.

Depth-based auto-scaling config for flaxchat.
All hyperparameters auto-derive from a single "depth" dial.
"""

import yaml
import json
from dataclasses import dataclass, field, asdict, fields
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
    attention_backend: str = "auto"  # auto | xla | splash

    def __post_init__(self):
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_kv_head <= 0:
            raise ValueError("n_layer, n_head, and n_kv_head must be positive")
        if self.n_embd <= 0 or self.n_embd % self.n_head:
            raise ValueError("n_embd must be positive and divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head for GQA")
        if not self.window_pattern or any(c not in "SLsl" for c in self.window_pattern):
            raise ValueError("window_pattern must be a non-empty string containing only S/L")
        if self.attention_backend not in {"auto", "xla", "splash"}:
            raise ValueError("attention_backend must be one of: auto, xla, splash")


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
    gradient_accumulation_dtype: str = "float32"
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

    def __post_init__(self):
        if self.precision not in {"bf16", "f32"}:
            raise ValueError("precision must be one of: bf16, f32")
        if self.data_parallel == 0 or self.data_parallel < -1:
            raise ValueError("data_parallel must be -1 or a positive integer")
        if self.fsdp <= 0 or self.tensor_parallel <= 0:
            raise ValueError("fsdp and tensor_parallel must be positive")


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

    _SECTIONS = ("model", "training", "tpu", "checkpoint", "logging")

    @staticmethod
    def _field_names(instance) -> set[str]:
        return {item.name for item in fields(instance)}

    @classmethod
    def _reject_unknown(cls, values: dict, instance, section: str):
        unknown = set(values) - cls._field_names(instance)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown {section} configuration field(s): {names}")

    def validate(self) -> "FlaxChatConfig":
        """Validate cross-field and mutable-section invariants."""
        training = self.training
        if training.device_batch_size <= 0:
            raise ValueError("device_batch_size must be positive")
        if training.total_batch_size != -1 and training.total_batch_size <= 0:
            raise ValueError("total_batch_size must be -1 or positive")
        if training.num_iterations != -1 and training.num_iterations <= 0:
            raise ValueError("num_iterations must be -1 or positive")
        if not 0 <= training.warmdown_ratio <= 1:
            raise ValueError("warmdown_ratio must be between 0 and 1")
        if not 0 <= training.final_lr_frac <= 1:
            raise ValueError("final_lr_frac must be between 0 and 1")
        if training.warmup_steps < 0 or training.eval_every < 0:
            raise ValueError("warmup_steps and eval_every must be non-negative")
        if training.gradient_accumulation_dtype not in {"float32", "bfloat16"}:
            raise ValueError("gradient_accumulation_dtype must be float32 or bfloat16")
        if self.tpu.precision not in {"bf16", "f32"}:
            raise ValueError("precision must be one of: bf16, f32")
        if self.tpu.data_parallel == 0 or self.tpu.data_parallel < -1:
            raise ValueError("data_parallel must be -1 or a positive integer")
        if self.tpu.fsdp <= 0 or self.tpu.tensor_parallel <= 0:
            raise ValueError("fsdp and tensor_parallel must be positive")
        if self.checkpoint.max_to_keep <= 0:
            raise ValueError("max_to_keep must be positive")
        if self.logging.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        return self

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

        # Apply any overrides. Typos must fail rather than silently changing a run.
        for key, value in overrides.items():
            matched = False
            if hasattr(config.training, key):
                setattr(config.training, key, value)
                matched = True
            elif hasattr(config.tpu, key):
                setattr(config.tpu, key, value)
                matched = True
            elif hasattr(config.checkpoint, key):
                setattr(config.checkpoint, key, value)
                matched = True
            elif hasattr(config.logging, key):
                setattr(config.logging, key, value)
                matched = True
            if not matched:
                raise ValueError(f"Unknown configuration override: {key}")

        return config.validate()

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
        if not isinstance(data, dict):
            raise TypeError("Configuration root must be a mapping")
        unknown_sections = set(data) - set(cls._SECTIONS) - {"depth"}
        if unknown_sections:
            names = ", ".join(sorted(unknown_sections))
            raise ValueError(f"Unknown top-level configuration field(s): {names}")

        # Also support depth= at top level for convenience
        if "depth" in data:
            if data.get("model"):
                raise ValueError("Specify either depth or model fields, not both")
            config = cls.from_depth(depth=data["depth"])
        else:
            model_kwargs = data.get("model", {})
            if not isinstance(model_kwargs, dict):
                raise TypeError("model configuration must be a mapping")
            defaults = GPTConfig()
            cls._reject_unknown(model_kwargs, defaults, "model")
            model = GPTConfig(**{
                f.name: model_kwargs.get(f.name, getattr(defaults, f.name))
                for f in fields(defaults)
            })
            config = cls(model=model)

        for section in ("training", "tpu", "checkpoint", "logging"):
            values = data.get(section, {})
            if not isinstance(values, dict):
                raise TypeError(f"{section} configuration must be a mapping")
            target = getattr(config, section)
            cls._reject_unknown(values, target, section)
            for key, value in values.items():
                setattr(target, key, value)
        return config.validate()

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "tpu": asdict(self.tpu),
            "checkpoint": asdict(self.checkpoint),
            "logging": asdict(self.logging),
        }
