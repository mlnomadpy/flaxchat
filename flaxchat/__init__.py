"""
flaxchat - A minimal end-to-end LLM training harness for TPUs

Port of nanochat (PyTorch/GPU) to JAX/Flax NNX for TPU pods,
designed for distributed training on TPU pods and GPUs.

Quick Start:
    import flaxchat

    engine = flaxchat.initialize({
        "depth": 12,
        "mode": "pretrain",
    })
    engine.train(train_loader)
"""

__version__ = "0.1.1"

# Keep metadata/cloud tooling usable before the numerical runtime is installed.
# Public training symbols preserve their existing import paths, loaded on demand.
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import (LOGICAL_AXIS_RULES, compute_init, get_mesh, setup_mesh,
                         shard_model_params, shard_batch_logical)
    from .config import FlaxChatConfig
    from .engine import Engine, generate, generate_with_cache, generate_fast, generate_speculative
    from .eval import evaluate_core, evaluate_bpb
    from .execution import execute_code, ExecutionResult
    from .gpt import GPT, GPTConfig
    from .prefetch import BackgroundPrefetcher

_EXPORTS = {
    "FlaxChatConfig": "config", "GPT": "gpt", "GPTConfig": "gpt",
    **{name: "engine" for name in ("Engine", "generate", "generate_with_cache", "generate_fast", "generate_speculative")},
    "evaluate_core": "eval", "evaluate_bpb": "eval",
    "execute_code": "execution", "ExecutionResult": "execution",
    **{name: "common" for name in ("compute_init", "get_mesh", "setup_mesh", "LOGICAL_AXIS_RULES", "shard_model_params", "shard_batch_logical")},
    "BackgroundPrefetcher": "prefetch",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "FlaxChatConfig", "GPT", "GPTConfig",
    "Engine", "generate", "generate_with_cache", "generate_fast", "generate_speculative",
    "evaluate_core", "evaluate_bpb",
    "execute_code", "ExecutionResult",
    "compute_init", "get_mesh", "setup_mesh",
    "LOGICAL_AXIS_RULES", "shard_model_params", "shard_batch_logical",
    "BackgroundPrefetcher",
    "__version__",
]
