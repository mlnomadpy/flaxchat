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

from flaxchat.config import FlaxChatConfig
from flaxchat.gpt import GPT, GPTConfig
from flaxchat.engine import Engine, generate, generate_with_cache, generate_fast, generate_speculative
from flaxchat.eval import evaluate_core, evaluate_bpb
from flaxchat.execution import execute_code, ExecutionResult
from flaxchat.common import compute_init, get_mesh, setup_mesh

__all__ = [
    "FlaxChatConfig", "GPT", "GPTConfig",
    "Engine", "generate", "generate_with_cache", "generate_fast", "generate_speculative",
    "evaluate_core", "evaluate_bpb",
    "execute_code", "ExecutionResult",
    "compute_init", "get_mesh", "setup_mesh",
]
