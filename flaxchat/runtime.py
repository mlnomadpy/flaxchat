"""Effective numerical environment recorded with each training recipe."""
import importlib.metadata
import os
import platform


def runtime_identity():
    packages = {}
    for name in ('jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint', 'libtpu'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'packages': packages,
            'xla_flags': os.environ.get('XLA_FLAGS', ''),
            'jax_default_matmul_precision': os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', ''),
            'jax_enable_x64': os.environ.get('JAX_ENABLE_X64', '')}
