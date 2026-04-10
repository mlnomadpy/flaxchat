"""
Common utilities for flaxchat.
TPU mesh setup, distributed helpers, logging, peak FLOPS.
"""

import os
import logging
import urllib.request
from filelock import FileLock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils


# ---------------------------------------------------------------------------
# Compute dtype (always bf16 on TPU, configurable via env)
# ---------------------------------------------------------------------------
_DTYPE_MAP = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}

def _detect_compute_dtype():
    env = os.environ.get("FLAXCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via FLAXCHAT_DTYPE={env}"
    # TPU natively supports bf16
    if jax.default_backend() == "tpu":
        return jnp.bfloat16, "auto-detected: TPU (bf16 native)"
    if jax.default_backend() == "gpu":
        return jnp.bfloat16, "auto-detected: GPU (bf16)"
    return jnp.float32, "auto-detected: CPU"

COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m',
        'WARNING': '\033[33m', 'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        return super().format(record)

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

setup_default_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Printing helpers (only process 0)
# ---------------------------------------------------------------------------
def print0(s="", **kwargs):
    if jax.process_index() == 0:
        print(s, **kwargs)

def print_banner():
    banner = r"""
      __ _                  _           _
     / _| | __ ___  ___ ___| |__   __ _| |_
    | |_| |/ _` \ \/ / / __| '_ \ / _` | __|
    |  _| | (_| |>  < | (__| | | | (_| | |_
    |_| |_|\__,_/_/\_\ \___|_| |_|\__,_|\__|
    """
    print0(banner)


# ---------------------------------------------------------------------------
# Base directory
# ---------------------------------------------------------------------------
def get_base_dir():
    if os.environ.get("FLAXCHAT_BASE_DIR"):
        base_dir = os.environ["FLAXCHAT_BASE_DIR"]
    else:
        home_dir = os.path.expanduser("~")
        base_dir = os.path.join(home_dir, ".cache", "flaxchat")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


# ---------------------------------------------------------------------------
# File download with lock (multi-process safe)
# ---------------------------------------------------------------------------
def download_file_with_lock(url, filename, postprocess_fn=None):
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")
        if postprocess_fn is not None:
            postprocess_fn(file_path)
    return file_path


# ---------------------------------------------------------------------------
# Mesh setup — creates a device mesh over ALL available devices.
# This is the core of flaxchat's parallelism. Every script uses this.
# ---------------------------------------------------------------------------

# Global mesh — set once at startup, used everywhere
_MESH: Mesh | None = None


def setup_mesh(
    data_parallel: int = -1,
    fsdp: int = 1,
    tensor_parallel: int = 1,
) -> Mesh:
    """
    Create a JAX device mesh over ALL devices and set it as the global default.

    Axis names: ('data', 'fsdp', 'tensor')
    - data: data parallelism (batch sharding across devices)
    - fsdp: fully-sharded data parallelism (param sharding)
    - tensor: tensor/model parallelism

    If data_parallel == -1, auto-compute: all devices go to data parallel.
    On multi-host TPU pods, this spans all hosts automatically.
    """
    global _MESH
    num_devices = jax.device_count()
    num_local = jax.local_device_count()
    num_hosts = jax.process_count()

    if data_parallel == -1:
        data_parallel = num_devices // (fsdp * tensor_parallel)

    assert data_parallel * fsdp * tensor_parallel == num_devices, (
        f"Mesh shape ({data_parallel}, {fsdp}, {tensor_parallel}) = "
        f"{data_parallel * fsdp * tensor_parallel} != {num_devices} devices"
    )

    devices = mesh_utils.create_device_mesh(
        (data_parallel, fsdp, tensor_parallel)
    )
    mesh = Mesh(devices, axis_names=('data', 'fsdp', 'tensor'))
    _MESH = mesh

    print0(f"Mesh: shape={mesh.shape}, devices={num_devices} "
           f"({num_local} local x {num_hosts} hosts), "
           f"backend={jax.default_backend()}")
    return mesh


def get_mesh() -> Mesh:
    """Get the global mesh. Call setup_mesh() first."""
    assert _MESH is not None, "Call setup_mesh() before using get_mesh()"
    return _MESH


def replicate_on_mesh(state, mesh: Mesh | None = None):
    """Replicate a pytree across all devices (no sharding)."""
    if mesh is None:
        mesh = get_mesh()
    replicated = NamedSharding(mesh, P())
    return jax.device_put(state, replicated)


def shard_model_fsdp(state, mesh: Mesh | None = None):
    """
    Shard model params across the 'fsdp' mesh axis.

    For FSDP: each device holds a shard of each parameter's first dimension.
    Use this for large models that don't fit in single device memory.

    Params with ndim >= 2 are sharded along dim 0 across 'fsdp'.
    Params with ndim < 2 (scalars, biases) are replicated.
    """
    if mesh is None:
        mesh = get_mesh()

    def _shard_leaf(param):
        if hasattr(param, 'ndim') and param.ndim >= 2:
            # Shard first dim across fsdp axis
            return jax.device_put(param, NamedSharding(mesh, P('fsdp')))
        else:
            # Replicate scalars/1D params
            return jax.device_put(param, NamedSharding(mesh, P()))

    return jax.tree.map(_shard_leaf, state)


def shard_on_mesh(array, mesh: Mesh | None = None):
    """Shard an array along the 'data' axis (batch dimension)."""
    if mesh is None:
        mesh = get_mesh()
    return jax.device_put(array, NamedSharding(mesh, P('data')))


def shard_batch(inputs, targets, mesh: Mesh | None = None):
    """Shard input/target arrays along the 'data' axis for data-parallel training."""
    return shard_on_mesh(inputs, mesh), shard_on_mesh(targets, mesh)


# ---------------------------------------------------------------------------
# Logical axis annotations (MaxText-style)
# Maps logical tensor axis names to mesh axis names.
# None = replicated.  For pure data-parallel the only sharded axis is batch.
# ---------------------------------------------------------------------------
LOGICAL_AXIS_RULES = [
    ("batch", "data"),
    ("embed", None),     # replicated
    ("heads", None),     # replicated
    ("mlp", None),       # replicated
    ("kv", None),        # replicated
    ("vocab", None),     # replicated
    ("length", None),    # replicated
]


def logical_to_mesh_sharding(logical_spec, mesh: Mesh | None = None):
    """Convert a logical PartitionSpec to a concrete NamedSharding using LOGICAL_AXIS_RULES."""
    if mesh is None:
        mesh = get_mesh()
    rules = dict(LOGICAL_AXIS_RULES)
    mesh_axes = tuple(rules.get(ax, None) if ax is not None else None
                      for ax in logical_spec)
    return NamedSharding(mesh, P(*mesh_axes))


def shard_model_params(model, mesh: Mesh | None = None):
    """
    Place model params on the mesh using logical axis rules.

    For data-parallel (default): all params are replicated (P()).
    The batch dimension is the only thing sharded — handled by shard_batch.
    """
    if mesh is None:
        mesh = get_mesh()
    replicated = NamedSharding(mesh, P())
    return jax.device_put(model, replicated)


def shard_batch_logical(batch, mesh: Mesh | None = None):
    """Shard a batch dict/pytree along the 'data' axis using logical annotations."""
    if mesh is None:
        mesh = get_mesh()
    batch_sharding = NamedSharding(mesh, P('data'))
    return jax.tree.map(lambda x: jax.device_put(x, batch_sharding), batch)


# ---------------------------------------------------------------------------
# Distributed init for multi-host TPU pods
# ---------------------------------------------------------------------------
def compute_init():
    """
    Initialize JAX for distributed training. Must be called BEFORE any JAX computation.

    For multi-host TPU pods:
    - Calls jax.distributed.initialize() to coordinate across hosts
    - Each host discovers its local devices and the global device mesh

    For single-host (GPU/CPU/single TPU VM):
    - No distributed init needed, just reports devices

    Returns the mesh after calling setup_mesh().
    """
    # Multi-host init (TPU pods, SLURM clusters)
    if jax.process_count() > 1 or 'JAX_COORDINATOR_ADDRESS' in os.environ:
        jax.distributed.initialize()

    print0(f"Devices: {jax.device_count()} total, "
           f"{jax.local_device_count()} local, "
           f"{jax.process_count()} hosts, "
           f"backend={jax.default_backend()}")
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    # Create mesh over ALL devices (data-parallel by default)
    mesh = setup_mesh()
    return mesh


# ---------------------------------------------------------------------------
# Peak FLOPS for TPU chips (bf16)
# ---------------------------------------------------------------------------
def get_peak_flops(device_kind: str | None = None) -> float:
    """Return peak bf16 FLOPS for a TPU chip type."""
    if device_kind is None:
        devices = jax.devices()
        if devices:
            device_kind = devices[0].device_kind
        else:
            return float('inf')

    kind = device_kind.lower()

    # TPU peak bf16 FLOPS per chip
    _TPU_PEAK_FLOPS = {
        "tpu v2":    45.5e12,
        "tpu v3":    123e12,
        "tpu v4":    275e12,
        "tpu v5 lite": 197e12,   # v5e
        "tpu v5e":   197e12,
        "tpu v5p":   459e12,
        "tpu v6e":   918e12,
        # GPU fallbacks
        "a100":      312e12,
        "h100":      989e12,
        "h200":      989e12,
    }

    for pattern, flops in _TPU_PEAK_FLOPS.items():
        if pattern in kind:
            return flops

    logger.warning(f"Peak flops undefined for: {device_kind}, MFU will show as 0%")
    return float('inf')


# ---------------------------------------------------------------------------
# Dummy wandb
# ---------------------------------------------------------------------------
class DummyWandb:
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
