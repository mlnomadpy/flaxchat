"""
Dataset management for flaxchat — same dataset as nanochat (ClimbMix-400B).
"""

import os
import glob
from flaxchat.common import get_base_dir, print0, download_file_with_lock

# ClimbMix-400B dataset on HuggingFace, pinned for reproducible manifests.
DATASET_REVISION = "915333b4f8b8684f39aeaafea600fea6f43fb703"
DATASET_URL_TEMPLATE = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/"
    f"{DATASET_REVISION}/data/shard-{{shard_id:05d}}.parquet"
)
TOTAL_SHARDS = 6542


def list_parquet_files(warn_on_legacy=False):
    """List all downloaded parquet files, sorted by name."""
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "shard-*.parquet")))
    if warn_on_legacy and len(files) == 0:
        print0("WARNING: No dataset files found. Run the data download script first.")
    return files


def download_shard(shard_id: int):
    """Download a single shard from HuggingFace."""
    if not isinstance(shard_id, int) or isinstance(shard_id, bool):
        raise TypeError("shard_id must be an integer")
    if not 0 <= shard_id < TOTAL_SHARDS:
        raise ValueError(f"shard_id must be in [0, {TOTAL_SHARDS})")
    url = DATASET_URL_TEMPLATE.format(shard_id=shard_id)
    filename = os.path.join("data", f"shard-{shard_id:05d}.parquet")
    return download_file_with_lock(url, filename)


def download_shards(start: int = 0, end: int = 170):
    """Download a range of shards."""
    if not 0 <= start <= end <= TOTAL_SHARDS:
        raise ValueError(f"expected 0 <= start <= end <= {TOTAL_SHARDS}")
    for shard_id in range(start, end):
        download_shard(shard_id)
        if shard_id % 10 == 0:
            print0(f"Downloaded shard {shard_id}/{end}")
