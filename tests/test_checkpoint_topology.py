"""Process-isolated checkpoint topology portability tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _run(mode, checkpoint_dir, devices):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.checkpoint_portability",
            mode,
            str(checkpoint_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.integration
def test_checkpoint_restores_from_eight_devices_to_one(tmp_path):
    checkpoint = tmp_path / "portable"
    writer = _run("save", checkpoint, 8)
    reader = _run("restore", checkpoint, 1)
    assert writer == {"mode": "save", "device_count": 8, "shard_count": 8}
    assert reader == {"mode": "restore", "device_count": 1, "shard_count": 1}
