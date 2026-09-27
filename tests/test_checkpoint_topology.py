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
@pytest.mark.parametrize("writer_devices,reader_devices", [(8, 1), (1, 8)])
def test_checkpoint_restores_across_device_counts(tmp_path, writer_devices, reader_devices):
    checkpoint = tmp_path / "portable"
    writer = _run("save", checkpoint, writer_devices)
    reader = _run("restore", checkpoint, reader_devices)
    assert writer == {"mode": "save", "device_count": writer_devices, "shard_count": writer_devices, "optimizer_updates": 7}
    assert reader == {"mode": "restore", "device_count": reader_devices, "shard_count": reader_devices, "optimizer_updates": 8}
