"""Coordinator must never claim the device used by its child workloads."""
import os
from pathlib import Path
import subprocess
import sys


def test_v6_preflight_does_not_initialize_parent_jax():
    code = """
import sys
from scripts.validate_encoder_v6 import preflight_imports
preflight_imports()
assert 'jax' not in sys.modules
assert 'flaxchat' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=Path(__file__).resolve().parents[1],
                            env=os.environ | {'JAX_PLATFORMS': 'tpu'},
                            capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stderr
