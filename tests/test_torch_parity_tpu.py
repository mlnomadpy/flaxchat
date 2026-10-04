"""Physical TPU parity acceptance, isolated JAX/TorchXLA environments.

Set YAT_PARITY_SOURCE, YAT_PARITY_TARGET, YAT_JAX_PYTHON and
YAT_TORCH_PYTHON to run. Missing physical inputs are visible skips, never passes.
"""

import os
from pathlib import Path
import unittest


class PhysicalTorchParityTests(unittest.TestCase):
    def test_release_matrix(self):
        variables = (
            "YAT_PARITY_SOURCE",
            "YAT_PARITY_TARGET",
            "YAT_JAX_PYTHON",
            "YAT_TORCH_PYTHON",
            "YAT_PARITY_EVIDENCE",
            "YAT_PARITY_HOST_RAM_BUDGET_BYTES",
            "YAT_PARITY_SCRATCH_BUDGET_BYTES",
            "YAT_PARITY_EVIDENCE_BUDGET_BYTES",
        )
        missing = [name for name in variables if not os.environ.get(name)]
        if missing:
            self.skipTest(
                "Physical TPU release artifacts/environments not configured: "
                + ", ".join(missing)
            )
        source, target, jax_python, torch_python, evidence = (
            os.environ[name] for name in variables[:5]
        )
        target = Path(target)
        from scripts.run_yat_parity_campaign import run_campaign

        summary = run_campaign(
            source,
            target,
            jax_python,
            torch_python,
            evidence,
            int(os.environ.get("YAT_PARITY_CAMPAIGN_SECONDS", "1700")),
            int(os.environ.get("YAT_PARITY_CASE_SECONDS", "1200")),
            int(os.environ.get("YAT_PARITY_MAX_CASES", "1")),
            host_ram_budget_bytes=int(os.environ[variables[5]]),
            scratch_budget_bytes=int(os.environ[variables[6]]),
            evidence_budget_bytes=int(os.environ[variables[7]]),
        )
        self.assertTrue(
            summary["full_matrix_qualified"],
            "Partial durable campaign; rerun under a new bounded lease to schedule missing cases",
        )


if __name__ == "__main__":
    unittest.main()
