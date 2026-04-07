"""
Training report generation — multi-phase markdown reports.

Captures: system info, git status, training curves, eval results,
cost estimation, and final metrics tables.

Port of nanochat's report.py.
"""

import os
import json
import time
import platform
import subprocess
from datetime import datetime

import jax

from flaxchat.common import print0, get_base_dir


# GPU/TPU hourly rates (approximate, spot pricing)
DEVICE_COSTS = {
    "h100": 3.00, "a100": 1.79, "h200": 3.50,
    "l40s": 1.20, "l4": 0.50, "t4": 0.35,
    "tpu v4": 1.50, "tpu v5e": 1.20, "tpu v5p": 2.00, "tpu v6e": 2.50,
}


def _get_git_info():
    """Get current git commit, branch, and dirty status."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          stderr=subprocess.DEVNULL).decode().strip()[:8]
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                          stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                              stderr=subprocess.DEVNULL).decode().strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def _get_system_info():
    """Get system/hardware info."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": jax.device_count(),
        "local_devices": jax.local_device_count(),
        "hosts": jax.process_count(),
    }
    try:
        devices = jax.devices()
        if devices:
            info["device_kind"] = devices[0].device_kind
    except Exception:
        pass
    return info


def _estimate_cost(time_seconds, device_kind=None):
    """Estimate training cost based on device and time."""
    if device_kind is None:
        try:
            device_kind = jax.devices()[0].device_kind
        except Exception:
            return None

    kind_lower = device_kind.lower()
    for pattern, rate in DEVICE_COSTS.items():
        if pattern in kind_lower:
            hours = time_seconds / 3600
            return round(rate * hours * jax.device_count(), 2)
    return None


class Report:
    """
    Multi-phase training report.

    Usage:
        report = Report("my_run")
        report.log("Tokenizer", {"vocab_size": 32768, "time": 5.0})
        report.log("Pretrain", {"steps": 5000, "val_loss": 2.20, "time": 3000})
        report.log("SFT", {"steps": 1000, "time": 600})
        report.log("Eval", {"mmlu": 0.22, "arc": 0.275})
        report.save()
    """

    def __init__(self, run_name="default"):
        self.run_name = run_name
        self.sections = []
        self.start_time = time.time()
        self.git_info = _get_git_info()
        self.system_info = _get_system_info()

    def log(self, section: str, data):
        """Log a section with data (dict or list of dicts)."""
        if isinstance(data, dict):
            data = [data]
        self.sections.append({"section": section, "data": data, "timestamp": time.time()})
        print0(f"[Report] Logged section: {section}")

    def _render_markdown(self):
        """Render the report as markdown."""
        lines = []
        lines.append(f"# flaxchat Training Report: {self.run_name}")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # System info
        lines.append("## System")
        lines.append(f"- **Platform**: {self.system_info['platform']}")
        lines.append(f"- **Python**: {self.system_info['python']}")
        lines.append(f"- **JAX**: {self.system_info['jax']}")
        lines.append(f"- **Backend**: {self.system_info['backend']}")
        lines.append(f"- **Devices**: {self.system_info['devices']} "
                      f"({self.system_info.get('device_kind', 'unknown')})")
        lines.append(f"- **Git**: {self.git_info['commit']} ({self.git_info['branch']})"
                      f"{' [dirty]' if self.git_info['dirty'] else ''}")
        lines.append("")

        # Sections
        for sec in self.sections:
            lines.append(f"## {sec['section']}")
            for item in sec["data"]:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, float):
                            lines.append(f"- **{k}**: {v:.6f}")
                        else:
                            lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"- {item}")
            lines.append("")

        # Cost estimation
        total_time = time.time() - self.start_time
        cost = _estimate_cost(total_time)
        lines.append("## Summary")
        lines.append(f"- **Total wall time**: {total_time:.0f}s ({total_time/60:.1f}m)")
        if cost is not None:
            lines.append(f"- **Estimated cost**: ${cost:.2f}")
        lines.append("")

        return "\n".join(lines)

    def save(self, path=None):
        """Save report to markdown file."""
        if path is None:
            base_dir = get_base_dir()
            report_dir = os.path.join(base_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            path = os.path.join(report_dir, f"{self.run_name}.md")

        md = self._render_markdown()
        with open(path, "w") as f:
            f.write(md)
        print0(f"[Report] Saved to {path}")

        # Also save as JSON for programmatic access
        json_path = path.replace(".md", ".json")
        with open(json_path, "w") as f:
            json.dump({
                "run_name": self.run_name,
                "system": self.system_info,
                "git": self.git_info,
                "sections": self.sections,
                "total_time": time.time() - self.start_time,
            }, f, indent=2, default=str)

        return path

    def to_dict(self):
        return {
            "run_name": self.run_name,
            "system": self.system_info,
            "git": self.git_info,
            "sections": self.sections,
        }


# Global report singleton
_REPORT = None


def get_report(run_name="default"):
    global _REPORT
    if _REPORT is None:
        _REPORT = Report(run_name)
    return _REPORT
