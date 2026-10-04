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


from flaxchat.cost_accounting import estimate_slice_cost


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


def _estimate_cost(time_seconds, device_kind=None, *, pricing=None):
    """No rate can be inferred from logical devices. Require whole-slice evidence."""
    if pricing is None:
        return None
    return estimate_slice_cost(time_seconds, pricing)


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

    def __init__(self, run_name="default", *, pricing=None):
        self.pricing = pricing
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
        cost = _estimate_cost(total_time, pricing=self.pricing)
        lines.append("## Summary")
        lines.append(f"- **Total wall time**: {total_time:.0f}s ({total_time/60:.1f}m)")
        if cost is not None:
            lines.append(f"- **Estimated whole-slice cost**: ${cost:.2f} (planning estimate; posted charges unknown)")
        if cost is None:
            lines.append("- **Estimated cost**: unknown (verified whole-slice pricing not supplied)")
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
                "pricing": self.pricing,
                "posted_cost": None,
                "run_name": self.run_name,
                "system": self.system_info,
                "git": self.git_info,
                "sections": self.sections,
                "total_time": time.time() - self.start_time,
            }, f, indent=2, default=str)

        return path

    def to_dict(self):
        return {
            "pricing": self.pricing,
                "posted_cost": None,
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
