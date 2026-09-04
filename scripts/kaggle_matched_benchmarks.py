"""Submit or monitor the one bundled matched-trainer Kaggle GPU kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from scripts.kaggle_tpu_tests import command, kaggle_cli, monitor_kernel, validate_revision


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "accelerators" / "kaggle" / "matched.py"
NANOCHAT_REVISION = "a445144d3905c6845fda2d3cab8e63248a70cd32"
MAXTEXT_REVISION = "87b18614b0ef6c68c4be43b4396c695645ecf055"


def render_bundle(
    kernel_id: str, revision: str, directory: Path, *, preflight: bool = False
) -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__FLAXCHAT_REVISION__": validate_revision(revision),
        "__NANOCHAT_REVISION__": NANOCHAT_REVISION,
        "__MAXTEXT_REVISION__": MAXTEXT_REVISION,
        "__MODE__": "preflight" if preflight else "gpu",
        "__JAX_REQUIREMENT__": "jax>=0.9.0" if preflight else "jax[cuda12]>=0.9.0",
    }
    for placeholder, value in replacements.items():
        source = source.replace(placeholder, value)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "matched.py").write_text(source, encoding="utf-8")
    owner, separator, slug = kernel_id.partition("/")
    if not separator or not owner or not slug:
        raise ValueError("kernel id must use owner/kernel-slug")
    metadata = {
        "id": kernel_id,
        "title": slug.replace("-", " ").title(),
        "code_file": "matched.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": str(not preflight).lower(),
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (directory / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id")
    parser.add_argument("--revision", help="exact flaxchat revision; defaults to HEAD")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="run dependency/model checks on free CPU")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "kaggle-matched")
    args = parser.parse_args()
    args.kernel_id = args.kernel_id or (
        "skywolfmo/flaxchat-matched-preflight"
        if args.preflight
        else "skywolfmo/flaxchat-matched-trainer-benchmark"
    )
    if kaggle_cli() is None:
        parser.error("Kaggle CLI not found")
    if not args.status_only:
        revision = args.revision or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        try:
            revision = validate_revision(revision)
        except ValueError as error:
            parser.error(str(error))
        with tempfile.TemporaryDirectory(prefix="flaxchat-kaggle-matched-") as directory:
            render_bundle(args.kernel_id, revision, Path(directory), preflight=args.preflight)
            command("kernels", "push", "-p", directory)
        print(f"Submitted one bundled matched benchmark at {revision}", flush=True)
        if not args.wait:
            return 0
    return monitor_kernel(
        args.kernel_id,
        args.output_dir,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
