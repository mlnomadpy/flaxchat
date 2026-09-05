"""Run deterministic inference from a manifest-bound published artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from flaxchat.chat import GenerationConfig, load_chat_service_from_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("examples/tinystories-v0.1.1"),
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args(argv)
    service = load_chat_service_from_artifact(args.artifact_dir)
    print(service.generate_text(
        args.prompt,
        GenerationConfig(max_tokens=args.max_tokens, temperature=0, seed=42),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
