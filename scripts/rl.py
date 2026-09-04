"""Thin CLI adapter for the RL application service."""

from __future__ import annotations

from flaxchat.stages.rl import run


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
