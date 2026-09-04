"""Thin CLI adapter for the SFT application service."""

from __future__ import annotations

from flaxchat.stages.sft import SFTRequest, build_parser, run


def main(argv: list[str] | None = None) -> int:
    request = SFTRequest.from_namespace(build_parser().parse_args(argv))
    return run(request).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
