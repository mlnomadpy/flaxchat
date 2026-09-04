"""Thin CLI adapter for the evaluation application service."""

from __future__ import annotations

from flaxchat.stages.eval import EvalRequest, build_parser, run


def main(argv: list[str] | None = None) -> int:
    request = EvalRequest.from_namespace(build_parser().parse_args(argv))
    return run(request).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
