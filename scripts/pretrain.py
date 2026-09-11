"""Thin CLI adapter for the pretraining application service."""

from __future__ import annotations

from flaxchat.stages.pretrain import PretrainRequest, build_parser, run


def main(argv: list[str] | None = None) -> int:
    request = PretrainRequest.from_namespace(build_parser().parse_args(argv))
    return run(request).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
