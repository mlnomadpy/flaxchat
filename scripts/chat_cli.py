"""CLI adapter for the manifest-validated chat service."""

from __future__ import annotations

import argparse

from flaxchat.chat import (
    GenerationConfig,
    load_chat_service,
    load_chat_service_from_artifact,
)
from flaxchat.common import print0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="d12")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument(
        "--artifact-dir",
        help="artifact directory; verifies checksums and loads its manifest",
    )
    parser.add_argument("-p", "--prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-type", choices=("base", "sft", "rl"), default="sft")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.artifact_dir and (args.checkpoint_path or args.tokenizer_path):
        parser.error("--artifact-dir cannot be combined with explicit checkpoint paths")
    service = (
        load_chat_service_from_artifact(args.artifact_dir)
        if args.artifact_dir
        else load_chat_service(
            args.model,
            args.checkpoint_type,
            checkpoint_path=args.checkpoint_path,
            tokenizer_path=args.tokenizer_path,
        )
    )
    generation = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    if args.prompt is not None:
        print(service.generate_text(args.prompt, generation))
        return 0
    print0("flaxchat CLI - type 'quit' to exit")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print0("\nBye!")
            return 0
        if user_input.lower() in ("quit", "exit", "q"):
            return 0
        if user_input:
            print(f"\nAssistant: {service.generate_text(user_input, generation).strip()}")


if __name__ == "__main__":
    raise SystemExit(main())
