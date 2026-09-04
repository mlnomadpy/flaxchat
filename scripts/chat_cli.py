"""CLI adapter for the manifest-validated chat service."""

from __future__ import annotations

import argparse

from flaxchat.chat import GenerationConfig, load_chat_service
from flaxchat.common import print0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="d12")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("-p", "--prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-type", choices=("base", "sft", "rl"), default="sft")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = load_chat_service(
        args.model,
        args.checkpoint_type,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
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
