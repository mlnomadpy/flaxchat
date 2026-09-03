"""
CLI chat interface.

Usage:
    python -m scripts.chat_cli --model=d12
    python -m scripts.chat_cli --model=d12 -p "Why is the sky blue?"
"""

import argparse

from flaxchat.chat import GenerationConfig, load_chat_service
from flaxchat.common import print0

parser = argparse.ArgumentParser(description="CLI Chat")
parser.add_argument("--model", type=str, default="d12", help="model tag")
parser.add_argument("-p", "--prompt", type=str, default=None, help="single prompt (non-interactive)")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--checkpoint-type", type=str, default="sft", choices=["base", "sft"])
args = parser.parse_args()

service = load_chat_service(args.model, args.checkpoint_type)
generation = GenerationConfig(
    max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
)

if args.prompt is not None:
    # Single prompt mode
    print(service.generate_text(args.prompt, generation))
else:
    # Interactive chat
    print0("flaxchat CLI - type 'quit' to exit")
    print0("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print0("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        print(f"\nAssistant: {service.generate_text(user_input, generation).strip()}")
