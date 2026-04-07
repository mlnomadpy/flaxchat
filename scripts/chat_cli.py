"""
CLI chat interface.

Usage:
    python -m scripts.chat_cli --model=d12
    python -m scripts.chat_cli --model=d12 -p "Why is the sky blue?"
"""

import argparse

from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig
from flaxchat.common import get_base_dir, print0
from flaxchat.tokenizer import get_tokenizer
from flaxchat.engine import Engine
from flaxchat.checkpoint import restore_model_from_checkpoint

parser = argparse.ArgumentParser(description="CLI Chat")
parser.add_argument("--model", type=str, default="d12", help="model tag")
parser.add_argument("-p", "--prompt", type=str, default=None, help="single prompt (non-interactive)")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--checkpoint-type", type=str, default="sft", choices=["base", "sft"])
args = parser.parse_args()

# Load tokenizer and model
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

depth = int(args.model.replace("d", ""))
config = FlaxChatConfig.from_depth(depth=depth, vocab_size=vocab_size)
model = GPT(config.model, rngs=nnx.Rngs(0))

base_dir = get_base_dir()
ckpt_dir = f"{args.checkpoint_type}_checkpoints"
checkpoint_dir = f"{base_dir}/{ckpt_dir}/{args.model}"
print0(f"Loading model from {checkpoint_dir}")
restore_model_from_checkpoint(model, checkpoint_dir)

engine = Engine(model, tokenizer)

if args.prompt is not None:
    # Single prompt mode
    text = engine.generate_text(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
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

        # Build conversation tokens
        bos = tokenizer.encode_special("<|bos|>")
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")

        tokens = [bos, user_start] + tokenizer.encode(user_input) + [user_end, assistant_start]

        all_tokens, texts = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Find assistant response in generated text
        response = texts[0]
        # Try to extract just the assistant part
        if "<|assistant_start|>" in response:
            response = response.split("<|assistant_start|>")[-1]
        if "<|assistant_end|>" in response:
            response = response.split("<|assistant_end|>")[0]

        print(f"\nAssistant: {response.strip()}")
