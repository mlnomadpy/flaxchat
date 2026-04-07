"""
Generate synthetic training data using LLM APIs (OpenRouter).

Creates diverse identity/capability conversations with controlled
topic categories, user personas, and conversation dynamics.

Port of nanochat's gen_synthetic_data.py.

Usage:
    python -m scripts.gen_synthetic --num=100 --output=data/synthetic.jsonl
    python -m scripts.gen_synthetic --num=1000 --api-key=$OPENROUTER_KEY
"""

import os
import json
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

TOPICS = [
    "identity", "architecture", "training", "capabilities",
    "limitations", "comparisons", "history", "deep_dives",
    "philosophical", "creative", "technical", "educational",
]

PERSONAS = [
    "curious student", "experienced developer", "non-technical user",
    "researcher", "journalist", "teacher", "child", "skeptic",
    "enthusiast", "business professional", "philosopher", "artist",
]

DYNAMICS = [
    "straightforward Q&A", "deep dive follow-ups", "challenging assumptions",
    "playful and casual", "formal and precise", "confused and needing clarification",
    "comparing alternatives", "debugging a problem", "brainstorming ideas",
    "learning step by step",
]

FIRST_MESSAGES = [
    "Hey!", "Hi there", "Hello", "I have a question",
    "Can you help me?", "I'm curious about something",
    "Hola!", "Bonjour!", "Ni hao!", "Guten Tag!",
    "What's up?", "I need some help with something",
]

SYSTEM_PROMPT_TEMPLATE = """You are generating a synthetic training conversation.

Topic: {topic}
User persona: {persona}
Conversation dynamic: {dynamic}

Generate a realistic {num_turns}-turn conversation between a user and an AI assistant.
The user should ask about: {topic}
The user personality is: {persona}
The conversation style is: {dynamic}

Output as JSON with format:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
"""


def generate_conversation(api_key, topic, persona, dynamic, num_turns=3, model="anthropic/claude-3.5-haiku"):
    """Generate one synthetic conversation via OpenRouter API."""
    import urllib.request

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        topic=topic, persona=persona, dynamic=dynamic, num_turns=num_turns
    )

    first_msg = random.choice(FIRST_MESSAGES)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Start the conversation. The user's first message is: '{first_msg}'"},
        ],
        "temperature": 0.9,
        "max_tokens": 2000,
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]

        # Try to parse JSON from response
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        conv = json.loads(content)

        # Add metadata
        conv["metadata"] = {
            "topic": topic, "persona": persona, "dynamic": dynamic,
            "model": model, "generated_at": time.time(),
        }
        return conv
    except Exception as e:
        print(f"Failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--num", type=int, default=100, help="Number of conversations")
    parser.add_argument("--output", type=str, default="data/synthetic.jsonl")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--model", type=str, default="anthropic/claude-3.5-haiku")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--turns", type=int, default=3, help="Turns per conversation")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Provide --api-key or set OPENROUTER_API_KEY env var")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Generate configs
    configs = []
    rng = random.Random(42)
    for i in range(args.num):
        configs.append({
            "topic": rng.choice(TOPICS),
            "persona": rng.choice(PERSONAS),
            "dynamic": rng.choice(DYNAMICS),
            "num_turns": args.turns,
        })

    print(f"Generating {args.num} conversations with {args.workers} workers...")
    results = []

    def gen_one(cfg):
        return generate_conversation(api_key, model=args.model, **cfg)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(gen_one, cfg) for cfg in configs]
        for i, future in enumerate(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{args.num} done ({len(results)} successful)")

    # Save
    with open(args.output, "w") as f:
        for conv in results:
            f.write(json.dumps(conv) + "\n")

    print(f"\nSaved {len(results)} conversations to {args.output}")

    # Stats
    topic_counts = {}
    for r in results:
        t = r.get("metadata", {}).get("topic", "unknown")
        topic_counts[t] = topic_counts.get(t, 0) + 1
    print("Topic distribution:")
    for t, c in sorted(topic_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
