"""
SpellingBee — letter counting and spelling tasks with tool use.

Teaches the model to:
1. Count letters in words (step-by-step reasoning)
2. Use the Python calculator tool for verification
3. Handle multilingual prompts

Port of nanochat's spellingbee.py.
"""

import random
from tasks.common import Task

# Templates for asking letter-counting questions
TEMPLATES = [
    "How many times does the letter '{letter}' appear in the word '{word}'?",
    "Count the occurrences of '{letter}' in '{word}'.",
    "In the word '{word}', how many '{letter}'s are there?",
    "How many '{letter}' letters are in '{word}'?",
    "Tell me how many times '{letter}' appears in '{word}'.",
    "What is the count of '{letter}' in the word '{word}'?",
    "Can you count the letter '{letter}' in '{word}'?",
    "How often does '{letter}' occur in '{word}'?",
    # Spanish
    "Cuantas veces aparece la letra '{letter}' en la palabra '{word}'?",
    "Cuenta las '{letter}' en '{word}'.",
    # French
    "Combien de fois la lettre '{letter}' apparait dans le mot '{word}'?",
    # German
    "Wie oft kommt der Buchstabe '{letter}' im Wort '{word}' vor?",
    # Chinese-style English
    "Please count letter '{letter}' in word '{word}'",
    # Casual
    "yo how many {letter}'s in {word}",
    "quick - count the {letter} in {word}",
    # Formal
    "I would like to know the frequency of the letter '{letter}' in the word '{word}'.",
    "Could you please determine how many times '{letter}' occurs in '{word}'?",
    # Direct
    "'{letter}' count in '{word}'",
    "Number of '{letter}' in '{word}'?",
    "{word} - how many {letter}?",
]

# Common words for spelling tasks
WORDS = [
    "strawberry", "banana", "mississippi", "accommodation", "necessary",
    "occurrence", "embarrassment", "millennium", "committee", "assessment",
    "broccoli", "cappuccino", "desiccated", "fluorescent", "guarantee",
    "harassment", "independent", "lieutenant", "maintenance", "occasionally",
    "parallel", "questionnaire", "recommend", "separate", "tomorrow",
    "unnecessary", "vacuum", "wednesday", "xylophone", "zucchini",
    "abbreviation", "acknowledge", "bureaucracy", "conscientious", "deteriorate",
    "exaggerate", "fascinate", "hemorrhage", "idiosyncrasy", "juxtaposition",
    "knowledgeable", "miscellaneous", "onomatopoeia", "pharmaceutical",
    "reconnaissance", "surveillance", "temperamental", "unequivocally",
]

LETTERS = list("abcdefghijklmnopqrstuvwxyz")


class SpellingBee(Task):
    """Letter counting with tool use — generates counting + Python verification."""

    def __init__(self, size=256, split="train", **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.split = split
        self.rng = random.Random(42 if split == "train" else 123)
        self._examples = self._generate_examples()

    def _generate_examples(self):
        examples = []
        for i in range(self.size):
            rng = random.Random(42 + i if self.split == "train" else 123 + i)
            word = rng.choice(WORDS)
            letter = rng.choice([l for l in set(word)])  # pick a letter that exists

            # Maybe lowercase
            if rng.random() < 0.5:
                word_display = word
            else:
                word_display = word.upper()

            # Pick template
            template = rng.choice(TEMPLATES)
            question = template.format(letter=letter, word=word_display)

            # Maybe remove question mark
            if rng.random() < 0.5:
                question = question.rstrip("?")

            # Count the actual answer
            count = word.lower().count(letter.lower())

            # Build assistant response with step-by-step counting + tool verification
            # Manual counting
            steps = []
            for j, char in enumerate(word.lower()):
                if char == letter.lower():
                    steps.append(f"Position {j + 1}: '{char}' - match!")

            manual_text = f"Let me count the letter '{letter}' in '{word}':\n"
            manual_text += "\n".join(steps) if steps else "No matches found."
            manual_text += f"\n\nThe letter '{letter}' appears {count} time{'s' if count != 1 else ''}."

            # Tool verification
            expr = f"'{word.lower()}'.count('{letter.lower()}')"

            assistant_parts = [
                {"type": "text", "text": manual_text + "\n\nLet me verify with Python: "},
                {"type": "python", "text": expr},
                {"type": "python_output", "text": str(count)},
                {"type": "text", "text": f"\n\nConfirmed: {count}."},
            ]

            examples.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant_parts},
                ],
                "answer": count,
                "word": word,
                "letter": letter,
            })
        return examples

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        return self._examples[index]

    def evaluate(self, conversation, assistant_response):
        """Check if the response contains the correct count."""
        answer = conversation.get("answer", -1)
        # Try to find the number in the response
        import re
        numbers = re.findall(r'\b(\d+)\b', assistant_response)
        if numbers:
            # Check if any number matches
            return int(any(int(n) == answer for n in numbers))
        return 0

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))


class SimpleSpelling(Task):
    """Simple spelling task — just spell the word letter by letter."""

    def __init__(self, size=128, split="train", **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self._examples = []
        for i in range(size):
            rng = random.Random(42 + i)
            word = rng.choice(WORDS)
            spelled = " ".join(word.upper())
            self._examples.append({
                "messages": [
                    {"role": "user", "content": f"Spell the word '{word}' letter by letter."},
                    {"role": "assistant", "content": spelled},
                ]
            })

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        return self._examples[index]

    def evaluate(self, conversation, assistant_response):
        expected = conversation["messages"][-1]["content"]
        return int(expected.lower() in assistant_response.lower())
