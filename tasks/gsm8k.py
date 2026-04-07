"""
GSM8K evaluation — Grade School Math (8K problems).
https://huggingface.co/datasets/openai/gsm8k
"""

import re
from datasets import load_dataset
from tasks.common import Task

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """Extract numerical answer after #### marker."""
    match = GSM_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


class GSM8K(Task):

    def __init__(self, subset="main", split="test", **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"]
        assert split in ["train", "test"]
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row['question']
        answer = row['answer']

        # Parse tool calls from <<expr=result>> format
        assistant_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                assistant_parts.append({"type": "python", "text": expr})
                assistant_parts.append({"type": "python_output", "text": result})
            else:
                assistant_parts.append({"type": "text", "text": part})

        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_parts},
            ]
        }

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str)
        last_text = conversation['messages'][-1]['content'][-1]['text']
        ref_num = extract_answer(last_text)
        pred_num = extract_answer(assistant_response)
        return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))
