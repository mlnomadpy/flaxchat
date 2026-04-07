"""
MMLU — Massive Multitask Language Understanding.
https://huggingface.co/datasets/cais/mmlu
"""

from datasets import load_dataset
from tasks.common import Task, render_mc


class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')

    def __init__(self, subset="all", split="validation", **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"]
        assert split in ["auxiliary_train", "validation", "dev", "test"]
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        subject = row["subject"]
        assert len(choices) == 4

        user_message = render_mc(question, self.letters, choices)
        return {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": self.letters[answer]},
            ],
            "subject": subject,
            "letters": self.letters,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters
        return assistant_response == conversation['messages'][-1]['content']
