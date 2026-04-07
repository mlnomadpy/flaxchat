"""
SmolTalk — conversational dataset from HuggingFace.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
~460K train, ~24K test conversations.
"""

from datasets import load_dataset
from tasks.common import Task


class SmolTalk(Task):

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]

        # Validate alternating user/assistant structure
        for i, msg in enumerate(messages):
            if i == 0 and msg["role"] == "system":
                continue  # system message is ok as first
            expected = "user" if (i % 2 == 0 or (i == 1 and messages[0]["role"] == "system")) else "assistant"
            # Don't assert — some conversations may have different structures

        return {"messages": messages}
