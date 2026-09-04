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

        return {"messages": messages}
