"""
Base class for all Tasks.
A Task is a dataset of conversations with evaluation criteria.
Port of nanochat's tasks/common.py.
"""

import random


class Task:
    """Base class. Supports lightweight slicing over a dataset."""

    def __init__(self, start=0, stop=None, step=1):
        assert start >= 0
        assert stop is None or stop >= start
        assert step >= 1
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self):
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        stop = self.num_examples() if self.stop is None else self.stop
        span = stop - self.start
        return (span + self.step - 1) // self.step

    def __getitem__(self, index: int):
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """Train on a mixture of datasets with deterministic shuffling."""

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(t) for t in tasks]
        self.num_conversations = sum(self.lengths)
        self.index_map = []
        for task_idx, length in enumerate(self.lengths):
            for local_idx in range(length):
                self.index_map.append((task_idx, local_idx))
        rng = random.Random(42)
        rng.shuffle(self.index_map)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """Sequentially train on a list of tasks (curriculum)."""

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(t) for t in tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range"
        for task_idx, length in enumerate(self.lengths):
            if index < length:
                return self.tasks[task_idx][index]
            index -= length


def render_mc(question, letters, choices):
    """Render a multiple-choice question. Letter AFTER choice for better binding."""
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query
