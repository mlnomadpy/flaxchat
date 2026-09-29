"""Replayable language-temperature batches for paired encoder training."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


class LanguagePairRows:
    """Sample language pairs by count**temperature, then distinct rows per batch.

    A batch is a pure function of (seed, step, source file, temperature). This
    preserves exact data replay after restoring an optimizer checkpoint.
    """

    def __init__(self, train_jsonl: Path, rows: int, seed: int, temperature: float):
        if rows < 1 or seed < 0 or not np.isfinite(temperature) or not 0 < temperature <= 1:
            raise ValueError("Invalid language-temperature sampling configuration")
        groups = defaultdict(list)
        count = 0
        with Path(train_jsonl).open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                count = index + 1
                item = json.loads(line)
                left, right = item.get("language1"), item.get("language2")
                if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
                    raise ValueError("Every training pair needs language provenance")
                groups[left + ":" + right].append(index)
        if count != rows or not groups:
            raise ValueError("Language inventory differs from token arrays")
        self.languages = tuple(sorted(groups))
        self.indices = tuple(np.asarray(groups[name], dtype=np.int64) for name in self.languages)
        counts = np.asarray([len(values) for values in self.indices], dtype=np.float64)
        weights = counts ** temperature
        self.probabilities = weights / weights.sum()
        self.seed = seed

    def batch(self, step: int, size: int) -> np.ndarray:
        if step < 0 or size < 1:
            raise ValueError("Invalid batch cursor")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        languages = rng.choice(len(self.indices), size=size, p=self.probabilities)
        result = np.empty(size, dtype=np.int64)
        for language in np.unique(languages):
            selected = np.flatnonzero(languages == language)
            pool = self.indices[int(language)]
            result[selected] = rng.choice(pool, size=len(selected), replace=len(selected) > len(pool))
        return result
