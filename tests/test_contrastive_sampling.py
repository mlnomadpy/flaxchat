import json

import numpy as np
import pytest

from flaxchat.contrastive_sampling import LanguagePairRows


def test_language_temperature_replays_and_reduces_imbalance(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = [{"language1": "en", "language2": "ar"} for _ in range(4)]
    rows += [{"language1": "en", "language2": "es"} for _ in range(64)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sampler = LanguagePairRows(source, len(rows), seed=7, temperature=.5)
    assert sampler.languages == ("en:ar", "en:es")
    assert sampler.probabilities[0] == pytest.approx(.2)
    first = sampler.batch(12, 8)
    assert np.array_equal(first, sampler.batch(12, 8))
    assert len(np.unique(first)) == len(first)
    assert all(0 <= index < len(rows) for index in first)
    assert any(not np.array_equal(first, sampler.batch(step, 8)) for step in range(13, 17))


def test_language_temperature_rejects_mismatched_inventory(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text('{"language1":"en","language2":"fr"}\n')
    with pytest.raises(ValueError, match="inventory"):
        LanguagePairRows(source, 2, seed=1, temperature=.5)
