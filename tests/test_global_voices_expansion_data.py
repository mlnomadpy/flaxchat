"""Data-only checks for pinned multilingual selection."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "gv_expansion", scripts / "expand_global_voices_contrastive.py"
)
expansion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expansion)


class GlobalVoicesExpansionTests(unittest.TestCase):
    def test_pinned_inventory(self):
        siblings = [
            {"rfilename": f"en-{chr(97 + i // 26)}{chr(97 + i % 26)}/train-00000-of-00001.parquet"}
            for i in range(28)
        ]
        api = {"sha": expansion.REVISION, "siblings": siblings}
        self.assertEqual(len(expansion.train_shards(api)), 28)
        with self.assertRaisesRegex(ValueError, "revision"):
            expansion.train_shards(api | {"sha": "main"})
        with self.assertRaisesRegex(ValueError, "28"):
            expansion.train_shards(api | {"siblings": siblings[:-1]})

    def test_hash_selection_and_row_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "source.parquet"
            english = [f"one sentence {i}" for i in range(20)]
            translated = [f"two sentence {i}" for i in range(20)]
            pq.write_table(pa.table({"english": english, "non_english": translated}), path)
            tok = Tokenizer(models.WordLevel({
                "[UNK]": 0, "one": 1, "two": 2, "sentence": 3
            }, unk_token="[UNK]"))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            first, stats = expansion.select_rows(path, tok, 5, 1, 10)
            second, _ = expansion.select_rows(path, tok, 5, 1, 10)
            self.assertEqual(first, second)
            self.assertEqual(stats, {"source_rows": 20, "eligible": 20, "selected": 5})
            self.assertEqual(len({row[0] for row in first}), 5)
            expected = sorted(range(20), key=lambda i: (
                hashlib.sha256((english[i] + "\0" + translated[i]).encode()).digest(), i
            ))[:5]
            self.assertEqual([row[0] for row in first], expected)

    def test_prior_stage_train_and_dev_are_sealed_together(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            files = []
            for name in ("train", "dev"):
                path = root / f"{name}.jsonl"
                path.write_text(json.dumps({
                    "sentence1": f"{name} english",
                    "sentence2": f"{name} translation",
                    "split": name,
                }) + "\n")
                files.append(path)
            dest = root / "heldout.jsonl"
            identity = expansion.combine_heldout(files, dest)
            self.assertEqual([entry["rows"] for entry in identity], [1, 1])
            self.assertEqual(len(dest.read_text().splitlines()), 2)
            self.assertEqual({json.loads(line)["sentence1"] for line in dest.read_text().splitlines()},
                             {"train english", "dev english"})


if __name__ == "__main__":
    assert "jax" not in sys.modules
    unittest.main()
