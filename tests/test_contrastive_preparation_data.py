"""Pure-data regressions; executable with Python, without JAX or model imports."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "contrastive_prep",
    Path(__file__).parents[1] / "scripts/prepare_encoder_contrastive.py",
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def fixture(self, root):
        training = [
            dict(
                sentence1=f"question {i}",
                sentence2=f"answer {i}",
                language1="en",
                language2="fr",
            )
            for i in range(100)
        ]
        training += [
            dict(
                sentence1=" QUESTION 0 ",
                sentence2="followup",
                language1="en",
                language2="fr",
            ),
            dict(
                sentence1="answer 0",
                sentence2="question 0",
                language1="fr",
                language2="en",
            ),
            dict(
                sentence1=" SEALED TEXT ",
                sentence2="leak",
                language1="en",
                language2="fr",
            ),
        ]
        for name, rows, split in [
            ("source", training, "train"),
            (
                "sealed",
                [dict(sentence1="sealed text", sentence2="secret translation")],
                "test",
            ),
        ]:
            p = root / (name + ".jsonl")
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / (name + ".json")).write_text(
                json.dumps(
                    dict(
                        files=[
                            dict(
                                path=p.name,
                                sha256=prep.file_hash(p),
                                dataset=name,
                                revision="a" * 40,
                                license="test-fixture",
                                split=split,
                            )
                        ]
                    )
                )
            )
        return dict(
            source_manifest=root / "source.json",
            sealed_manifest=root / "sealed.json",
            source_manifest_sha256=prep.file_hash(root / "source.json"),
            sealed_manifest_sha256=prep.file_hash(root / "sealed.json"),
            dev_fraction=0.2,
        )

    def test_separation_leakage_components_replay(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            args = self.fixture(r)
            a = prep.prepare(output=r / "a", **args)
            prep.prepare(output=r / "b", **args)
            self.assertEqual(
                (r / "a/manifest.json").read_bytes(),
                (r / "b/manifest.json").read_bytes(),
            )
            prep.verify(r / "a")
            self.assertEqual(a["excluded"]["sealed_text_overlap"], 1)
            self.assertEqual(a["excluded"]["duplicate_or_reversed_pair"], 1)
            linked = []
            for split in ("train", "dev"):
                for _, row in prep.rows(r / "a" / f"{split}.jsonl"):
                    self.assertNotIn(prep.text_hash("sealed text"), row["text_hashes"])
                    if prep.text_hash("question 0") in row["text_hashes"]:
                        linked.append(split)
            self.assertEqual(len(linked), 2)
            self.assertEqual(len(set(linked)), 1)
            with (r / "a/train.jsonl").open("a") as f:
                f.write("{}\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                prep.verify(r / "a")

    def test_pin_and_split_rejection(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            args = self.fixture(r)
            (r / "source.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prep.prepare(output=r / "a", **args)
            self.assertFalse((r / "a").exists())
            args = self.fixture(r)
            m = json.loads((r / "source.json").read_text())
            m["files"][0]["split"] = "test"
            (r / "source.json").write_text(json.dumps(m))
            args["source_manifest_sha256"] = prep.file_hash(r / "source.json")
            with self.assertRaisesRegex(ValueError, "split"):
                prep.prepare(output=r / "a", **args)

    def test_tokenized_identity_and_overlength(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        import numpy as np

        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            args = self.fixture(r)
            tok = Tokenizer(
                models.WordLevel(
                    {"[PAD]": 0, "[UNK]": 1, "question": 2, "answer": 3},
                    unk_token="[UNK]",
                )
            )
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tok.save(str(r / "tokenizer.json"))
            extra = dict(
                tokenizer=r / "tokenizer.json",
                tokenizer_sha256=prep.file_hash(r / "tokenizer.json"),
                sequence_length=8,
            )
            prep.prepare(output=r / "a", **args, **extra)
            arrays, m = prep.load_tokenized(
                r / "a", r / "tokenizer.json", split="train", sequence_length=8
            )
            self.assertEqual(
                arrays["query_tokens"].shape, (m["files"]["train.jsonl"]["rows"], 8)
            )
            self.assertEqual(arrays["positive_group_ids"].dtype, np.int32)
            self.assertEqual(
                len(set(arrays["positive_group_ids"].tolist())),
                len(arrays["positive_group_ids"]),
            )
            with self.assertRaisesRegex(ValueError, "sequence length"):
                prep.load_tokenized(
                    r / "a", r / "tokenizer.json", split="train", sequence_length=9
                )
            with self.assertRaisesRegex(ValueError, "overlength"):
                prep.prepare(
                    output=r / "bad", **args, **(extra | dict(sequence_length=1))
                )
            self.assertFalse((r / "bad").exists())

    def test_normalization(self):
        self.assertEqual(prep.text_hash(" Ａ  B "), prep.text_hash("a b"))


if __name__ == "__main__":
    assert not any(
        k == "jax" or k.startswith("jax.") or k == "flaxchat" for k in sys.modules
    )
    unittest.main()
