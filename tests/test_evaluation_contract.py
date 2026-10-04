"""Receipt metadata tests: never import an accelerator or execute a model."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluation_contract import POLICY, aggregate
from scripts.ci_scope import select_scope
from scripts.merge_yat_mteb_shards import merge


def receipt():
    return {
        "identity": {"protocol": POLICY},
        "inventory": {
            "a": {
                "category": "Retrieval",
                "dataset_revision": "a" * 40,
                "eval_splits": ["test"],
                "hf_subsets": ["en", "fr"],
                "metric": "main_score",
                "expected_rows": [["test", "en"], ["test", "fr"]],
                "aggregation_policy": POLICY,
            }
        },
        "selected_tasks": ["a"],
        "results": {
            "a": {
                "category": "Retrieval",
                "result": {
                    "task_name": "a",
                    "dataset_revision": "a" * 40,
                    "scores": {
                        "test": [
                            {"hf_subset": "en", "main_score": 0.2},
                            {"hf_subset": "fr", "main_score": 0.4},
                        ]
                    },
                },
            }
        },
        "failures": {},
    }


class EvaluationContractTests(unittest.TestCase):
    def test_inventory_rejects_floating_and_shrunken_coverage(self):
        for revision in (None, "main", "v1"):
            data = receipt()
            data["inventory"]["a"]["dataset_revision"] = revision
            with self.assertRaises(ValueError):
                aggregate(data)
        data = receipt()
        data["inventory"]["a"]["expected_rows"].pop()
        data["results"]["a"]["result"]["scores"]["test"].pop()
        with self.assertRaises(ValueError):
            aggregate(data)

    def test_runtime_environment_changes_identity(self):
        import os
        from unittest.mock import patch
        from scripts.evaluation_contract import provenance

        with patch.dict(os.environ, {"XLA_FLAGS": "first"}):
            before = provenance([], devices=[{"kind": "TPU"}], batch_size=8)
        with patch.dict(os.environ, {"XLA_FLAGS": "second"}):
            after = provenance([], devices=[{"kind": "TPU"}], batch_size=8)
        self.assertNotEqual(before, after)
        self.assertTrue(before["interpreter"]["executable_sha256"])

    def test_valid_macro(self):
        result = aggregate(receipt())
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(result["mean_category_score"], 0.3)

    def test_nonfinite_empty_missing_duplicate_extra_rows(self):
        for change in (
            "nan",
            "inf",
            "empty",
            "missing",
            "duplicate",
            "extra",
            "split",
            "revision",
            "category",
        ):
            with self.subTest(change=change):
                data = receipt()
                raw = data["results"]["a"]["result"]
                rows = raw["scores"]["test"]
                if change in {"nan", "inf"}:
                    rows[0]["main_score"] = float(change)
                elif change == "empty":
                    raw["scores"] = {}
                elif change == "missing":
                    rows.pop()
                elif change == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif change == "extra":
                    rows.append({"hf_subset": "de", "main_score": 0.5})
                elif change == "split":
                    raw["scores"]["other"] = rows
                elif change == "revision":
                    raw["dataset_revision"] = "drift"
                elif change == "category":
                    data["results"]["a"]["category"] = "STS"
                with self.assertRaises(ValueError):
                    aggregate(data)

    def test_partial_is_not_full(self):
        data = receipt()
        data["inventory"]["b"] = copy.deepcopy(data["inventory"]["a"])
        result = aggregate(data)
        self.assertFalse(result["complete"])
        self.assertTrue(result["selected_complete"])

    def test_merge_rejects_invalid_rows_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps(receipt()))
            invalid = receipt()
            invalid["results"]["a"]["result"]["scores"]["test"] = []
            second.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                merge([first, second], root / "out.json")
            invalid = receipt()
            invalid["identity"] = {"protocol": "changed"}
            second.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                merge([first, second], root / "out.json")

    def test_module_deadline_stays_inside_campaign(self):
        import os
        import sys
        from unittest.mock import patch
        from scripts import validate_test_suite as suite

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests/test_fixture.py").write_text("")
            bounds = []

            def bounded(command, log, timeout, environment):
                bounds.append(timeout)
                log.write_text("metadata simulation")
                if "-c" not in command:
                    xml = next(
                        value.split("=", 1)[1]
                        for value in command
                        if value.startswith("--junitxml=")
                    )
                    Path(xml).write_text(
                        '<testsuite><testcase classname="fixture" name="checked"/></testsuite>'
                    )
                return 0

            previous = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "suite",
                            "--output",
                            str(root / "out"),
                            "--prefix",
                            "gs://mock/evidence",
                            "--timeout-seconds",
                            "1800",
                            "--module-timeout-seconds",
                            "1500",
                        ],
                    ),
                    patch.object(suite, "bounded", side_effect=bounded),
                    patch.object(suite.subprocess, "run"),
                ):
                    self.assertEqual(suite.main(), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(bounds, [120, 1500])
            result = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(result["module_timeout_seconds"], 1500)

    def test_physical_inventory_rejects_mixed_skip_and_missing_nodes(self):
        from scripts.validate_test_suite import validate_inventory

        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "tests.xml"
            xml.write_text(
                '<testsuite><testcase classname="model" name="pass"/><testcase classname="model" name="skip"><skipped/></testcase></testsuite>'
            )
            self.assertFalse(validate_inventory(xml, 0)[1])
            xml.write_text(
                '<testsuite><testcase classname="model" name="pass"/></testsuite>'
            )
            self.assertFalse(validate_inventory(xml, 0, ["model::missing"])[1])
            self.assertTrue(validate_inventory(xml, 0, ["model::pass"])[1])

    def test_embedding_preflight_and_trainer_routes(self):
        for path in (
            "scripts/preflight_yat_embedding_stage.py",
            "scripts/train_yat_embedding_finetune.py",
        ):
            scope = select_scope([path])
            self.assertIn("tests/metadata/test_embedding_stage.py", scope["tests"])
            self.assertIn(
                "tests/test_embedding_trainer_physical_tpu.py",
                scope["manual_physical_tests"],
            )

    def test_port_only_route(self):
        scope = select_scope(["torch_port/yat_encoder.py"])
        self.assertIn("tests/test_release_contract.py", scope["tests"])
        self.assertIn("tests/test_evaluation_contract.py", scope["tests"])


if __name__ == "__main__":
    unittest.main()
