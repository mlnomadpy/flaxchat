"""Model-free admission and metric verification, no JAX imports/execution."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "flaxchat" / f"{name}.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


contract, quality = module("embedding_contract"), module("embedding_quality")


class AdmissionTests(unittest.TestCase):
    def test_original_cursor_acceptance_and_actual_no_work_event(self):
        state = {
            "completed_steps": np.array([7], np.int32),
            "optimizer_updates": np.array([7], np.int32),
        }
        with self.assertRaisesRegex(ValueError, "stop precedes"):
            contract.validate_cursor(state, horizon=9, stop=6, committed_step=7)
        with self.assertRaisesRegex(ValueError, "disagree"):
            contract.validate_cursor(
                state | {"optimizer_updates": np.array([6], np.int32)},
                horizon=9,
                stop=9,
                committed_step=7,
            )
        start = contract.validate_cursor(state, horizon=9, stop=7, committed_step=7)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(contract.emit_completion_status(start, 7))
            self.assertFalse(contract.emit_completion_status(start, 9))
        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [{"event": "embedding_already_completed", "step": 7, "requested_stop": 7}],
        )

    def test_restored_optimizer_counter_must_match_authenticated_cursor(self):
        contract.validate_optimizer_cursor(np.array(7, np.int32), 7)
        for value in (
            np.array(6, np.int32),
            np.array([7], np.int32),
            np.array(7.0, np.float32),
            np.array(True),
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "optimizer counter"),
            ):
                contract.validate_optimizer_cursor(value, 7)

    def test_cursor_requires_committed_step(self):
        state = {
            "completed_steps": np.array([7], np.int32),
            "optimizer_updates": np.array([7], np.int32),
        }
        self.assertEqual(
            contract.validate_cursor(state, horizon=9, stop=7, committed_step=7), 7
        )
        for stop, step in [(6, 7), (9, 8)]:
            with self.assertRaises(ValueError):
                contract.validate_cursor(
                    state, horizon=9, stop=stop, committed_step=step
                )
        state["completed_steps"] = np.array(7, np.int32)
        with self.assertRaises(ValueError):
            contract.validate_cursor(state, horizon=9, stop=9, committed_step=7)

    def test_semantic_parent_rejected(self):
        config = {"yat_bias": 1, "yat_epsilon": 0.01}
        metadata = {
            "step": 2,
            "resolved_config": {"encoder": config},
            "tokenizer_identity": "abc",
        }
        contract.validate_parent_metadata(metadata, config, "abc")
        with self.assertRaises(ValueError):
            contract.validate_parent_metadata(metadata, config, "different")
        with self.assertRaises(ValueError):
            contract.validate_parent_metadata(
                metadata, {**config, "yat_epsilon": 0.02}, "abc"
            )

    def test_retrieval_duplicate_relevance_and_regression(self):
        q = np.array([[1.0, 0], [1.0, 0], [0, 1.0]])
        ids = np.array([1, 1, 2])
        docs = np.array([4, 4, 5])
        groups = np.array([7, 7, 8])
        result = quality.retrieval_metrics(q, q, ids, docs, groups)
        self.assertEqual(result["unique_documents"], 2)
        self.assertEqual(result["mrr"], 1.0)
        self.assertTrue(
            quality.quality_gate(
                {"code": result}, {"code": result}, max_regression=0.02
            )["passed"]
        )
        worse = {**result, "mrr": 0.5}
        self.assertFalse(
            quality.quality_gate(
                {"code": worse}, {"code": result}, max_regression=0.02
            )["passed"]
        )
        with self.assertRaises(ValueError):
            quality.retrieval_metrics(q * np.nan, q, ids, docs, groups)
        with self.assertRaises(ValueError):
            quality.quality_gate({}, {}, max_regression=0.02)

    def test_gate_rejects_malformed_baseline_and_uses_source_macro(self):
        good = {"mrr": 0.8, "recall_at_1": 0.7, "recall_at_10": 0.9}
        bad = {**good, "mrr": float("nan")}
        with self.assertRaises(ValueError):
            quality.quality_gate({"text": good}, {"text": bad}, max_regression=0.02)
        with self.assertRaises(ValueError):
            quality.quality_gate(
                {"text": good}, {"text": good}, max_regression=float("nan")
            )
        values = {
            "text": good,
            "code": {**good, "mrr": 0.4},
            "text/language/1": {**good, "mrr": 0.2},
        }
        self.assertAlmostEqual(
            quality.quality_gate(values, values, max_regression=0.02)["score"], 0.6
        )

    def test_padding_vocab_shape_dtype_rejected(self):
        encoder = {"pad_token_id": 0, "vocab_size": 8, "max_position_embeddings": 8}
        manifest = {
            "pad_id": 0,
            "vocab_size": 8,
            "query_length": 2,
            "document_length": 2,
            "rows": {"train": 2},
            "tokenization": {"special_token_ids": [0, 4]},
        }
        arrays = {
            "train": {
                key: np.ones((2, 2), np.int32)
                for key in ("query_tokens", "positive_tokens", "negative_tokens")
            }
        }
        arrays["train"]["query_text_ids"] = np.array([1, 2], np.int32)
        arrays["train"]["negative_valid"] = np.array([True, False])
        contract.validate_arrays(arrays, manifest, encoder)
        for bad in ({**manifest, "pad_id": 1}, {**manifest, "vocab_size": 9}):
            with self.assertRaises(ValueError):
                contract.validate_arrays(arrays, bad, encoder)
        arrays["train"]["query_tokens"][0, 0] = 8
        with self.assertRaises(ValueError):
            contract.validate_arrays(arrays, manifest, encoder)


if __name__ == "__main__":
    unittest.main()
