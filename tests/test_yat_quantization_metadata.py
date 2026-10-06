"""Data-export tests only: no model construction or forward execution."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from scripts.quantize_yat_encoder import export, quantize_rows


def _config_class_without_numerical_imports():
    # Execute the actual metadata-only class definition without importing Torch,
    # Safetensors/Torch, constructing a model, or initializing any backend.
    source = Path(__file__).resolve().parents[1] / "torch_port/yat_encoder.py"
    parsed = ast.parse(source.read_text())
    node = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
                and node.name == "YatEncoderConfig")
    namespace = {"dataclass": dataclass, "json": json, "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["YatEncoderConfig"]


class QuantizationFormatTests(unittest.TestCase):
    def test_torch_config_rejects_qat_masters_and_unknown_policies(self):
        config_class = _config_class_without_numerical_imports()
        config = {name: 1 for name in config_class.__dataclass_fields__}
        config.update(yat_bias=1, yat_epsilon=.01, attention_score="yat_softmax",
            ffn_type="yat_glu", compute_dtype="bfloat16", residual_dtype="float32",
            yat_compute_mode="bf16", yat_ffn_compute_mode="bf16_adaptive",
            yat_attention_implementation="centered_fp32_scores")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            config_class.from_json(path)
            config["weight_quantization"] = "int8_per_channel_ste"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "QAT FP32 masters"):
                config_class.from_json(path)
            config_class.from_json(path, compressed_int8=True)
            config["weight_quantization"] = "ternary_unqualified"
            path.write_text(json.dumps(config))
            for compressed in (False, True):
                with self.assertRaisesRegex(ValueError, "Unsupported source"):
                    config_class.from_json(path, compressed_int8=compressed)

    def test_signed_rows_zero_and_chunk_invariance(self):
        rows = np.array([[0, 0, 0], [-3, 1, 2], [0.1, -0.2, 0.3]], dtype=np.float32)
        q, scale, error = quantize_rows(rows, chunk_rows=1)
        q2, scale2, _ = quantize_rows(rows, chunk_rows=3)
        np.testing.assert_array_equal(q, q2)
        np.testing.assert_array_equal(scale, scale2)
        np.testing.assert_array_equal(q[0], 0)
        self.assertEqual(scale[0, 0], 1)
        self.assertGreaterEqual(int(q.min()), -127)
        self.assertLessEqual(int(q.max()), 127)
        self.assertTrue(
            np.all(np.abs(rows - q.astype(np.float32) * scale) <= scale / 2 + 1e-7)
        )
        self.assertGreater(error, 0)

    def test_reject_nonfinite_and_wrong_dtype(self):
        for array in (
            np.array([[np.nan]], dtype=np.float32),
            np.array([[np.inf]], dtype=np.float32),
            np.ones((2, 2), dtype=np.float64),
        ):
            with self.assertRaises(ValueError):
                quantize_rows(array)

    def test_real_integer_storage_and_preserved_scalar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "out"
            source.mkdir()
            save_file(
                {
                    "embedding.weight": np.ones((8, 16), dtype=np.float32),
                    "layers.0.yat_alpha": np.array(1.57, dtype=np.float32),
                },
                source / "model.safetensors",
            )
            (source / "config.json").write_text(
                json.dumps({"yat_bias": 1, "yat_epsilon": 0.01,
                            "weight_quantization": "int8_per_channel_ste"})
            )
            (source / "tokenizer.json").write_text("{}")
            sha = hashlib.sha256(
                (source / "model.safetensors").read_bytes()
            ).hexdigest()
            receipt = export(source, output, expected_sha256=sha)
            with safe_open(output / "model.safetensors", framework="np") as f:
                self.assertEqual(f.get_tensor("embedding.qweight").dtype, np.int8)
                self.assertEqual(f.get_tensor("embedding.scale").dtype, np.float32)
                self.assertEqual(f.get_tensor("layers.0.yat_alpha"), np.float32(1.57))
            self.assertFalse(receipt["physical_tpu_validated"])
            self.assertEqual(receipt["source_weight_quantization"], "int8_per_channel_ste")
            self.assertEqual(receipt["source_weight_storage"], "fp32-masters")
            self.assertEqual(receipt["weight_storage"], "int8-matrices-with-fp32-scales-and-unquantized-vectors")
            with self.assertRaises(FileExistsError):
                export(source, output, expected_sha256=sha)
            with self.assertRaises(ValueError):
                export(source, root / "bad", expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
