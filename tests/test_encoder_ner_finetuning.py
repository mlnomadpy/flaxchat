from dataclasses import asdict
import json

import pytest

from flaxchat.encoder import EncoderConfig
from flaxchat.ner import ner_metrics
from scripts.finetune_encoder_classifier import evaluate, parser, train
from scripts.prepare_encoder_ner import prepare
from tests.test_encoder_ner_preparation import fixture


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ner_training_exact_resume_and_full_word_evaluation(tmp_path, dtype):
    source, tokenizer, _, kwargs = fixture(
        tmp_path,
        rows=[
            {"tokens": ["playing", "Tokyo"], "ner_tags": [0, 1]},
            {"tokens": ["Tokyo", "playing"], "ner_tags": [1, 0]},
            {"tokens": ["Tokyo"], "ner_tags": [1]},
            {"tokens": ["playing"], "ner_tags": [0]},
            {"tokens": ["Tokyo"], "ner_tags": [1]},
        ],
    )
    for split in ("train", "validation"):
        prepare(
            source,
            tokenizer,
            tmp_path / split,
            **(kwargs | {"split": split, "max_rows": 5}),
        )
    config = EncoderConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    common = [
        "--task",
        "token_classification",
        "--config",
        str(config_path),
        "--data",
        str(tmp_path / "train"),
        "--steps",
        "4",
        "--batch-size",
        "4",
        "--save-every",
        "2",
        "--warmup-steps",
        "1",
        "--dtype",
        dtype,
    ]
    for output, extra in [
        ("full", []),
        ("resumed", ["--stop-after", "2"]),
        ("resumed", ["--resume"]),
    ]:
        train(
            parser().parse_args(common + ["--output", str(tmp_path / output)] + extra)
        )
    manifests = [
        json.loads((tmp_path / name / "4/manifest/metadata").read_text())
        for name in ("full", "resumed")
    ]
    for key in ("model_state", "optimizer_state", "training_state"):
        assert manifests[0][key] and manifests[0][key] == manifests[1][key]
    report_path = tmp_path / "report.json"
    args = parser().parse_args(
        [
            "--mode",
            "evaluate",
            "--output",
            str(tmp_path / "full"),
            "--checkpoint-step",
            "4",
            "--eval-data",
            str(tmp_path / "validation"),
            "--batch-size",
            "4",
            "--report",
            str(report_path),
        ]
    )
    evaluate(args)
    report = json.loads(report_path.read_text())
    assert report["production_quality_qualified"] is False
    assert report["training_metadata"]["resolved_config"]["pooling"] == "none"
    assert report["metrics"]["overall"]["words"] == 7
    assert report["metrics"]["overall"]["sentences"] == 5
    dataset = report["datasets"][0]
    assert [len(r) for r in dataset["predictions"]] == [2, 2, 1, 1, 1]
    assert report["metrics"] == ner_metrics(
        dataset["references"], dataset["predictions"], ["en"] * 5
    )
    with pytest.raises(ValueError, match="new report"):
        evaluate(args)
    args.report = str(tmp_path / "invalid.json")
    args.eval_data = [str(tmp_path / "train")]
    with pytest.raises(ValueError, match="validation or test"):
        evaluate(args)
    assert not (tmp_path / "invalid.json").exists()
    args.eval_data = [str(tmp_path / "validation")] * 2
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(args)
    path = tmp_path / "validation/manifest.json"
    metadata = json.loads(path.read_text())
    metadata["label_names"] = ["O", "B-PER", "I-PER"]
    path.write_text(json.dumps(metadata))
    args.eval_data = [str(tmp_path / "validation")]
    with pytest.raises(ValueError, match="identity"):
        evaluate(args)
