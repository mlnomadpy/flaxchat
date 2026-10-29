"""Frozen argument construction for the first GOAT contrastive pilot; no backend."""

from pathlib import Path
import json


def stage_arguments(
    root, output, *, scope="production", stop=10000, resume=False, dev_rows=None
):
    root = Path(root)
    inputs = root / "inputs"
    weights = json.loads((inputs / "source-weights.json").read_text())
    if set(weights) != {
        "marco_replay",
        "code_replay",
        "bitext_replay",
        *("native_" + language for language in ["ar", "bn", "fr", "hi", "ja", "ru", "sw", "zh"]),
    }:
        raise ValueError("Frozen eleven-source mixture required")
    if scope not in ("production", "qualification") or not 1 <= stop <= 10000:
        raise ValueError("Invalid bounded stage scope/cursor")
    argv = [
        "--parent-public",
        str(root / "parent"),
        "--parent-manifest",
        str(inputs / "parent-files.json"),
        "--source-weights",
        ",".join(n + "=" + str(weights[n]) for n in sorted(weights)),
        "--output",
        str(output),
        "--steps",
        "10000",
        "--stop-after",
        str(stop),
        "--batch-size",
        "128",
        "--learning-rate",
        "3e-5",
        "--temperature",
        "0.05",
        "--warmup",
        "200",
        "--weight-decay",
        "0.01",
        "--seed",
        "1028",
        "--save-every",
        "2" if scope == "qualification" else "250",
        "--eval-every",
        "2" if scope == "qualification" else "250",
        "--keep-checkpoints",
        "3" if scope == "qualification" else "64",
        "--batch-policy",
        "homogeneous",
        "--language-exponent",
        "0.5",
        "--dev-max-rows",
        str(dev_rows or (128 if scope == "qualification" else 256)),
        "--encoder-chunk-size",
        "0",
        "--training-scope",
        scope,
        "--max-dev-regression",
        "0.02",
        "--weight-quantization",
        "none",
    ]
    if scope == "qualification":
        argv += ["--quality-regression-action", "report-only"]
    for name in sorted(weights):
        argv += ["--data", name + "=" + str(inputs / "training" / name)]
    for task in ["retrieval", "bitext", "code"]:
        argv += ["--retrieval-dev", task + "=" + str(inputs / "development" / task)]
    argv += ["--sts-dev", "sts=" + str(inputs / "development/sts")]
    if resume:
        argv += ["--resume"]
    return argv
