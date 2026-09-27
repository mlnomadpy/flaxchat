"""Compare the released encoder with a frozen Transformers FP32 oracle.

Export and verify run in separate processes, avoiding simultaneous framework
model copies. This checks implementation fidelity, not downstream quality.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flaxchat.encoder_data import file_hash


TEXTS = (
    "The multilingual model reads documents and answers questions about language.",
    "Le modèle multilingue lit des documents et comprend plusieurs langues.",
    "Das mehrsprachige Modell verarbeitet Dokumente in verschiedenen Sprachen.",
    "El modelo multilingüe comprende documentos y preguntas en varios idiomas.",
    "النموذج متعدد اللغات يقرأ الوثائق ويفهم الأسئلة بلغات مختلفة.",
    "多语言模型能够阅读文档并理解不同语言的问题。",
    "बहुभाषी मॉडल दस्तावेज़ पढ़ता है और विभिन्न भाषाओं में प्रश्न समझता है।",
    "Mfumo wa lugha nyingi unasoma nyaraka na kuelewa maswali katika lugha tofauti.",
)


def compare_arrays(reference, candidate, *, atol, rtol):
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    if (
        a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite([atol, rtol]).all()
        or atol < 0
        or rtol < 0
    ):
        raise ValueError(
            "Require matching nonempty finite arrays and nonnegative tolerances"
        )
    difference = a - b
    return dict(
        passed=bool(np.all(np.abs(difference) <= atol + rtol * np.abs(a))),
        max_absolute_error=float(np.abs(difference).max()),
        relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(a), 1e-30)),
        atol=atol,
        rtol=rtol,
        elements=a.size,
    )


def export(snapshot, output):
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoModelForMaskedLM
    from scripts.train_encoder import pretrained_inventory

    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    identity = pretrained_inventory(snapshot)
    tokenizer_path = Path(snapshot) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    torch.set_num_threads(4)
    model = AutoModelForMaskedLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    arrays = {}
    for length in (32, 128, 256):
        tokenizer.enable_truncation(max_length=length)
        tokenizer.enable_padding(length=length, pad_id=model.config.pad_token_id)
        # Alternate short padded rows and repeated text spanning local-attention windows.
        encodings = tokenizer.encode_batch(
            [text * (1 if i % 2 else 24) for i, text in enumerate(TEXTS)]
        )
        ids = np.asarray([e.ids for e in encodings], dtype=np.int32)
        mask = ids != model.config.pad_token_id
        ids[:, 1] = model.config.mask_token_id
        positions = np.stack(
            [np.ones(len(ids), dtype=np.int32), mask.sum(axis=1) // 2], axis=1
        )
        with torch.no_grad():
            hidden = model.model(
                input_ids=torch.from_numpy(ids.astype(np.int64)),
                attention_mask=torch.from_numpy(mask),
            ).last_hidden_state
            selected = hidden[
                torch.arange(len(ids))[:, None], torch.from_numpy(positions)
            ]
            logits = model.decoder(model.head(selected)).numpy()
            pooled = (hidden * torch.from_numpy(mask)[..., None]).sum(
                1
            ) / torch.from_numpy(mask.sum(1))[:, None]
        for name, value in dict(
            ids=ids,
            mask=mask,
            positions=positions,
            hidden=hidden.numpy(),
            pooled=pooled.numpy(),
            logits=logits,
        ).items():
            arrays[f"{length}_{name}"] = value
    np.savez(output / "oracle.npz", **arrays)
    manifest = dict(
        format="flaxchat-encoder-reference-v1",
        weights=identity,
        tokenizer_sha256=file_hash(tokenizer_path),
        config_sha256=file_hash(Path(snapshot) / "config.json"),
        oracle_sha256=file_hash(output / "oracle.npz"),
        lengths=[32, 128, 256],
        languages=["en", "fr", "de", "es", "ar", "zh", "hi", "sw"],
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        attention="eager",
        dtype="float32",
        export_script_sha256=file_hash(__file__),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(snapshot, oracle, output):
    import jax
    from flax import nnx
    from flaxchat.encoder import EncoderConfig, ModernBert
    from scripts.train_encoder import load_pretrained, pretrained_inventory

    root = Path(oracle)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["format"] != "flaxchat-encoder-reference-v1"
        or file_hash(root / "oracle.npz") != manifest["oracle_sha256"]
        or pretrained_inventory(snapshot) != manifest["weights"]
        or file_hash(Path(snapshot) / "config.json") != manifest["config_sha256"]
        or file_hash(Path(snapshot) / "tokenizer.json") != manifest["tokenizer_sha256"]
    ):
        raise ValueError("Reference snapshot or oracle identity mismatch")
    config = EncoderConfig.from_hf(
        json.loads((Path(snapshot) / "config.json").read_text()),
        compute_dtype="float32",
        residual_dtype="float32",
        use_remat=False,
    )
    model = ModernBert(config, rngs=nnx.Rngs(0))
    load_pretrained(model, snapshot)
    encode = nnx.jit(lambda m, x: m.encode(x))
    project = nnx.jit(lambda m, x: m.project(x))
    data = np.load(root / "oracle.npz", allow_pickle=False)
    checks = []
    for length in manifest["lengths"]:
        ids, mask, positions = (
            data[f"{length}_{key}"] for key in ("ids", "mask", "positions")
        )
        hidden = np.asarray(encode(model, ids))
        selected = hidden[np.arange(len(ids))[:, None], positions]
        actual = dict(
            hidden=hidden[mask],
            pooled=(hidden * mask[..., None]).sum(1) / mask.sum(1)[:, None],
            logits=np.asarray(project(model, selected)),
        )
        for name, value in actual.items():
            expected = data[f"{length}_{name}"]
            if name == "hidden":
                expected = expected[mask]
            checks.append(
                dict(
                    length=length,
                    tensor=name,
                    **compare_arrays(expected, value, atol=0.002, rtol=0.001),
                )
            )
    report = dict(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        scope="released-weight FP32 encoder implementation fidelity",
        production_quality_qualified=False,
        reference_manifest_sha256=file_hash(root / "manifest.json"),
        backend=jax.default_backend(),
        source_python_sha256=hashlib.sha256(
            Path("flaxchat/encoder.py").read_bytes()
        ).hexdigest(),
    )
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["export", "verify"], required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    if a.mode == "verify" and not a.output:
        p.error("--output required for verify")
    report = (
        export(a.snapshot, a.oracle)
        if a.mode == "export"
        else verify(a.snapshot, a.oracle, a.output)
    )
    print(json.dumps(report), flush=True)
    return int(report.get("passed") is False)


if __name__ == "__main__":
    raise SystemExit(main())
