from dataclasses import asdict
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flaxchat.encoder import EncoderConfig, ModernBert, import_hf_weights
from flaxchat.encoder_tasks import EncoderClassifier, classification_statistics
from scripts.prepare_encoder_classification import prepare, load_rows
from scripts.finetune_encoder_classifier import parser, train, evaluate


def fixture(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    tok = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "hello": 5,
                "world": 6,
            },
            unk_token="[UNK]",
        )
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(dict(text="hello world", text_pair="world hello", label=i % 3))
            for i in range(11)
        )
    )
    for split in ("train", "validation"):
        prepare(
            source,
            path,
            tmp_path / split,
            dataset="toy",
            revision="pinned",
            split=split,
            num_labels=3,
            sequence_length=8,
            max_rows=11,
        )
    config = EncoderConfig(
        vocab_size=7,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
    )
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    return config, source, path


def test_pair_preparation_preserves_separators_and_pinned_labels(tmp_path):
    config, _, _ = fixture(tmp_path)
    x, y, m = load_rows(tmp_path / "train", config, expected_split="train")
    assert list(x[0]) == [2, 5, 6, 3, 6, 5, 3, 0]
    assert list(y) == [i % 3 for i in range(11)]
    assert m["dataset"] == "toy" and m["revision"] == "pinned"
    with pytest.raises(ValueError, match="split"):
        load_rows(tmp_path / "train", config, expected_split="validation")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_classifier_exact_resume_and_full_development_evaluation(
    tmp_path, dtype, monkeypatch
):
    fixture(tmp_path)
    common = [
        "--config",
        str(tmp_path / "config.json"),
        "--data",
        str(tmp_path / "train"),
        "--steps",
        "5",
        "--batch-size",
        "4",
        "--save-every",
        "2",
        "--warmup-steps",
        "1",
        "--dtype",
        dtype,
    ]

    def run(name, extra=()):
        train(parser().parse_args(common + ["--output", str(tmp_path / name), *extra]))

    run("full")
    run("resumed", ["--stop-after", "2"])
    run("resumed", ["--resume"])
    run("resumed", ["--resume"])
    a = json.loads((tmp_path / "full/5/manifest/metadata").read_text())
    b = json.loads((tmp_path / "resumed/5/manifest/metadata").read_text())
    for key in ("model_state", "optimizer_state", "training_state"):
        assert a[key] == b[key] and a[key]
    with pytest.raises(ValueError, match="identity mismatch"):
        run("resumed", ["--resume", "--seed", "99"])
    report = tmp_path / "evaluation.json"
    args = parser().parse_args(
        [
            "--mode",
            "evaluate",
            "--output",
            str(tmp_path / "full"),
            "--checkpoint-step",
            "5",
            "--eval-data",
            str(tmp_path / "validation"),
            "--report",
            str(report),
            "--batch-size",
            "4",
        ]
    )
    evaluate(args)
    result = json.loads(report.read_text())
    assert result["datasets"][0]["examples"] == 11
    assert not result["production_quality_qualified"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid evaluation inputs reached model allocation")

    monkeypatch.setattr("scripts.finetune_encoder_classifier.ModernBert", forbidden)
    args.eval_data *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(args)
    args.eval_data = [str(tmp_path / "train")]
    with pytest.raises(ValueError, match="validation or test"):
        evaluate(args)


@pytest.mark.parametrize("bad", ["checksum", "range", "shape", "dtype"])
def test_corrupt_labels_rejected_before_device_training(tmp_path, bad):
    config, _, _ = fixture(tmp_path)
    from flaxchat.encoder_data import file_hash

    root = tmp_path / "train"
    p = root / "labels.npy"
    labels = np.load(p)
    if bad == "shape":
        labels = labels[:, None]
    elif bad == "dtype":
        labels = labels.astype(np.float32)
    else:
        labels[0] = 9
    np.save(p, labels)
    if bad != "checksum":
        m = json.loads((root / "manifest.json").read_text())
        m["labels_sha256"] = file_hash(p)
        (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):
        load_rows(root, config)


def test_classification_padding_does_not_change_loss_or_gradients():
    logits = jnp.array([[2.0, -1.0, 0.0], [0.0, 1.0, -1.0], [9.0, 8.0, 7.0]])
    labels = jnp.array([0, 2, -1])
    fn = jax.value_and_grad(lambda z: classification_statistics(z, labels)[0])
    loss, grad = fn(logits)
    expected = classification_statistics(logits[:2], labels[:2])[0]
    np.testing.assert_allclose(loss, expected)
    np.testing.assert_array_equal(grad[-1], 0)
    empty = classification_statistics(logits, jnp.full(3, -1))
    assert all(float(x) == 0 for x in empty)


def test_classifier_matches_transformers_logits_loss_and_gradient():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hf = transformers.ModernBertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        local_attention=2,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=160000.0,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        reference_compile=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        classifier_dropout=0.0,
        classifier_bias=False,
        classifier_pooling="mean",
        num_labels=3,
    )
    hf._attn_implementation = "eager"
    torch.manual_seed(7)
    mlm = transformers.ModernBertForMaskedLM(hf).eval()
    reference = transformers.ModernBertForSequenceClassification(hf).eval()
    reference.load_state_dict(mlm.state_dict(), strict=False)
    config = EncoderConfig.from_hf(hf.to_dict(), use_remat=False)
    encoder = ModernBert(config, rngs=nnx.Rngs(0))
    import_hf_weights(
        encoder, {k: v.detach().numpy() for k, v in mlm.state_dict().items()}
    )
    model = EncoderClassifier(encoder, 3, rngs=nnx.Rngs(1))
    model.classifier.kernel[...] = jnp.asarray(
        reference.classifier.weight.detach().numpy().T
    )
    model.classifier.bias[...] = jnp.asarray(reference.classifier.bias.detach().numpy())
    ids = np.array([[1, 5, 6, 0], [1, 7, 8, 9]], np.int32)
    labels = np.array([0, 2], np.int32)
    output = reference(
        input_ids=torch.tensor(ids, dtype=torch.long),
        attention_mask=torch.tensor(ids != 0),
        labels=torch.tensor(labels, dtype=torch.long),
    )
    output.loss.backward()
    np.testing.assert_allclose(
        np.asarray(model(ids)), output.logits.detach().numpy(), atol=2e-6, rtol=1e-5
    )
    loss, grad = nnx.value_and_grad(
        lambda m: classification_statistics(m(ids), jnp.asarray(labels))[0]
    )(model)
    np.testing.assert_allclose(loss, output.loss.item(), atol=2e-6)
    np.testing.assert_allclose(
        grad["classifier"]["kernel"][...],
        reference.classifier.weight.grad.numpy().T,
        atol=2e-6,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        grad["encoder"]["embedding"]["embedding"][...],
        reference.model.embeddings.tok_embeddings.weight.grad.numpy(),
        atol=2e-6,
        rtol=1e-4,
    )


@pytest.mark.parametrize("language", [None, "", "   ", 7])
def test_missing_language_identity_rejected(tmp_path, language):
    config, _, _ = fixture(tmp_path)
    path = tmp_path / "train/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["language"] = language
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        load_rows(path.parent, config)


@pytest.mark.integration
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_two_process_classifier_committed_fault_recovery(tmp_path, dtype):
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("FLAXCHAT_RUN_DISTRIBUTED_CPU") != "1":
        pytest.skip("Set FLAXCHAT_RUN_DISTRIBUTED_CPU=1 for localhost process tests")
    fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    common = [
        "--config",
        str(tmp_path / "config.json"),
        "--data",
        str(tmp_path / "train"),
        "--steps",
        "5",
        "--batch-size",
        "4",
        "--save-every",
        "2",
        "--warmup-steps",
        "1",
        "--dtype",
        dtype,
        "--shared-local-checkpoints",
    ]
    worker = """
import os, signal
from scripts import finetune_encoder_classifier as trainer
if os.environ['CLASSIFIER_FAULT'] == '1':
    original = trainer.save_checkpoint
    def save_then_kill(manager, step, *args, **kwargs):
        result = original(manager, step, *args, **kwargs)
        manager.wait_until_finished()
        if step == 2:
            print('FAULT_INJECTION: classifier SIGKILL after committed step 2', flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        return result
    trainer.save_checkpoint = save_then_kill
trainer.main()
"""
    for phase, output, fault, extra, expected in [
        ("baseline", "full", "0", [], 0),
        ("fault", "resumed", "1", [], -9),
        ("resume", "resumed", "0", ["--resume"], 0),
    ]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        jobs = []
        try:
            for rank in range(2):
                env = os.environ | dict(
                    JAX_PLATFORMS="cpu",
                    JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}",
                    JAX_PROCESS_COUNT="2",
                    JAX_PROCESS_INDEX=str(rank),
                    XLA_FLAGS="--xla_force_host_platform_device_count=1",
                    CLASSIFIER_FAULT=fault,
                )
                env.pop("JAX_NUM_CPU_DEVICES", None)
                path = tmp_path / f"{phase}-{rank}.log"
                log = path.open("w")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        worker,
                        *common,
                        "--output",
                        str(tmp_path / output),
                        *extra,
                    ],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                jobs.append((process, log, path))
            for process, _, path in jobs:
                assert process.wait(timeout=90) == expected, path.read_text()[-12000:]
                if fault == "1":
                    assert "SIGKILL after committed step 2" in path.read_text()
        finally:
            for process, log, _ in jobs:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                log.close()
    manifests = [
        json.loads((tmp_path / name / "5/manifest/metadata").read_text())
        for name in ("full", "resumed")
    ]
    for key in ("model_state", "optimizer_state", "training_state"):
        assert manifests[0][key] == manifests[1][key]
    events = [
        json.loads(line)
        for line in (tmp_path / "baseline-0.log").read_text().splitlines()
        if line.startswith("{")
    ]
    identity = next(e for e in events if e.get("event") == "classifier_run_config")
    assert identity["processes"] == 2 and identity["devices"] == 2


def test_local_checkpoint_override_cannot_bypass_physical_multihost_guard(monkeypatch):
    args = parser().parse_args(["--output", "/tmp/unused-classifier-guard"])
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    with pytest.raises(ValueError, match="require GCS"):
        train(args)
    args.shared_local_checkpoints = True
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    with pytest.raises(ValueError, match="CPU-test only"):
        train(args)
