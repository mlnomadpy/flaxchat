from dataclasses import asdict
import json
from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.encoder_data import file_hash
from scripts.export_encoder_retrieval import export, parameter_digest
from scripts.evaluate_retrieval_embeddings import evaluate


def fixture(tmp_path):
    config = EncoderConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        compute_dtype="float32",
        use_remat=False,
    )
    model = ModernBert(config, rngs=nnx.Rngs(7))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    metadata = dict(
        model_family="modernbert",
        tokenizer_identity="fixture-tokenizer",
        resolved_config=dict(encoder=asdict(config), special_token_ids=[0, 1, 2, 3, 4]),
    )
    checkpoint = tmp_path / "checkpoint"
    with create_checkpoint_manager(
        str(checkpoint), async_checkpointing=False
    ) as manager:
        save_checkpoint(manager, 2, model, optimizer, metadata)
        manager.wait_until_finished()
    tokens = np.array([[2, 5, 3, 0], [2, 6, 3, 0], [2, 7, 3, 0]], np.int32)
    for name in ("queries", "corpus"):
        directory = tmp_path / name
        directory.mkdir()
        np.save(directory / "tokens.npy", tokens)
        (directory / "manifest.json").write_text(
            json.dumps(
                dict(
                    format="flaxchat-encoder-rows-v1",
                    split="validation",
                    dataset="fixture",
                    revision="v1",
                    vocab_size=16,
                    pad_token_id=0,
                    mask_token_id=4,
                    special_token_ids=[0, 1, 2, 3, 4],
                    tokenizer_sha256="fixture-tokenizer",
                    tokens_sha256=file_hash(directory / "tokens.npy"),
                )
            )
        )
    judgments = tmp_path / "judgments.json"
    judgments.write_text(
        json.dumps(
            dict(
                dataset="fixture",
                revision="v1",
                subset="fixture-eng",
                split="validation",
                tokenization_policy="one document per row",
                query_ids=["q1", "q2", "q3"],
                document_ids=["d1", "d2", "d3"],
                qrels={f"q{i}": {f"d{i}": 1} for i in range(1, 4)},
                languages={"q1": "ar", "q2": "en", "q3": "fr"},
            )
        )
    )
    return (
        model,
        tokens,
        (
            str(checkpoint),
            tmp_path / "queries",
            tmp_path / "corpus",
            judgments,
            tmp_path / "export",
        ),
    )


def test_restored_checkpoint_to_rankings_with_partial_batch(tmp_path):
    model, tokens, args = fixture(tmp_path)
    expected = np.asarray(model.pool(jnp.asarray(tokens)))
    identity = parameter_digest(model)
    result = export(*args, batch_size=2)
    assert result["checkpoint_step"] == 2
    assert result["parameter_sha256"] == identity
    assert result["encoder_config"] == asdict(model.config)
    assert result["model_identity"].startswith("sha256:")
    np.testing.assert_allclose(
        np.load(args[-1] / "queries.npy"), expected, atol=1e-6, rtol=1e-5
    )
    report = evaluate(args[-1], cutoffs=(1, 3), document_block=2)
    assert report["metrics"]["query_mean"]["recall@1"] == 1
    assert report["quality_qualified"] is False
    from scripts.evaluate_encoder_bitext import evaluate_subset

    assert evaluate_subset(args[-1])["metrics"]["f1"] == 1
    with pytest.raises(ValueError, match="existing"):
        export(*args, batch_size=2)


def test_invalid_input_identity_rejected_before_output(tmp_path):
    _, _, args = fixture(tmp_path)
    path = args[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        export(*args)
    assert not args[-1].exists()


def test_session_reuses_restore_digest_and_inference_trace(tmp_path, monkeypatch):
    import scripts.export_encoder_retrieval as exporter

    model, tokens, args = fixture(tmp_path)
    expected = np.asarray(model.pool(jnp.asarray(tokens)))
    calls = dict(restore=0, digest=0, trace=0)
    restore, digest, jit = (
        exporter.restore_model_from_checkpoint,
        exporter.parameter_digest,
        exporter.nnx.jit,
    )

    def counted_restore(*a, **kw):
        calls["restore"] += 1
        return restore(*a, **kw)

    def counted_digest(*a, **kw):
        calls["digest"] += 1
        return digest(*a, **kw)

    def counted_jit(fn):
        def traced(*a, **kw):
            calls["trace"] += 1
            return fn(*a, **kw)

        return jit(traced)

    monkeypatch.setattr(exporter, "restore_model_from_checkpoint", counted_restore)
    monkeypatch.setattr(exporter, "parameter_digest", counted_digest)
    monkeypatch.setattr(exporter.nnx, "jit", counted_jit)
    session = exporter.EmbeddingSession(args[0], step=2)
    first = session.export(*args[1:], batch_size=2)
    second_path = tmp_path / "second"
    second = session.export(*args[1:4], second_path, batch_size=2)
    assert calls == dict(restore=1, digest=1, trace=1)
    assert first["model_identity"] == second["model_identity"]
    for out in [args[-1], second_path]:
        np.testing.assert_allclose(
            np.load(out / "queries.npy"), expected, atol=1e-6, rtol=1e-5
        )


def test_released_session_exports_actual_loaded_weights(tmp_path):
    from safetensors.numpy import save_file
    from scripts.train_encoder import pretrained_shapes, load_pretrained
    from scripts.export_encoder_retrieval import EmbeddingSession

    model, tokens, args = fixture(tmp_path)
    config = model.config
    snapshot = tmp_path / "released"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(asdict(config) | {"model_type": "modernbert"})
    )
    (snapshot / "tokenizer.json").write_text(
        json.dumps(
            {"added_tokens": [{"id": i, "special": True} for i in [0, 1, 2, 3, 4]]}
        )
    )
    rng = np.random.default_rng(5)
    save_file(
        {
            k: rng.normal(size=shape).astype(np.float32)
            for k, shape in pretrained_shapes(config).items()
        },
        str(snapshot / "model.safetensors"),
    )
    for directory in args[1:3]:
        p = directory / "manifest.json"
        m = json.loads(p.read_text())
        m["tokenizer_sha256"] = file_hash(snapshot / "tokenizer.json")
        p.write_text(json.dumps(m))
    load_pretrained(model, snapshot)
    expected = np.asarray(model.pool(jnp.asarray(tokens)))
    session = EmbeddingSession.from_pretrained(snapshot, config=config)
    result = session.export(*args[1:], batch_size=2)
    assert result["checkpoint"] is None and result["checkpoint_step"] is None
    assert result["origin"]["released_weights"]["model.safetensors"] == file_hash(
        snapshot / "model.safetensors"
    )
    assert result["parameter_sha256"] == parameter_digest(model)
    np.testing.assert_allclose(
        np.load(args[-1] / "queries.npy"), expected, atol=1e-6, rtol=1e-5
    )
    from dataclasses import replace

    with pytest.raises(ValueError, match="architecture"):
        EmbeddingSession.from_pretrained(
            snapshot, config=replace(config, hidden_size=12)
        )
