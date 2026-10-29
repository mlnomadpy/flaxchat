"""Physical-TPU-only acceptance. Never falls back to CPU model execution."""

from dataclasses import asdict, replace
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.checkpoint import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_model_from_checkpoint,
    CheckpointCompatibilityError,
)
from flaxchat.public_encoder import export_state_safetensors, load_public_encoder


@pytest.fixture(autouse=True)
def require_tpu():
    if jax.default_backend() != "tpu":
        pytest.fail("Physical TPU required: this suite must not run models on CPU")


def config(**kwargs):
    return EncoderConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=5,
        num_attention_heads=4,
        local_attention=4,
        max_position_embeddings=16,
        loop_core_start=1,
        loop_core_layers=3,
        loop_steps=3,
        compute_dtype="bfloat16",
        residual_dtype="float32",
        attention_score="goat_input",
        ffn_type="yat_glu",
        **kwargs,
    )


def inputs():
    return jnp.array([[1, 4, 6, 7, 4, 0], [1, 4, 8, 9, 4, 0]], jnp.int32)


def reference(model, ids, segments, positions, *, steps=None, strength=0.0):
    """Independent eager recurrence over the same parameter objects."""
    c = model.config
    x = model.embedding_norm(
        model.embedding(ids).astype(getattr(jnp, c.residual_dtype))
    )
    for i in range(c.loop_core_start):
        x = model.layers[i](x, segments, positions)
    first = None
    for _ in range(c.loop_steps if steps is None else steps):
        for i in range(c.loop_core_start, c.loop_core_start + c.loop_core_layers):
            x = model.layers[i](x, segments, positions)
        if first is None:
            first = x
    x = x + strength * (x - first)
    for i in range(c.loop_core_start + c.loop_core_layers, c.num_hidden_layers):
        x = model.layers[i](x, segments, positions)
    return jnp.where((segments >= 0)[..., None], model.final_norm(x), 0)


@pytest.mark.parametrize("remat", [False, True])
@pytest.mark.parametrize("geometry", ["mixed", "bf16"])
def test_scan_matches_unrolled_outputs_and_all_parameter_gradients(remat, geometry):
    model = ModernBert(
        config(use_remat=remat, yat_compute_mode=geometry), rngs=nnx.Rngs(7)
    )
    ids = inputs()
    segments = jnp.array([[0, 0, 0, 1, 1, -1], [0, 0, 0, 0, 0, -1]], jnp.int32)
    positions = jnp.array([[0, 1, 2, 0, 1, 0], [0, 1, 2, 3, 4, 0]], jnp.int32)
    expected = reference(model, ids, segments, positions)
    actual = nnx.jit(
        lambda m: m.encode(ids, segment_ids=segments, position_ids=positions)
    )(model)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-3)
    assert np.asarray(actual)[:, -1].sum() == 0
    probe = jnp.linspace(-1, 1, actual.size).reshape(actual.shape)
    compiled = nnx.jit(
        nnx.value_and_grad(
            lambda m: (
                m.encode(ids, segment_ids=segments, position_ids=positions) * probe
            ).sum()
        )
    )
    _, grad = compiled(model)
    _, ref_grad = nnx.value_and_grad(
        lambda m: (reference(m, ids, segments, positions) * probe).sum()
    )(model)
    for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(ref_grad), strict=True):
        assert np.isfinite(np.asarray(a)).all()
        np.testing.assert_allclose(a, b, atol=2e-2, rtol=3e-2)
    # Recurrence adds no parameter copies and retains alpha as trainable state.
    single = ModernBert(replace(model.config, loop_steps=1), rngs=nnx.Rngs(7))
    assert sum(a.size for a in jax.tree.leaves(nnx.state(model))) == sum(
        a.size for a in jax.tree.leaves(nnx.state(single))
    )
    paths = [
        jax.tree_util.keystr(p)
        for p, _ in jax.tree_util.tree_flatten_with_path(grad)[0]
    ]
    assert any("yat_attention_alpha" in p for p in paths)
    assert any("yat_alpha" in p for p in paths)


def test_loopcd_hidden_logits_and_adaptive_formula():
    model = ModernBert(config(use_remat=False), rngs=nnx.Rngs(5))
    ids = inputs()
    segments = jnp.where(ids == 0, -1, 0)
    positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    expected = reference(model, ids, segments, positions, steps=2, strength=0.5)
    actual = nnx.jit(lambda m: m.encode(ids, loop_steps=2, loop_guidance_strength=0.5))(
        model
    )
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-3)
    final = model.project(model.encode(ids))
    early = model.project(model.encode(ids, loop_steps=1))
    np.testing.assert_allclose(
        model.loopcd_logits(ids, strength=0.5),
        final + 0.5 * (final - early),
        atol=2e-3,
        rtol=2e-3,
    )
    probs = jax.nn.softmax(final, axis=-1)
    top = jax.lax.top_k(probs, 2)[0]
    weight = 0.5 * (1 - top[..., 0] + top[..., 1])[..., None]
    guided = nnx.jit(lambda m: m.loopcd_logits(ids, strength=0.5, adaptive=True))(model)
    np.testing.assert_allclose(
        guided, final + weight * (final - early), atol=2e-3, rtol=2e-3
    )
    np.testing.assert_array_equal(model.loopcd_logits(ids, strength=0), final)
    np.testing.assert_allclose(
        model.encode(ids, loop_steps=1, loop_guidance_strength=5),
        model.encode(ids, loop_steps=1),
        atol=0,
        rtol=0,
    )


def test_one_loop_preserves_nonrecurrent_forward():
    c = config(use_remat=False)
    recurrent = ModernBert(replace(c, loop_steps=1), rngs=nnx.Rngs(1))
    plain = ModernBert(replace(c, loop_steps=1, loop_core_layers=0), rngs=nnx.Rngs(1))
    np.testing.assert_array_equal(recurrent.encode(inputs()), plain.encode(inputs()))


def test_mlm_optimizer_recovery_and_public_export(tmp_path):
    c = config(use_remat=True)
    model = ModernBert(c, rngs=nnx.Rngs(2))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)
    ids = inputs()
    targets = jnp.where(ids == 4, 6, -1)

    @nnx.jit
    def step(m, opt):
        loss, grad = nnx.value_and_grad(lambda candidate: candidate(ids, targets))(m)
        opt.update(m, grad)
        return loss

    loss = step(model, optimizer)
    assert np.isfinite(float(loss))
    metadata = {"model_family": "modernbert", "resolved_config": {"encoder": asdict(c)}}
    root = str(tmp_path / "checkpoints")
    manager = create_checkpoint_manager(root, async_checkpointing=False)
    try:
        save_checkpoint(
            manager,
            1,
            model,
            optimizer,
            metadata,
            training_state={"completed_steps": jnp.array([1], jnp.int32)},
        )
        manager.wait_until_finished()
    finally:
        manager.close()
    restored = ModernBert(c, rngs=nnx.Rngs(9))
    restored_opt = nnx.Optimizer(restored, optax.adamw(1e-4), wrt=nnx.Param)
    _, state = restore_model_from_checkpoint(
        restored, root, optimizer=restored_opt, load_training_state=True
    )
    assert np.asarray(state["completed_steps"]).tolist() == [1]
    step(model, optimizer)
    step(restored, restored_opt)
    for left, right in zip(
        jax.tree.leaves((nnx.state(model), nnx.state(optimizer))),
        jax.tree.leaves((nnx.state(restored), nnx.state(restored_opt))),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    wrong = ModernBert(replace(c, loop_steps=2), rngs=nnx.Rngs(9))
    with pytest.raises(CheckpointCompatibilityError, match="loop architecture"):
        restore_model_from_checkpoint(wrong, root)
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text(json.dumps(asdict(c)))
    export_state_safetensors(
        nnx.to_pure_dict(nnx.state(model)), export / "model.safetensors"
    )
    public = load_public_encoder(export)
    np.testing.assert_array_equal(public.pool(ids), model.pool(ids))
