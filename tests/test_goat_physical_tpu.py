"""GOAT numerical acceptance: physical TPU only, never CPU emulation."""
import os
import math
from dataclasses import replace
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('FLAXCHAT_PHYSICAL_TPU') != '1',
                                reason='Physical TPU admission required')


@pytest.mark.parametrize('radius', [None, 0, 1, 3])
@pytest.mark.parametrize('architecture', ['goat', 'goat_input'])
@pytest.mark.parametrize('mode,implementation', [('mixed', 'standard'), ('bf16', 'standard'),
                                               ('bf16', 'centered_fp32_scores')])
def test_goat_forward_backward_and_empty_rows(radius, mode, implementation, architecture):
    import jax
    import jax.numpy as jnp
    from flaxchat.encoder import bidirectional_attention, _rope
    assert jax.default_backend() == 'tpu'
    dtype = jnp.float32 if mode == 'mixed' else jnp.bfloat16
    v = (jax.random.normal(jax.random.key(19), (1, 7, 2, 4)) * .2).astype(dtype)
    segments = jnp.array([[0, 0, 0, 0, 1, -1, -1]])
    positions = jnp.arange(7)[None]
    def values(x):
        # For input GOAT this fixed linear value projection is independent of
        # the geometry; differentiating x must still include both pathways.
        return jnp.roll(x, 1, axis=-1) * .7 if architecture == 'goat_input' else x
    def actual(v, alpha):
        z = _rope(v, positions, 160000.)
        return bidirectional_attention(z, z, values(v), segments, radius=radius, score=architecture,
                                       alpha=alpha, yat_compute_mode=mode, yat_attention_block_size=3,
                                       yat_attention_implementation=implementation)
    def reference(v, alpha):
        z = _rope(v, positions, 160000.).astype(jnp.float32).transpose(0, 2, 1, 3)
        dot = jnp.einsum('bhid,bhjd->bhij', z, z, precision=jax.lax.Precision.HIGHEST)
        distance = ((z[:, :, :, None] - z[:, :, None, :]) ** 2).sum(-1)
        idx = jnp.arange(7)
        mask = (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] >= 0)
        mask &= (idx[:, None] != idx[None, :])[None]
        if radius is not None:
            mask &= (abs(idx[:, None] - idx[None, :]) <= radius)[None]
        live = mask.any(-1)
        logits = alpha * (dot + 1)**2 / (distance + .01)
        logits = jnp.where(mask[:, None], logits, -jnp.inf)
        logits = jnp.where(live[:, None, :, None], logits, 0.)
        weights = jnp.where(live[:, None, :, None], jax.nn.softmax(logits, -1), 0.)
        return jnp.einsum('bhij,bjhd->bihd', weights, values(v).astype(jnp.float32),
                          precision=jax.lax.Precision.HIGHEST)
    alpha = jnp.float32(.1)
    output = jax.jit(actual)(v, alpha)
    assert np.isfinite(output.astype(jnp.float32)).all()
    np.testing.assert_array_equal(np.asarray(output[:, 4:], dtype=np.float32), 0)
    tolerance = .04 if mode == 'bf16' else 2e-5
    np.testing.assert_allclose(output.astype(jnp.float32), reference(v, alpha), atol=tolerance, rtol=tolerance)
    actual_grads = jax.jit(jax.grad(lambda v, a: (actual(v, a).astype(jnp.float32)**2).sum(), (0, 1)))(v, alpha)
    ref_grads = jax.grad(lambda v, a: (reference(v, a)**2).sum(), (0, 1))(v, alpha)
    for a, b in zip(actual_grads, ref_grads, strict=True):
        assert np.isfinite(a.astype(jnp.float32)).all()
        np.testing.assert_allclose(a.astype(jnp.float32), b.astype(jnp.float32), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('architecture', ['goat', 'goat_input'])
def test_projection_removal_and_explicit_migration(architecture):
    import jax
    from flax import nnx
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.goat import migrate_yat_to_goat
    assert jax.default_backend() == 'tpu'
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
                           num_hidden_layers=2, num_attention_heads=2, attention_score='yat_softmax')
    source = ModernBert(config, rngs=nnx.Rngs(1))
    target = ModernBert(replace(config, attention_score=architecture), rngs=nnx.Rngs(2))
    migrate_yat_to_goat(source, target)
    for a, b in zip(source.layers, target.layers, strict=True):
        assert not hasattr(b, 'qkv')
        np.testing.assert_array_equal(b.v.kernel[...], a.qkv.kernel[..., 16:])
    def count(m):
        return sum(x.size for x in jax.tree.leaves(nnx.state(m, nnx.Param)))
    assert count(source) - count(target) == 2 * config.num_hidden_layers * config.hidden_size**2


@pytest.mark.parametrize('index', [0, 1])
def test_input_goat_geometry_is_independent_of_value_projection(index, monkeypatch):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import flaxchat.encoder as encoder
    assert jax.default_backend() == 'tpu'
    config = encoder.EncoderConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, local_attention=4, attention_score='goat_input',
        compute_dtype='bfloat16', yat_compute_mode='bf16',
        yat_attention_implementation='centered_fp32_scores',
        yat_attention_block_size=4, yat_global_attention_block_size=4)
    block = encoder.EncoderBlock(config, index, rngs=nnx.Rngs(61))
    x = jax.random.normal(jax.random.key(62), (1, 8, 16))
    positions = jnp.arange(8)[None]
    segments = jnp.zeros((1, 8), dtype=jnp.int32)
    attention = encoder.bidirectional_attention
    captured = []
    def observe(q, k, v, segments, **kwargs):
        output = attention(q, k, v, segments, **kwargs)
        captured.append((q, k, v, output))
        return output
    monkeypatch.setattr(encoder, 'bidirectional_attention', observe)
    block(x, segments, positions)
    block.v.kernel[...] *= .5
    block(x, segments, positions)
    before, after = captured
    assert not hasattr(block, 'qkv')
    h = block.attn_norm(x) if index else x
    expected = encoder._rope(h.astype(jnp.bfloat16).reshape(1, 8, 2, 8),
                             positions, config.local_rope_theta if index else config.global_rope_theta)
    for score_operand in (before[0], before[1], after[0], after[1]):
        np.testing.assert_array_equal(score_operand, expected)
    # Identical score inputs/alpha imply identical YAT logits and weights, while
    # the independent value projection must change aggregation.
    assert np.any(np.asarray(before[2], dtype=np.float32) != np.asarray(after[2], dtype=np.float32))
    assert np.any(np.asarray(before[3], dtype=np.float32) != np.asarray(after[3], dtype=np.float32))


@pytest.mark.parametrize('architecture', ['goat', 'goat_input'])
def test_migrated_bf16_mlm_update_checkpoint_exact_resume(tmp_path, architecture):
    """Exercise the actual parent precision policy and persistent Adam state."""
    from dataclasses import asdict
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from flaxchat.common import replicate_on_mesh
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.goat import migrate_yat_to_goat
    from flaxchat.checkpoint import (create_checkpoint_manager, save_checkpoint,
                                    restore_model_from_checkpoint)
    assert jax.default_backend() == 'tpu'
    parent_config = EncoderConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, global_attn_every_n_layers=2, local_attention=4,
        ffn_type='yat_glu', attention_score='yat_softmax', compute_dtype='bfloat16',
        yat_compute_mode='bf16', yat_ffn_compute_mode='bf16_adaptive',
        yat_attention_implementation='centered_fp32_scores',
        yat_attention_block_size=4, yat_global_attention_block_size=4,
        mlm_projection='masked', mlm_loss_backend='xla_full', loss_chunk_size=4,
        yat_attention_alpha=.1)
    config = replace(parent_config, attention_score=architecture)
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    parent = ModernBert(parent_config, rngs=nnx.Rngs(42))
    model = ModernBert(config, rngs=nnx.Rngs(43))
    migrate_yat_to_goat(parent, model)
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    optimizer = nnx.Optimizer(model, optax.adamw(2e-5), wrt=nnx.Param)
    nnx.update(optimizer, replicate_on_mesh(nnx.state(optimizer), mesh))
    ids = jnp.array([[1, 4, 7, 8, 4, 9, 2, 0], [1, 4, 5, 6, 4, 7, 2, 0]])
    labels = jnp.array([[-1, 6, -1, -1, 5, -1, -1, -1], [-1, 8, -1, -1, 9, -1, -1, -1]])
    # The production projection maps across every data device. Preserve both
    # example patterns while satisfying that same global-batch contract.
    repeats = jax.device_count() // math.gcd(2, jax.device_count())
    ids, labels = jnp.tile(ids, (repeats, 1)), jnp.tile(labels, (repeats, 1))
    data_sharding = NamedSharding(mesh, P('data'))
    ids, labels = (jax.device_put(x, data_sharding) for x in (ids, labels))
    @nnx.jit
    def update(model, optimizer, ids, labels):
        loss, grads = nnx.value_and_grad(lambda m: m(ids, labels))(model)
        optimizer.update(model, grads)
        return loss, grads
    loss, grads = update(model, optimizer, ids, labels)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(x, dtype=np.float32)).all() for x in jax.tree.leaves(grads))
    for layer in grads['layers'].values():
        assert np.any(np.asarray(layer['v']['kernel'][...], dtype=np.float32) != 0)
        assert float(jnp.abs(layer['yat_attention_alpha'][...])) > 0
    path = str(tmp_path / 'goat-checkpoint')
    manager = create_checkpoint_manager(path, async_checkpointing=False)
    try:
        assert save_checkpoint(manager, 1, model, optimizer,
                               {'model_family': 'modernbert', 'resolved_config': asdict(config)})
        manager.wait_until_finished()
    finally:
        manager.close()
    restored = ModernBert(config, rngs=nnx.Rngs(44))
    # Orbax derives restore placement from these live targets, exactly as the
    # trainer does. An unreplicated target commits restored arrays to TPU 0.
    nnx.update(restored, replicate_on_mesh(nnx.state(restored), mesh))
    restored_optimizer = nnx.Optimizer(restored, optax.adamw(2e-5), wrt=nnx.Param)
    nnx.update(restored_optimizer, replicate_on_mesh(nnx.state(restored_optimizer), mesh))
    restore_model_from_checkpoint(restored, path, step=1, optimizer=restored_optimizer,
                                  expected_identity={'resolved_config': asdict(config)})
    nnx.update(restored_optimizer, replicate_on_mesh(nnx.state(restored_optimizer), mesh))
    for leaf in jax.tree.leaves(nnx.state((restored, restored_optimizer))):
        assert leaf.sharding.device_set == set(jax.devices())
    loss_a, _ = update(model, optimizer, ids, labels)
    loss_b, _ = update(restored, restored_optimizer, ids, labels)
    np.testing.assert_array_equal(loss_a, loss_b)
    for a, b in zip(jax.tree.leaves(nnx.state((model, optimizer))),
                    jax.tree.leaves(nnx.state((restored, restored_optimizer))), strict=True):
        np.testing.assert_array_equal(a, b)
