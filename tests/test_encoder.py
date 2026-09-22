from dataclasses import replace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.encoder import EncoderConfig, ModernBert, bidirectional_attention, import_hf_weights
from flaxchat.mlm import mask_tokens


def config(**kwargs):
    return EncoderConfig(**(dict(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, local_attention=2,
        loss_chunk_size=3, use_remat=False) | kwargs))


def test_attention_future_local_padding_and_segments():
    q = k = jnp.zeros((1, 5, 1, 2))
    v = jnp.broadcast_to(jnp.arange(5)[None, :, None, None], q.shape).astype(jnp.float32)
    segments = jnp.array([[0, 0, 0, 1, -1]])
    full = bidirectional_attention(q, k, v, segments)
    np.testing.assert_allclose(full[0, :, 0, 0], [1, 1, 1, 3, 0])
    local = bidirectional_attention(q, k, v, segments, radius=1)
    np.testing.assert_allclose(local[0, :, 0, 0], [.5, 1, 1.5, 3, 0])


def test_masking_is_partition_and_resume_independent():
    x = np.tile(np.arange(32), (8, 1))
    kwargs = dict(seed=42, step=9, vocab_size=32, mask_token_id=4,
                  special_token_ids=[0, 1, 2, 4], probability=.5)
    a, y = mask_tokens(x, example_ids=np.arange(8), **kwargs)
    b, z = mask_tokens(x[4:], example_ids=np.arange(4, 8), **kwargs)
    np.testing.assert_array_equal(a[4:], b)
    np.testing.assert_array_equal(y[4:], z)
    assert (y[:, [0, 1, 2, 4]] == -1).all()
    np.testing.assert_array_equal(y[y >= 0], x[y >= 0])
    c, _ = mask_tokens(x, example_ids=np.arange(8), **(kwargs | dict(step=10)))
    assert not np.array_equal(a, c)


def test_chunk_loss_gradients_remat_and_padding():
    m = ModernBert(config(), rngs=nnx.Rngs(0))
    x = jnp.array([[1, 4, 6, 0], [1, 7, 4, 0]])
    y = jnp.array([[-1, 5, -1, -1], [-1, -1, 8, -1]])
    def reference(model):
        logits = model(x)
        logp = jax.nn.log_softmax(logits, -1)
        return -(logp[0, 1, 5] + logp[1, 2, 8]) / 2
    loss, grads = nnx.value_and_grad(lambda model: model(x, y))(m)
    ref, refgrads = nnx.value_and_grad(reference)(m)
    np.testing.assert_allclose(loss, ref, rtol=1e-6)
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(refgrads), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
    remat = ModernBert(replace(config(), use_remat=True), rngs=nnx.Rngs(0))
    np.testing.assert_allclose(nnx.jit(lambda m: m(x, y))(remat), loss, rtol=1e-6)
    h = m.encode(x)
    np.testing.assert_array_equal(h[:, -1], 0)
    np.testing.assert_allclose(m.pool(x), h[:, :3].mean(1), atol=1e-6)
    assert float(m(x, jnp.full_like(x, -1))) == 0


def test_document_isolation():
    m = ModernBert(config(), rngs=nnx.Rngs(2))
    segments = jnp.array([[0, 0, 1, 1]])
    a = m.encode(jnp.array([[3, 4, 5, 6]]), segment_ids=segments)
    b = m.encode(jnp.array([[3, 4, 9, 10]]), segment_ids=segments)
    np.testing.assert_array_equal(a[:, :2], b[:, :2])


@pytest.mark.parametrize('compute', ['float32', 'bfloat16'])
@pytest.mark.parametrize('selection', ['sparse', 'overflow', 'empty'])
def test_masked_projection_preserves_loss_and_all_gradients(compute, selection):
    cfg = config(compute_dtype=compute, loss_chunk_size=4, num_hidden_layers=1)
    dense = ModernBert(cfg, rngs=nnx.Rngs(7))
    sparse = ModernBert(replace(cfg, mlm_projection='masked'), rngs=nnx.Rngs(7))
    x = jnp.tile(jnp.arange(1, 17), (2, 1)).at[0, -1].set(0)
    labels = jnp.full_like(x, -1)
    if selection == 'sparse':
        labels = labels.at[0, 1].set(5).at[1, 3:7].set(9)
    elif selection == 'overflow':
        # Overflow on just one row must retain every target on both rows.
        labels = labels.at[0, :5].set(8).at[1, 8].set(12)
    # Labels at padding/negative segments must be ignored before compaction.
    labels = labels.at[0, -1].set(7).at[1, -1].set(6)
    segments = jnp.zeros_like(x).at[:, -1].set(-1)
    call = nnx.jit(nnx.value_and_grad(lambda m, y: m(x, y, segment_ids=segments)))
    expected, expected_grads = call(dense, labels)
    actual, actual_grads = call(sparse, labels)
    tolerance = 3e-3 if compute == 'bfloat16' else 3e-6
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    for a, b in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads), strict=True):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)
    if selection == 'empty':
        assert float(actual) == 0
        assert all(np.all(np.asarray(g) == 0) for g in jax.tree.leaves(actual_grads))


@pytest.mark.parametrize('backend', ['xla', 'xla_full'])
def test_masked_projection_global_loss_with_sharded_rows(backend):
    if jax.device_count() < 2:
        pytest.skip('Requires multiple devices (JAX_NUM_CPU_DEVICES=4 for local check)')
    from jax.sharding import Mesh
    from flaxchat.common import replicate_on_mesh
    from flaxchat.training import place_host_batch
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    cfg = config(loss_chunk_size=4, num_hidden_layers=1)
    dense = ModernBert(cfg, rngs=nnx.Rngs(7))
    sparse = ModernBert(replace(cfg, mlm_projection='masked', mlm_loss_backend=backend), rngs=nnx.Rngs(7))
    for model in (dense, sparse):
        nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    x = np.tile(np.arange(1, 17, dtype=np.int32), (jax.local_device_count(), 1))
    call = nnx.jit(nnx.value_and_grad(lambda m, x, y: m(x, y)))
    for overflow in (False, True):
        y = np.full_like(x, -1)
        y[:, 1] = 5
        # Last shard either has more selected tokens or triggers global fallback.
        y[-1, :5 if overflow else 3] = 9
        placed_x, placed_y = place_host_batch(x, mesh), place_host_batch(y, mesh)
        a, ga = call(dense, placed_x, placed_y)
        b, gb = call(sparse, placed_x, placed_y)
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-6)
        for expected, actual in zip(jax.tree.leaves(ga), jax.tree.leaves(gb), strict=True):
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)


def test_huggingface_forward_loss_and_gradient_parity(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    hf_config = transformers.ModernBertConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, local_attention=2,
        global_attn_every_n_layers=3, global_rope_theta=160000., local_rope_theta=160000.,
        norm_eps=1e-5, pad_token_id=0, reference_compile=False,
        attention_dropout=0., embedding_dropout=0., mlp_dropout=0., classifier_bias=False)
    hf_config._attn_implementation = 'eager'
    torch.manual_seed(7)
    ref = transformers.ModernBertForMaskedLM(hf_config).eval()
    m = ModernBert(config(), rngs=nnx.Rngs(0))
    import json
    from scripts.convert_encoder_checkpoint import convert
    from scripts.train_encoder import load_pretrained
    (tmp_path / 'config.json').write_text(json.dumps(hf_config.to_dict()))
    torch.save(ref.state_dict(), tmp_path / 'pytorch_model.bin')
    conversion = convert(tmp_path)
    assert len(conversion['source_sha256']) == 64
    with pytest.raises(ValueError, match='already exists'):
        convert(tmp_path)
    assert len(load_pretrained(m, tmp_path)['model.safetensors']) == 64
    x = np.array([[1, 4, 6, 0], [1, 7, 4, 0]], dtype=np.int32)
    y = np.array([[-1, 5, -1, -1], [-1, -1, 8, -1]], dtype=np.int32)
    out = ref(torch.tensor(x, dtype=torch.long), attention_mask=torch.tensor(x != 0),
              labels=torch.tensor(np.where(y < 0, -100, y), dtype=torch.long))
    np.testing.assert_allclose(np.asarray(m(jnp.array(x)))[:, :3], out.logits.detach().numpy()[:, :3], atol=2e-6)
    loss, grads = nnx.value_and_grad(lambda model: model(jnp.array(x), jnp.array(y)))(m)
    np.testing.assert_allclose(loss, out.loss.detach().numpy(), atol=2e-6)
    out.loss.backward()
    np.testing.assert_allclose(grads['layers'][0]['qkv']['kernel'][...],
                               ref.model.layers[0].attn.Wqkv.weight.grad.numpy().T, atol=3e-6)


def test_import_rejection_is_atomic():
    m = ModernBert(config(), rngs=nnx.Rngs(0))
    before = np.asarray(m.embedding.embedding[...]).copy()
    with pytest.raises(ValueError, match='keys mismatch'):
        import_hf_weights(m, {'model.embeddings.tok_embeddings.weight': before * 0})
    np.testing.assert_array_equal(m.embedding.embedding[...], before)


@pytest.mark.parametrize('overrides', [dict(hidden_size=15), dict(norm_eps=0),
    dict(loss_chunk_size=0), dict(local_attention=3), dict(compute_dtype='float16')])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


def test_all_padding_is_finite_with_zero_gradients():
    m = ModernBert(config(), rngs=nnx.Rngs(0))
    x = jnp.zeros((1, 4), dtype=jnp.int32)
    loss, grads = nnx.value_and_grad(lambda m: m(x, jnp.ones_like(x)))(m)
    assert float(loss) == 0
    assert all(np.all(np.asarray(g) == 0) for g in jax.tree.leaves(grads))
    np.testing.assert_array_equal(m.pool(x), 0)


@pytest.mark.accelerator
def test_tpu_splash_encoder_attention_and_gradients():
    if jax.default_backend() != 'tpu':
        pytest.skip('Requires a physical TPU')
    q, k, v = jax.random.normal(jax.random.key(0), (3, 1, 128, 2, 64))
    segments = jnp.array([[0] * 64 + [1] * 32 + [-1] * 32])
    for radius in (None, 16):
        def call(q, backend, radius=radius):
            return bidirectional_attention(q, k, v, segments, radius=radius, backend=backend)
        np.testing.assert_allclose(call(q, 'splash'), call(q, 'xla'), atol=3e-3, rtol=3e-3)
        a = jax.grad(lambda q: call(q, 'splash').sum())(q)
        b = jax.grad(lambda q: call(q, 'xla').sum())(q)
        np.testing.assert_allclose(a, b, atol=3e-3, rtol=3e-3)


def test_hf_config_contract():
    raw = dict(model_type='modernbert', hidden_size=16, num_attention_heads=2)
    assert EncoderConfig.from_hf(raw).hidden_size == 16
    for bad in (dict(attention_dropout=.1), dict(hidden_activation='relu'),
                dict(tie_word_embeddings=False), dict(classifier_activation='relu'),
                dict(rope_scaling={'type': 'linear'})):
        with pytest.raises(ValueError):
            EncoderConfig.from_hf(raw | bad)


@pytest.mark.parametrize('residual_dtype', ['float32', 'bfloat16'])
def test_bfloat16_training_gradients_are_finite(residual_dtype):
    m = ModernBert(config(compute_dtype='bfloat16', residual_dtype=residual_dtype, use_remat=True), rngs=nnx.Rngs(1))
    x = jnp.array([[1, 4, 6, 0]])
    loss, grads = nnx.value_and_grad(lambda model: model(x, jnp.array([[-1, 5, -1, -1]])))(m)
    assert m.encode(x).dtype == getattr(jnp, residual_dtype)
    assert all(v.dtype == jnp.float32 for v in jax.tree.leaves(nnx.state(m, nnx.Param)))
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grads))


@pytest.mark.parametrize('overrides', [dict(probability=2), dict(seed=-1),
    dict(mask_token_id=32), dict(special_token_ids=list(range(32)))])
def test_masking_rejects_invalid_configuration(overrides):
    kwargs = dict(seed=0, step=0, example_ids=np.array([0]), vocab_size=32,
                  mask_token_id=4, special_token_ids=[0, 4]) | overrides
    with pytest.raises(ValueError):
        mask_tokens(np.array([[1, 2, 3]]), **kwargs)


def test_reference_rejects_wrong_precision_and_checkpoint():
    import json
    from scripts.validate_encoder_checkpoint import check_reference
    identity = {'model.safetensors': 'expected-hash'}
    data = dict(snapshot_identity=json.dumps(identity), dtype='float32', residual_dtype='float32')
    assert check_reference(data, dtype='float32', residual_dtype='float32', identity=identity)
    with pytest.raises(ValueError, match='precision mismatch'):
        check_reference(data, dtype='bfloat16', residual_dtype='float32', identity=identity)
    assert not check_reference(data, dtype='bfloat16', residual_dtype='float32',
                               identity=identity, diagnostic=True)
    with pytest.raises(ValueError, match='snapshot identity mismatch'):
        check_reference(data, dtype='float32', residual_dtype='float32',
                        identity={'model.safetensors': 'wrong'}, diagnostic=True)
    with pytest.raises(ValueError, match='lacks snapshot provenance'):
        check_reference({}, dtype='float32', residual_dtype='float32', identity=identity)
