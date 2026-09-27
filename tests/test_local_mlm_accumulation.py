"""Run with four forced CPU devices; physical TPU qualification is separate."""
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import pytest
from jax.sharding import Mesh

from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.training import gradients_for_mlm_microbatches, gradients_for_local_mlm_microbatches

pytestmark = pytest.mark.skipif(jax.device_count() not in (4, 8, 16, 32), reason='requires 4, 8, 16 or 32 devices')


def accumulation_case(yat, mask_case, rows_per_device=1):
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, compute_dtype='bfloat16' if yat else 'float32',
        residual_dtype='bfloat16' if yat else 'float32', loss_chunk_size=4,
        ffn_type='yat_glu' if yat else 'geglu', attention_score='yat_softmax' if yat else 'dot_product',
        yat_compute_mode='bf16_adaptive' if yat else 'mixed', yat_attention_alpha=.1,
        yat_local_shards=yat, yat_attention_block_size=4, yat_global_attention_block_size=4)
    model = ModernBert(config, rngs=nnx.Rngs(42))
    inputs = jnp.array([[[5,6,7,0],[6,7,8,0],[8,9,7,0],[9,8,6,0]]]*2)
    inputs = jnp.tile(inputs, (1, jax.device_count() // 4 * rows_per_device, 1))
    labels = jnp.full(inputs.shape, -1)
    if mask_case != 'empty_all':
        # Uneven counts across both devices and microsteps; first microstep
        # entirely empty. Equal averaging by shard or microstep is incorrect.
        labels = labels.at[1,0,0].set(6).at[1,2,:3].set(jnp.array([7,8,9]))
        if mask_case == 'uneven':
            labels = labels.at[0,1,:2].set(jnp.array([5,6]))
    return model, inputs, labels


def record_case(model, inputs, labels, before, reference, actual, case_name, reference_kind):
    config = model.config
    # Preserve actual device tensors before assertions: CPU seed replay does
    # not necessarily reproduce TPU initialization or BF16 arithmetic.
    import os
    evidence_directory = os.environ.get('FLAXCHAT_NUMERICAL_EVIDENCE_DIR')
    if evidence_directory:
        from dataclasses import asdict
        import hashlib
        from pathlib import Path
        from scripts.numerical_evidence import write_numerical_evidence
        from flaxchat.runtime import runtime_identity
        arrays = {'inputs': inputs, 'targets': labels,
                  'reference_loss': reference[0], 'candidate_loss': actual[0]}
        paths = {}
        for prefix, tree in [('parameter', before), ('reference', reference[1]), ('candidate', actual[1])]:
            for index, (path, value) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
                name = f'{prefix}_{index}'
                arrays[name] = value
                paths[name] = jax.tree_util.keystr(path)
        sources = [*Path('flaxchat').glob('*.py'), Path(__file__)]
        write_numerical_evidence(Path(evidence_directory) / f'{case_name}-{jax.process_index()}',
            arrays, metadata={'config': asdict(config), 'leaf_paths': paths, 'reference_kind': reference_kind,
                'runtime': runtime_identity(), 'backend': jax.default_backend(),
                'effective_matmul_precision': jax.config.values['jax_default_matmul_precision'],
                'devices': [str(d) for d in jax.devices()],
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'scope': 'Exact device inputs, initialized parameters and compared gradients; not a correctness oracle.'})


@pytest.mark.parametrize('yat', [False, True])
@pytest.mark.parametrize('mask_case', ['empty_first', 'uneven', 'empty_all'])
def test_local_accumulation_preserves_global_target_weighting(yat, mask_case):
    model, inputs, labels = accumulation_case(yat, mask_case)
    config = model.config
    before = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    reference = nnx.jit(gradients_for_mlm_microbatches)(model, inputs, labels)
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    actual = nnx.jit(partial(gradients_for_local_mlm_microbatches, mesh=mesh))(model, inputs, labels)
    record_case(model, inputs, labels, before, reference, actual,
                f'mlm-{mask_case}-{yat}', 'legacy_batched')
    for a,b in zip(jax.tree.leaves(reference), jax.tree.leaves(actual), strict=True):
        a,b = np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
        assert np.isfinite(b).all()
        np.testing.assert_allclose(a,b,atol=.003 if yat else 2e-6,rtol=.04 if yat else 2e-5)
        if mask_case == 'empty_all':
            np.testing.assert_array_equal(b,0)
    assert model.config == config and all(layer.config == config for layer in model.layers)
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(nnx.state(model,nnx.Param)),strict=True):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('backend', ['xla', 'xla_local', 'pallas'])
def test_local_accumulation_training_exact_resume(tmp_path, monkeypatch, backend):
    from tests.test_encoder_training import test_preparation_training_and_exact_resume
    test_preparation_training_and_exact_resume(tmp_path, monkeypatch,
        'bfloat16', 'bfloat16', 'masked', backend, 'yat_adaptive_local',
        attention_alpha=.1, local_gradient_accumulation=True)


def test_accumulation_ignores_targets_on_padding():
    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=1, num_attention_heads=2, compute_dtype='float32',
        residual_dtype='float32', loss_chunk_size=4)
    model = ModernBert(config, rngs=nnx.Rngs(7))
    inputs = jnp.tile(jnp.array([[[5, 0, 0]], [[6, 7, 0]]]),
                      (1, jax.device_count(), 1))
    clean = jnp.where(inputs != 0, inputs, -1)
    contaminated = jnp.where(inputs == 0, 8, clean)
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    for fn in [gradients_for_mlm_microbatches,
               partial(gradients_for_local_mlm_microbatches, mesh=mesh)]:
        compiled = nnx.jit(fn)
        expected = compiled(model, inputs, clean)
        actual = compiled(model, inputs, contaminated)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('backend', ['xla_full', 'xla_local', 'pallas'])
@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_local_projection_matches_global_weighted_loss(backend, reduction):
    from flaxchat.fused_cross_entropy import sharded_fused_loss
    from jax.sharding import PartitionSpec as P
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    h = jnp.asarray(np.random.default_rng(1).normal(size=(jax.device_count(),2,128))*.1, jnp.float32)
    w = jnp.asarray(np.random.default_rng(2).normal(size=(16,128))*.1, jnp.float32)
    b = jnp.zeros(16)
    y = jnp.full((jax.device_count(),2),-1).at[:4].set(jnp.array([[-1,-1],[1,-1],[2,3],[4,-1]]))
    def reference(weight):
        return sharded_fused_loss(h,weight,b,y,backend=backend,interpret=True,tile=128,chunk_size=2)
    def candidate(weight):
        def local(hh,ww,bb,yy):
            count=(yy>=0).sum()
            loss=sharded_fused_loss(hh,ww,bb,yy,backend=backend,interpret=True,
                tile=128,chunk_size=2,local_only=True,reduction=reduction)
            numerator=loss if reduction == 'sum' else loss*count
            return jax.lax.psum(numerator,'data')/jnp.maximum(jax.lax.psum(count,'data'),1)
        return jax.shard_map(local,mesh=mesh,in_specs=(P('data'),P(),P(),P('data')),
                             out_specs=P(),check_vma=False)(h,weight,b,y)
    a=jax.jit(jax.value_and_grad(reference))(w)
    c=jax.jit(jax.value_and_grad(candidate))(w)
    for left,right in zip(a,c,strict=True):
        np.testing.assert_allclose(left,right,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('yat', [False, True])
@pytest.mark.parametrize('mask_case', ['empty_first', 'uneven', 'empty_all'])
@pytest.mark.parametrize('rows_per_device', [1, 2])
def test_local_accumulation_matches_per_example_reference(yat, mask_case, rows_per_device):
    from scripts.replay_mlm_accumulation import per_example_reference, flatten
    model, inputs, labels = accumulation_case(yat, mask_case, rows_per_device)
    config = model.config
    before = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    if rows_per_device > 1 and mask_case != 'empty_all':
        labels = labels.at[1, 1, :2].set(jnp.array([8, 9]))
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    actual_loss, actual_grads = nnx.jit(
        partial(gradients_for_local_mlm_microbatches, mesh=mesh)
    )(model, inputs, labels)
    loss, grads, count = per_example_reference(model, inputs, labels)
    pairs, treedef = jax.tree_util.tree_flatten_with_path(actual_grads)
    reference_grads = jax.tree_util.tree_unflatten(
        treedef, [grads[jax.tree_util.keystr(path)] for path, _ in pairs])
    record_case(model, inputs, labels, before, (loss, reference_grads),
                (actual_loss, actual_grads),
                f'independent-{mask_case}-{yat}-{rows_per_device}', 'per_example')
    assert count == int(np.asarray((labels >= 0) & (inputs != model.config.pad_token_id)).sum())
    np.testing.assert_allclose(actual_loss, loss, atol=2e-5, rtol=2e-4)
    actual = flatten(actual_grads)
    assert actual.keys() == grads.keys()
    # Preserve the established BF16 error budget for batched derivatives.
    # Per-example BF16 derivatives and a batched BF16 derivative round their
    # parameter reductions differently. The tighter observed single-row bound
    # must not be generalized to multi-row batches.
    atol, rtol = (.003, .04) if yat else (2e-6, 2e-5)
    for name in grads:
        assert np.isfinite(actual[name]).all()
        np.testing.assert_allclose(actual[name], grads[name], atol=atol, rtol=rtol,
                                   err_msg=name)
        if rows_per_device == 1:
            np.testing.assert_allclose(actual[name], grads[name], atol=2e-5, rtol=2e-4,
                                       err_msg=name)
    # The old-path agreement gate remains separate and retains its tolerance.
    assert model.config == config and all(layer.config == config for layer in model.layers)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True):
        np.testing.assert_array_equal(a, b)
