"""Physical global-denominator gradient cache parity, including replayed RNG."""
import os
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
pytestmark = pytest.mark.skipif(os.environ.get('FLAXCHAT_PHYSICAL_TPU') != '1', reason='Physical TPU admission only')


def _loss_diagnostic(value):
    """Keep a JSON-safe scalar and its failure state before acceptance asserts."""
    scalar = np.asarray(value)
    if scalar.shape != ():
        raise AssertionError('Diagnostic loss must be scalar')
    number = float(scalar)
    return {'value': number if np.isfinite(number) else None,
            'finite': bool(np.isfinite(number)),
            'nonfinite_kind': None if np.isfinite(number) else (
                'nan' if np.isnan(number) else 'positive_infinity' if number > 0 else 'negative_infinity')}


def _comparison_summary(sections):
    """Locate failures without inferring their cause from aggregate magnitudes."""
    return {name: {'leaf_count': len(records),
                   'failing_leaves': [record['path'] for record in records
                                      if not record['finite'] or record['mismatch_count']],
                   'mismatch_count': sum(record['mismatch_count'] for record in records),
                   'nonfinite_leaf_count': sum(not record['finite'] for record in records)}
            for name, records in sections.items()}



def _embedding_boundary_summary(sections):
    """Exact boundary observations only; never infer a kernel cause or acceptance."""
    names = ('cache_full_embeddings', 'cache_full_embedding_cotangents',
             'cache_chunked_embeddings', 'cache_chunked_embedding_cotangents',
             'chunked_full_embeddings', 'chunked_full_embedding_cotangents')
    result = {}
    for name in names:
        records = sections.get(name)
        result[name] = None if records is None else {
            'all_finite': all(row['finite'] for row in records),
            'exactly_equal': all(row['finite'] and row['mismatch_count'] == 0 for row in records),
            'mismatch_count': sum(row['mismatch_count'] for row in records)}
    return result


def _leaf_diagnostics(actual, expected, *, rtol, atol):
    """Host statistics on already-produced TPU leaves; no acceptance relaxation."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    if a.shape != b.shape:
        raise AssertionError('Diagnostic shape mismatch')
    delta = a - b
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    mismatches = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    indices = np.argwhere(mismatches)[:8]
    reference_norm = float(np.linalg.norm(b.ravel()))
    delta_norm = float(np.linalg.norm(delta.ravel()))
    return {'shape': list(a.shape), 'actual_dtype': str(actual.dtype), 'expected_dtype': str(expected.dtype),
        'finite': finite, 'rtol': rtol, 'atol': atol,
        'max_abs_delta': float(np.max(np.abs(delta))) if finite and a.size else None,
        'absolute_l2_delta': delta_norm if finite else None,
        'relative_l2_delta': delta_norm / reference_norm if finite and reference_norm else None,
        'reference_l2': reference_norm if finite else None, 'mismatch_count': int(mismatches.sum()),
        'element_count': a.size, 'first_mismatches': [
            {'index': location.tolist(), 'actual': float(a[tuple(location)]) if np.isfinite(a[tuple(location)]) else None,
             'expected': float(b[tuple(location)]) if np.isfinite(b[tuple(location)]) else None}
            for location in indices]}


def test_global_gradient_cache_preserves_loss_remote_gradients_and_rng():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from flaxchat.embedding_gradient_cache import make_cached_encoder
    from flaxchat.contrastive import hard_negative_infonce
    assert jax.default_backend() == 'tpu'
    mesh = Mesh(np.array(jax.devices()), ('data',))
    width = jax.device_count()
    batch = 2 * width
    # This fixture needs at least two rows even on a one-device topology.
    rng = np.random.default_rng(29)
    params = {'kernel': jnp.array(rng.normal(size=(128, 128)) * .05, jnp.float32), 'bias': jnp.zeros(128)}
    params = jax.device_put(params, NamedSharding(mesh, P()))
    inputs = [jax.device_put(jnp.array(rng.normal(size=(batch, 128)), jnp.float32), NamedSharding(mesh, P('data')))
              for _ in range(3)]
    keys = jax.random.split(jax.random.key(29), batch)
    def encode(parameters, frozen, rows, row_keys):
        del frozen
        # Stateless per-row randomness verifies that cache replay uses the same
        # realization; it does not claim ModernBert has dropout.
        noise = jax.vmap(lambda key: jax.random.normal(key, (128,)))(row_keys) * .01
        return jnp.tanh(rows @ parameters['kernel'] + parameters['bias']) + noise
    cached = make_cached_encoder(encode, width)
    ids = jnp.arange(batch, dtype=jnp.int32)
    def objective(parameters, encoder):
        q, p, n = [encoder(parameters, (), rows, keys) for rows in inputs]
        return hard_negative_infonce(q, p, n, ids + 1, ids + 10001, ids + 20001,
            ids + 1, jnp.ones(batch, bool), expected_global_batch=batch,
            known_positive_text_ids=(ids + 10001)[:, None])
    direct_value, direct_grad = jax.jit(jax.value_and_grad(lambda p: objective(p, encode)))(params)
    cache_value, cache_grad = jax.jit(jax.value_and_grad(lambda p: objective(p, cached)))(params)
    np.testing.assert_allclose(float(cache_value), float(direct_value), rtol=2e-5, atol=2e-5)
    for actual, expected in zip(jax.tree.leaves(cache_grad), jax.tree.leaves(direct_grad), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-4, atol=2e-5)



@pytest.mark.parametrize('pair_only', [True, False])
def test_nnx_yat_cache_matches_encoder_gradients_and_optimizer_update(pair_only, tmp_path, record_property):
    _nnx_cache_case(pair_only, tmp_path, record_property, shape_matched=False)


@pytest.mark.parametrize('pair_only,yat', [(True, False), (False, False), (True, True), (False, True)],
                         ids=['True', 'False', 'yat-pair', 'yat-triplet'])
def test_nnx_cache_shape_matched_reference_and_adam_moments(pair_only, yat, tmp_path, record_property):
    """Additional diagnosis; original full-batch acceptance stays mandatory."""
    _nnx_cache_case(pair_only, tmp_path, record_property, shape_matched=True, yat=yat)


def _nnx_cache_case(pair_only, tmp_path, record_property, *, shape_matched, yat=False):
    """Qualify the actual NNX adapter independently of the pure-JAX cache oracle."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from flaxchat.common import replicate_on_mesh
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.embedding_gradient_cache import nnx_cached_pool
    from flaxchat.contrastive import hard_negative_infonce
    from flaxchat.embedding_contract import source_identity
    from flaxchat.runtime import runtime_identity
    assert jax.default_backend() == 'tpu'
    width = jax.device_count()
    batch = 2 * width
    mesh = Mesh(np.array(jax.devices()), ('data',))
    config = EncoderConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=512,
        local_attention=128, use_remat=False, compute_dtype='bfloat16', residual_dtype='float32',
        ffn_type='yat_glu', attention_score='yat_softmax', yat_compute_mode='bf16',
        yat_ffn_compute_mode='bf16_adaptive', yat_attention_implementation='centered_fp32_scores',
        yat_attention_block_size=128, yat_global_attention_block_size=128)
    direct_model = ModernBert(config, rngs=nnx.Rngs(29))
    cached_model = ModernBert(config, rngs=nnx.Rngs(29))
    models = [direct_model, cached_model]
    if shape_matched:
        models.append(ModernBert(config, rngs=nnx.Rngs(29)))
    objective_identity = None
    if yat:
        from argparse import Namespace
        from flaxchat.embedding_stage import contrastive_objective_identity
        from flaxchat.embedding_objective import configure_objective_state, objective_loss_arguments
        objective_identity = contrastive_objective_identity(Namespace(
            contrastive_similarity='yat', yat_infonce_alpha_init=.01))
        for model in models:
            configure_objective_state(model, objective_identity)
    for model in models:
        nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    initial_loss_arguments = objective_loss_arguments(direct_model, objective_identity) if yat else {}
    initial_raw_alpha = np.asarray(direct_model.contrastive_raw_alpha[...]).copy() if yat else None
    optimizer_config = {'learning_rate': 1e-5, 'b1': .9, 'b2': .999,
                        'eps': 1e-8, 'eps_root': 0., 'weight_decay': 1e-4}
    optimizers = [nnx.Optimizer(model, optax.adamw(**optimizer_config), wrt=nnx.Param)
                  for model in models]
    initial = {jax.tree_util.keystr(path): np.asarray(leaf).copy()
               for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(direct_model))[0]}
    rng = np.random.default_rng(29)
    tokens = [rng.integers(5, 128, (batch, 128), dtype=np.int32) for _ in range(3)]
    for rows in tokens:
        rows[:, -11:] = 0  # Exercise nonuniform physical encoder masks.
        rows[0, -32:] = 0
    inputs = [jax.device_put(jnp.asarray(rows), NamedSharding(mesh, P('data'))) for rows in tokens]
    ids = jnp.arange(batch, dtype=jnp.int32) + 1

    def objective(model, cached):
        if cached == 'chunked-autodiff':
            def pool(rows):
                return jnp.concatenate([model.pool(rows[first:first + width])
                                        for first in range(0, batch, width)])
        else:
            pool = (lambda rows: nnx_cached_pool(model, rows, width)) if cached else model.pool
        q, p = pool(inputs[0]), pool(inputs[1])
        n = jnp.zeros_like(p) if pair_only else pool(inputs[2])
        arguments = objective_loss_arguments(model, objective_identity) if yat else {}
        loss = embedding_objective(q, p, n, arguments)
        # Preserve pre-update TPU outputs as aux; no extra encoder execution.
        return loss, {'query': q, 'positive': p, 'negative': n}

    def embedding_objective(q, p, n, arguments=None):
        arguments = initial_loss_arguments if arguments is None else arguments
        return hard_negative_infonce(q, p, n, ids, ids + 10001, ids + 20001, ids,
            jnp.full(batch, not pair_only), expected_global_batch=batch,
            known_positive_text_ids=(ids + 10001)[:, None], **arguments)

    @nnx.jit
    def direct_step(model, optimizer):
        (loss, embeddings), gradients = nnx.value_and_grad(lambda inner: objective(inner, False), has_aux=True)(model)
        optimizer.update(model, gradients)
        return loss, gradients, embeddings

    @nnx.jit
    def cached_step(model, optimizer):
        (loss, embeddings), gradients = nnx.value_and_grad(lambda inner: objective(inner, True), has_aux=True)(model)
        optimizer.update(model, gradients)
        return loss, gradients, embeddings

    direct_loss, direct_grad, direct_embeddings = direct_step(direct_model, optimizers[0])
    cached_loss, cached_grad, cached_embeddings = cached_step(cached_model, optimizers[1])
    chunked_loss, chunked_grad, chunked_embeddings = None, None, None
    if shape_matched:
        @nnx.jit
        def chunked_step(model, optimizer):
            (loss, embeddings), gradients = nnx.value_and_grad(lambda inner: objective(inner, 'chunked-autodiff'), has_aux=True)(model)
            optimizer.update(model, gradients)
            return loss, gradients, embeddings
        chunked_loss, chunked_grad, chunked_embeddings = chunked_step(models[2], optimizers[2])
    embedding_pullback = jax.jit(jax.grad(embedding_objective, argnums=(0, 1, 2)))
    def embedding_cotangents(embeddings):
        values = embedding_pullback(embeddings['query'], embeddings['positive'], embeddings['negative'])
        return dict(zip(('query', 'positive', 'negative'), values, strict=True))
    direct_cotangents = embedding_cotangents(direct_embeddings)
    cached_cotangents = embedding_cotangents(cached_embeddings)
    chunked_cotangents = embedding_cotangents(chunked_embeddings) if shape_matched else None
    diagnostics = {'format': 'flaxchat-physical-cache-numerical-diagnostic-v2',
        'pair_only': pair_only, 'backend': jax.default_backend(), 'devices': width,
        'processes': jax.process_count(), 'cache_chunk_rows': width, 'global_batch_rows': batch,
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source': source_identity(Path(__file__).parents[1]), 'runtime': runtime_identity(),
        'effective_jax_settings': {'jax_default_matmul_precision': jax.config.jax_default_matmul_precision,
                                   'jax_enable_x64': bool(jax.config.jax_enable_x64)},
        'physical_device_kinds': sorted({device.device_kind for device in jax.devices()}),
        'effective_encoder_config': asdict(config),
        'optimizer_config': {'name': 'adamw', **optimizer_config},
        'input_sha256': [hashlib.sha256(rows.tobytes()).hexdigest() for rows in tokens],
        'input_shape': list(tokens[0].shape), 'input_dtype': str(tokens[0].dtype),
        'loss': {'actual': _loss_diagnostic(cached_loss), 'expected': _loss_diagnostic(direct_loss)},
        'shape_matched_reference': shape_matched,
        'contrastive_objective': objective_identity,
        'embedding_boundary_diagnostics': {
            'scope': 'pre-update TPU aux embeddings and independently differentiated global loss cotangents',
            'encoder_reexecuted_for_diagnostics': False,
            'comparison': 'exact differences reported only; original acceptance bounds unchanged',
            'purpose': 'separate forward batch-shape drift from parameter-VJP/reduction drift; no inferred cause'},
        'sections': {}}
    cases = [('gradient', cached_grad, direct_grad, 2e-2, 2e-3),
             ('model', nnx.state(cached_model), nnx.state(direct_model), 2e-3, 2e-5),
             ('optimizer', nnx.state(optimizers[1]), nnx.state(optimizers[0]), 2e-2, 2e-6)]
    cases.extend([('cache_full_embeddings', cached_embeddings, direct_embeddings, 0., 0.),
                  ('cache_full_embedding_cotangents', cached_cotangents, direct_cotangents, 0., 0.)])
    if shape_matched:
        cases.extend([('cache_chunked_embeddings', cached_embeddings, chunked_embeddings, 0., 0.),
                      ('cache_chunked_embedding_cotangents', cached_cotangents, chunked_cotangents, 0., 0.),
                      ('chunked_full_embeddings', chunked_embeddings, direct_embeddings, 0., 0.),
                      ('chunked_full_embedding_cotangents', chunked_cotangents, direct_cotangents, 0., 0.)])
    # The original gradient acceptance is intentionally much looser than the
    # first-moment comparison. Show the gradients at the implied Adam-mu bound
    # as diagnostic evidence, without changing any mandatory tolerance.
    mu_factor = 1 - optimizer_config['b1']
    nu_factor = 1 - optimizer_config['b2']
    diagnostics['adam_mu_implied_gradient_bound'] = {
        'rtol': 2e-2, 'atol': 2e-6 / mu_factor,
        'derivation': 'step-one mu=(1-b1)*gradient; original optimizer atol/(1-b1)',
        'acceptance_policy_changed': False}
    cases.append(('gradient_at_adam_mu_bound', cached_grad, direct_grad,
                  2e-2, 2e-6 / mu_factor))
    if shape_matched:
        diagnostics['loss']['ordinary_chunked'] = _loss_diagnostic(chunked_loss)
        cases.extend([('cache_vs_chunked_gradient', cached_grad, chunked_grad, 2e-2, 2e-3),
                      ('cache_vs_chunked_model', nnx.state(cached_model), nnx.state(models[2]), 2e-3, 2e-5),
                      ('cache_vs_chunked_optimizer', nnx.state(optimizers[1]), nnx.state(optimizers[2]), 2e-2, 2e-6),
                      ('chunked_vs_full_gradient', chunked_grad, direct_grad, 2e-2, 2e-3),
                      ('chunked_vs_full_optimizer', nnx.state(optimizers[2]), nnx.state(optimizers[0]), 2e-2, 2e-6)])
    # At step one, Adam's uncorrected moments are (1-b1)*g and
    # (1-b2)*g**2. Check each optimizer against its OWN gradients, separating
    # optimizer bookkeeping from encoder reduction/shape differences.
    for name, optimizer, gradients in [('direct', optimizers[0], direct_grad), ('cache', optimizers[1], cached_grad)] + (
            [('chunked', optimizers[2], chunked_grad)] if shape_matched else []):
        adam_state = nnx.as_pure(optimizer.opt_state)[0]
        plain_gradients = nnx.as_pure(gradients)
        cases.extend([(name + '_adam_mu_identity', adam_state.mu,
                       jax.tree.map(lambda value: mu_factor * value, plain_gradients), 2e-6, 2e-8),
                      (name + '_adam_nu_identity', adam_state.nu,
                       jax.tree.map(lambda value: nu_factor * value ** 2, plain_gradients), 2e-6, 2e-8)])
    # Independently attribute the cross-path moment drift to its gradient drift.
    # Own-moment identities alone do not expose where a permissive gradient
    # tolerance can still exceed the stricter optimizer-state bound.
    direct_adam, cache_adam = [nnx.as_pure(optimizer.opt_state)[0] for optimizer in optimizers[:2]]
    direct_plain, cache_plain = nnx.as_pure(direct_grad), nnx.as_pure(cached_grad)
    cases.extend([
        ('cache_full_adam_mu_delta_identity',
         jax.tree.map(lambda cache, full: cache - full, cache_adam.mu, direct_adam.mu),
         jax.tree.map(lambda cache, full: mu_factor * (cache - full), cache_plain, direct_plain),
         2e-6, 2e-8),
        ('cache_full_adam_nu_delta_identity',
         jax.tree.map(lambda cache, full: cache - full, cache_adam.nu, direct_adam.nu),
         jax.tree.map(lambda cache, full: nu_factor * (cache ** 2 - full ** 2), cache_plain, direct_plain),
         2e-6, 2e-8)])
    for section, actual_tree, expected_tree, rtol, atol in cases:
        assert jax.tree.structure(actual_tree) == jax.tree.structure(expected_tree)
        actual_leaves = jax.tree_util.tree_flatten_with_path(actual_tree)[0]
        expected_leaves = jax.tree_util.tree_flatten_with_path(expected_tree)[0]
        records = []
        for (path, actual), (expected_path, expected) in zip(actual_leaves, expected_leaves, strict=True):
            assert path == expected_path
            name = jax.tree_util.keystr(path)
            result = {'path': name, **_leaf_diagnostics(actual, expected, rtol=rtol, atol=atol)}
            if section == 'model':
                before = initial[name].astype(np.float64)
                result['actual_update_l2'] = float(np.linalg.norm((np.asarray(actual).astype(np.float64) - before).ravel()))
                result['expected_update_l2'] = float(np.linalg.norm((np.asarray(expected).astype(np.float64) - before).ravel()))
            records.append(result)
        diagnostics['sections'][section] = records
    diagnostics['comparison_summary'] = _comparison_summary(diagnostics['sections'])
    diagnostics['embedding_boundary_observations'] = _embedding_boundary_summary(diagnostics['sections'])
    encoded = json.dumps(diagnostics, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256((encoded + '\n').encode()).hexdigest()
    evidence = tmp_path / f'cache-{pair_only}-{digest}.json'
    with evidence.open('x') as stream:
        stream.write(encoded + '\n')
    # JUnit properties persist the full diagnostic even if the next assertion
    # fails; content-addressed local evidence also refuses overwrite.
    record_property('gradient_cache_diagnostic_sha256', digest)
    record_property('gradient_cache_diagnostic_json', encoded)
    print(f'FLAXCHAT_CACHE_DIAGNOSTIC {encoded}', flush=True)
    assert all(result['finite'] for result in diagnostics['loss'].values()), diagnostics['loss']
    for section in diagnostics['sections']:
        if section.endswith(('_adam_mu_identity', '_adam_nu_identity')) or section.startswith('cache_vs_chunked_'):
            for result in diagnostics['sections'][section]:
                assert result['finite'] and result['mismatch_count'] == 0, f'{section}: {result}'
    np.testing.assert_allclose(float(cached_loss), float(direct_loss), rtol=2e-3, atol=2e-3)
    if shape_matched:
        np.testing.assert_allclose(float(cached_loss), float(chunked_loss), rtol=2e-3, atol=2e-3)
    for left, right in zip(jax.tree.leaves(cached_grad), jax.tree.leaves(direct_grad), strict=True):
        assert np.isfinite(np.asarray(left)).all()
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=2e-2, atol=2e-3)
    for left, right in zip(jax.tree.leaves(nnx.state(cached_model)), jax.tree.leaves(nnx.state(direct_model)), strict=True):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=2e-3, atol=2e-5)
    left_state, right_state = nnx.state(optimizers[1]), nnx.state(optimizers[0])
    assert jax.tree.structure(left_state) == jax.tree.structure(right_state)
    for left, right in zip(jax.tree.leaves(left_state), jax.tree.leaves(right_state), strict=True):
        actual, expected = np.asarray(left), np.asarray(right)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-6)
    assert int(optimizers[0].step[...]) == int(optimizers[1].step[...]) == 1

    if yat:
        for gradient, model in zip((direct_grad, cached_grad, chunked_grad), models, strict=True):
            alpha_gradient = np.asarray(nnx.as_pure(gradient)['contrastive_raw_alpha'])
            assert np.isfinite(alpha_gradient).all() and np.any(alpha_gradient != 0)
            assert not np.array_equal(np.asarray(model.contrastive_raw_alpha[...]), initial_raw_alpha)
