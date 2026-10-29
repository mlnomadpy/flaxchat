"""Physical-only final-batch numerics; full checkpoint migration is a separate gate."""

import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLAXCHAT_PHYSICAL_TPU") != "1",
    reason="Physical TPU admission required; never emulate on CPU",
)


@pytest.mark.parametrize("valid_rows", [1, 3, 63])
def test_padded_goat_tail_matches_global_masked_token_weighting(valid_rows):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.common import replicate_on_mesh
    from flaxchat.mlm import mask_prepared_rows
    from flaxchat.training import (
        gradients_for_local_mlm_microbatches,
        gradients_for_mlm_microbatches,
    )

    assert (
        jax.default_backend() == "tpu"
        and jax.device_count() == 8
        and jax.process_count() == 1
    )
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    config = EncoderConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        compute_dtype="bfloat16",
        residual_dtype="float32",
        ffn_type="yat_glu",
        attention_score="goat_input",
        yat_compute_mode="bf16",
        yat_ffn_compute_mode="bf16_adaptive",
        yat_attention_alpha=0.1,
        yat_attention_implementation="centered_fp32_scores",
        yat_attention_block_size=4,
        yat_global_attention_block_size=4,
    )
    model = ModernBert(config, rngs=nnx.Rngs(3))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    tokens = np.tile(np.arange(5, 13, dtype=np.int32), (64, 1))
    ids = np.r_[np.arange(valid_rows), np.full(64 - valid_rows, -1)].astype(np.int64)
    x, y = mask_prepared_rows(
        tokens,
        ids,
        pad_token_id=0,
        allow_padding=True,
        seed=9,
        step=100,
        vocab_size=64,
        mask_token_id=4,
        special_token_ids=[0, 4],
        probability=1.0,
    )
    assert int(np.count_nonzero(y >= 0)) == valid_rows * 8
    layout = NamedSharding(mesh, P(None, "data", None))
    x = jax.device_put(x.reshape(4, 16, 8), layout)
    y = jax.device_put(y.reshape(4, 16, 8), layout)
    reference = nnx.jit(gradients_for_mlm_microbatches)(model, x, y)
    actual = nnx.jit(
        lambda m, a, b: gradients_for_local_mlm_microbatches(
            m, a, b, mesh, padded_rows=True
        )
    )(model, x, y)
    np.testing.assert_allclose(
        float(actual[0]), float(reference[0]), rtol=0.002, atol=0.002
    )
    for expected, observed in zip(
        jax.tree.leaves(reference[1]), jax.tree.leaves(actual[1]), strict=True
    ):
        a, b = (
            np.asarray(expected, dtype=np.float32),
            np.asarray(observed, dtype=np.float32),
        )
        assert np.isfinite(b).all()
        assert np.linalg.norm(a - b) <= 0.02 * np.linalg.norm(a) + 2e-5
    assert jnp.isfinite(actual[0])

    if valid_rows == 63:
        # A full batch must retain the optimized local path, exactly.
        from flaxchat.training import _gradients_for_local_mlm_microbatches

        full_x, full_y = mask_prepared_rows(
            tokens,
            np.arange(64),
            pad_token_id=0,
            allow_padding=True,
            seed=9,
            step=100,
            vocab_size=64,
            mask_token_id=4,
            special_token_ids=[0, 4],
            probability=1.0,
        )
        full_x = jax.device_put(full_x.reshape(4, 16, 8), layout)
        full_y = jax.device_put(full_y.reshape(4, 16, 8), layout)
        optimized = nnx.jit(
            lambda m, a, b: _gradients_for_local_mlm_microbatches(m, a, b, mesh)
        )(model, full_x, full_y)
        routed = nnx.jit(
            lambda m, a, b: gradients_for_local_mlm_microbatches(m, a, b, mesh)
        )(model, full_x, full_y)
        for original, current in zip(
            jax.tree.leaves(optimized), jax.tree.leaves(routed), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(original), np.asarray(current))
