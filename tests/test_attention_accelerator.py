import jax
import jax.numpy as jnp
import pytest

from flaxchat.gpt import exact_attention


pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU required"),
]


@pytest.mark.parametrize("window_left", [32, 128])
def test_splash_matches_xla_forward_and_gradient(window_left):
    q, k, v = jax.random.normal(
        jax.random.key(7), (3, 1, 128, 2, 16), dtype=jnp.bfloat16
    )

    def objective(query, backend):
        output = exact_attention(query, k, v, window_left=window_left, backend=backend)
        return jnp.mean(output.astype(jnp.float32) ** 2), output

    (xla_loss, xla_output), xla_grad = jax.value_and_grad(objective, has_aux=True)(q, "xla")
    (splash_loss, splash_output), splash_grad = jax.value_and_grad(objective, has_aux=True)(q, "splash")

    assert jnp.allclose(splash_output, xla_output, atol=3e-2, rtol=3e-2)
    assert jnp.allclose(splash_loss, xla_loss, atol=1e-3, rtol=1e-2)
    assert jnp.allclose(splash_grad, xla_grad, atol=3e-3, rtol=5e-2)
