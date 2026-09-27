"""Physical TPU regression checks for encoder state placement."""
import jax
from flax import nnx
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from flaxchat.encoder import EncoderConfig, ModernBert, import_hf_weights
from flaxchat.training import initialize_sharded


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_pretrained_import_preserves_tpu_shards_and_is_atomic(factor):
    if jax.default_backend() != 'tpu' or jax.device_count() != 4 or jax.process_count() != 1:
        pytest.skip('Requires one physical four-device TPU host')
    mesh = Mesh(np.asarray(jax.devices()).reshape(-1, factor), ('data', 'fsdp'))
    config = EncoderConfig(vocab_size=128, hidden_size=32, intermediate_size=48,
                           num_hidden_layers=2, num_attention_heads=2,
                           max_position_embeddings=128, local_attention=16)
    model = initialize_sharded(lambda: ModernBert(config, rngs=nnx.Rngs(7)), mesh, fsdp=factor)
    # Explicit checkpoint names and destinations also exercise transposed kernels.
    destinations = {
        'model.embeddings.tok_embeddings.weight': (model.embedding.embedding, False),
        'model.embeddings.norm.weight': (model.embedding_norm.scale, False),
        'model.final_norm.weight': (model.final_norm.scale, False),
        'head.dense.weight': (model.head_dense.kernel, True),
        'head.norm.weight': (model.head_norm.scale, False),
        'decoder.bias': (model.decoder_bias, False),
    }
    for index, layer in enumerate(model.layers):
        prefix = f'model.layers.{index}.'
        for name, variable, transpose in (
            ('attn.Wqkv.weight', layer.qkv.kernel, True),
            ('attn.Wo.weight', layer.attn_out.kernel, True),
            ('mlp.Wi.weight', layer.wi.kernel, True),
            ('mlp.Wo.weight', layer.wo.kernel, True),
            ('mlp_norm.weight', layer.mlp_norm.scale, False),
        ):
            destinations[prefix + name] = variable, transpose
        if layer.attn_norm is not None:
            destinations[prefix + 'attn_norm.weight'] = layer.attn_norm.scale, False
    tensors = {}
    for index, (name, (variable, transpose)) in enumerate(destinations.items()):
        values = (np.arange(np.prod(variable.shape), dtype=np.float32).reshape(variable.shape) + index) / 128
        tensors[name] = values.T.copy() if transpose else values
    tensors['decoder.weight'] = tensors['model.embeddings.tok_embeddings.weight'].copy()
    import_hf_weights(model, tensors)
    for name, (variable, transpose) in destinations.items():
        value = variable[...]
        spec = P('fsdp') if factor > 1 and value.ndim >= 2 and value.shape[0] % factor == 0 else P()
        assert value.sharding.is_equivalent_to(NamedSharding(mesh, spec), value.ndim)
        assert len(value.addressable_shards) == 4
        expected = tensors[name].T if transpose else tensors[name]
        for shard in value.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])
    before = [np.asarray(value).copy() for value in jax.tree.leaves(nnx.state(model))]
    malformed = dict(tensors)
    last_name = next(reversed(destinations))
    malformed[last_name] = np.full_like(tensors[last_name], np.nan)
    with pytest.raises(ValueError, match='Invalid checkpoint tensor'):
        import_hf_weights(model, malformed)
    for expected, actual in zip(before, jax.tree.leaves(nnx.state(model)), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), expected)
