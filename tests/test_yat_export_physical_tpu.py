"""Physical TPU export verification; never run numerical model code on CPU."""
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import pytest


def test_yat_checkpoint_export_authenticates_then_excludes_objective_alpha(tmp_path, monkeypatch):
    import jax
    if jax.default_backend() != 'tpu':
        pytest.skip('Requires physical TPU')
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
    from flaxchat.embedding_stage import contrastive_objective_identity
    from flaxchat.encoder import EncoderConfig, ModernBert
    from flaxchat.public_encoder import load_public_encoder, _named_leaves
    from scripts import export_public_encoder
    from scripts.release_contract import digest, validate_export

    config = EncoderConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
                           num_hidden_layers=1, num_attention_heads=2,
                           max_position_embeddings=16, compute_dtype='float32', use_remat=False)
    model = ModernBert(config, rngs=nnx.Rngs(7))
    model.contrastive_raw_alpha = nnx.Param(jnp.asarray(-4.6, jnp.float32))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}')
    metadata = dict(model_family='yat_embedding_finetune', tokenizer_identity=digest(tokenizer),
                    source_python_sha256='a' * 64,
                    resolved_config=dict(encoder=asdict(config), contrastive_objective=contrastive_objective_identity(
                        SimpleNamespace(contrastive_similarity='yat', yat_infonce_alpha_init=.01))))
    checkpoint = tmp_path / 'checkpoint'
    with create_checkpoint_manager(str(checkpoint), async_checkpointing=False) as manager:
        save_checkpoint(manager, 2, model, optimizer, metadata)
        manager.wait_until_finished()
    root = checkpoint / '2'
    output = tmp_path / 'public'
    argv = ['export_public_encoder', str(root / 'model'), str(root / 'metadata' / 'metadata'),
            str(root / 'manifest' / 'metadata'), str(tokenizer), str(output),
            '--model-family', 'yat_embedding_finetune', '--step', '2']
    monkeypatch.setattr(sys, 'argv', argv)
    export_public_encoder.main()
    report = json.loads((output / 'export.json').read_text())
    validate_export(output, report)
    restored = load_public_encoder(output)
    assert not hasattr(restored, 'contrastive_raw_alpha')
    expected_count = len(_named_leaves(nnx.to_pure_dict(nnx.state(model)))) - 1
    assert report['tensors'] == expected_count
    assert len(report['excluded_training_only_tensors']) == 1
    tokens = jnp.asarray([[2, 5, 3, 0], [2, 6, 3, 0]], jnp.int32)
    np.testing.assert_array_equal(np.asarray(restored.pool(tokens)), np.asarray(model.pool(tokens)))

    # Even the omitted scalar's bytes must authenticate before a public file is written.
    manifest_file = root / 'manifest' / 'metadata'
    manifest = json.loads(manifest_file.read_text())
    manifest['model_state']["['contrastive_raw_alpha']"]['sha256'] = 'b' * 64
    manifest_file.write_text(json.dumps(manifest))
    bad_output = tmp_path / 'bad-public'
    monkeypatch.setattr(sys, 'argv', argv[:5] + [str(bad_output)] + argv[6:])
    with pytest.raises(ValueError, match='full model state'):
        export_public_encoder.main()
    assert not (bad_output / 'model.safetensors').exists()
