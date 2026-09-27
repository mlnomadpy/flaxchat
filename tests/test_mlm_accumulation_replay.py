from dataclasses import asdict

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import pytest

from flaxchat.encoder import EncoderConfig, ModernBert
from scripts.replay_mlm_accumulation import (
    restore_parameters,
    replay,
    compare_gradients,
)
from scripts.numerical_evidence import write_numerical_evidence


def fixture():
    config = EncoderConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=1,
        compute_dtype="float32",
        residual_dtype="float32",
    )
    model = ModernBert(config, rngs=nnx.Rngs(17))
    pairs = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]
    arrays = {f"parameter_{i}": np.asarray(v) for i, (_, v) in enumerate(pairs)}
    paths = {
        f"parameter_{i}": jax.tree_util.keystr(k) for i, (k, _) in enumerate(pairs)
    }
    return config, model, arrays, paths


@pytest.mark.parametrize("corruption", ["path", "shape", "dtype", "missing"])
def test_restore_rejects_nonmatching_parameter_identity(corruption):
    _, model, arrays, paths = fixture()
    if corruption == "path":
        paths["parameter_0"] = "untrusted path string"
    if corruption == "shape":
        arrays["parameter_0"] = np.zeros(1, np.float32)
    if corruption == "dtype":
        arrays["parameter_0"] = arrays["parameter_0"].astype(np.float64)
    if corruption == "missing":
        del paths["parameter_0"]
    with pytest.raises(ValueError, match="parameter"):
        restore_parameters(model, arrays, paths)


@pytest.mark.parametrize("replay_kind", ["per_example", "distributed", "corrupted", "infinite_loss"])
def test_saved_weights_replace_seed_and_replay(tmp_path, monkeypatch, replay_kind):
    from flaxchat.training import gradients_for_mlm_microbatches
    from flaxchat.runtime import runtime_identity

    config, model, arrays, paths = fixture()
    x = jnp.array([[[5, 6, 0]], [[6, 7, 0]]])
    y = jnp.array([[[5, -1, -1]], [[6, 7, -1]]])
    loss, grads = nnx.jit(gradients_for_mlm_microbatches)(model, x, y)
    arrays.update(
        inputs=np.asarray(x),
        targets=np.asarray(y),
        reference_loss=np.asarray(loss),
        candidate_loss=np.asarray(loss),
    )
    for prefix in ["reference", "candidate"]:
        for i, (key, value) in enumerate(
            jax.tree_util.tree_flatten_with_path(grads)[0]
        ):
            arrays[f"{prefix}_{i}"] = value
            paths[f"{prefix}_{i}"] = jax.tree_util.keystr(key)
    write_numerical_evidence(
        tmp_path / "capture",
        arrays,
        metadata=dict(
            config=asdict(config),
            leaf_paths=paths,
            devices=[str(d) for d in jax.devices()],
            backend=jax.default_backend(),
            runtime=runtime_identity(),
        ),
    )
    if replay_kind in ("corrupted", "infinite_loss"):
        from scripts import replay_mlm_accumulation as module
        original = module.gradients_for_local_mlm_microbatches
        def corrupted(*args, **kwargs):
            value, grads = original(*args, **kwargs)
            if replay_kind == "infinite_loss":
                return jnp.float32(jnp.inf), grads
            return value, jax.tree.map(lambda v: v + 1, grads)
        monkeypatch.setattr(module, "gradients_for_local_mlm_microbatches", corrupted)
    result = replay(tmp_path / "capture", per_example_only=replay_kind == "per_example",
                    boundary_mode="attention_output")
    assert result["boundary_mode"] == "attention_output"
    if replay_kind != "per_example":
        gates = result["replayed_reference_gates"]
        assert gates["reference"]["gradients_passed"]
        assert gates["candidate"]["gradients_passed"] == (replay_kind != "corrupted")
        assert gates["candidate"]["loss_passed"] == (replay_kind != "infinite_loss")
        if replay_kind == "infinite_loss":
            import json
            assert result["losses"]["candidate"] is None
            assert not result["loss_finite"]["candidate"]
            json.dumps(result, allow_nan=False)
        assert "candidate_replay_vs_per_example" in result["comparisons"]
    else:
        assert not result["replayed_reference_gates"]
    assert result["masked_targets"] == 3
    assert result["losses"]["per_example"] == pytest.approx(float(loss), abs=1e-6)
    assert all(
        row["absolute_max"] < 2e-6
        for row in result["comparisons"]["reference_vs_per_example"]
    )
    target = ModernBert(config, rngs=nnx.Rngs(99))
    restore_parameters(target, arrays, paths)
    for i, value in enumerate(jax.tree.leaves(nnx.state(target, nnx.Param))):
        np.testing.assert_array_equal(value, arrays[f"parameter_{i}"])


def test_nonfinite_gradients_remain_visible():
    result = compare_gradients({"x": np.array([1.0])}, {"x": np.array([np.nan])})
    assert result[0]["finite"] is False and result[0]["relative_l2"] is None
    with pytest.raises(ValueError, match="paths"):
        compare_gradients({"x": np.ones(1)}, {"y": np.ones(1)})


@pytest.mark.parametrize('passed', [True, False])
def test_cli_required_gates_write_report_before_exit(tmp_path, monkeypatch, passed):
    import json
    import sys
    from scripts import replay_mlm_accumulation as module
    report = {'replayed_reference_gates': {
        name: {'gradients_passed': passed, 'loss_passed': True}
        for name in ['reference', 'candidate']}}
    monkeypatch.setattr(module, 'replay', lambda *a, **kw: report)
    output = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['replay', 'capture', '--output', str(output),
                                     '--require-reference-gates'])
    if passed:
        module.main()
    else:
        with pytest.raises(SystemExit) as exc:
            module.main()
        assert exc.value.code == 1
    assert json.loads(output.read_text()) == report


def test_cli_rejects_required_gates_without_distributed_replay(tmp_path, monkeypatch):
    import sys
    from scripts import replay_mlm_accumulation as module
    monkeypatch.setattr(sys, 'argv', ['replay', 'capture', '--output', str(tmp_path / 'out'),
                                     '--require-reference-gates', '--per-example-only'])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not (tmp_path / 'out').exists()
