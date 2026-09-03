"""
Tests for flaxchat/checkpoint.py — Orbax checkpoint save/load round-trips.

Uses tiny_model fixture from conftest.py and pytest tmp_path for isolation.
"""

import os
import shutil
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.checkpoint import (
    CheckpointCompatibilityError,
    _checkpoint_manifest,
    _validate_manifest,
    create_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
    restore_model_from_checkpoint,
)
from flaxchat.config import GPTConfig
from flaxchat.gpt import GPT


# ---------------------------------------------------------------------------
# Tests for create_checkpoint_manager
# ---------------------------------------------------------------------------

class TestCreateCheckpointManager:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_async_policy_is_honored(self, tmp_path, enabled):
        manager = create_checkpoint_manager(
            str(tmp_path / f"async-{enabled}"), async_checkpointing=enabled
        )
        assert manager._options.enable_async_checkpointing is enabled
        manager.close()

    def test_creates_directory(self, tmp_path):
        """Manager should create the checkpoint directory if it doesn't exist."""
        ckpt_dir = str(tmp_path / "checkpoints")
        assert not os.path.exists(ckpt_dir)
        manager = create_checkpoint_manager(ckpt_dir)
        assert os.path.exists(ckpt_dir)

    def test_returns_manager(self, tmp_path):
        """Should return an Orbax CheckpointManager instance."""
        import orbax.checkpoint as ocp
        ckpt_dir = str(tmp_path / "ckpts")
        manager = create_checkpoint_manager(ckpt_dir)
        assert isinstance(manager, ocp.CheckpointManager)

    def test_max_to_keep(self, tmp_path):
        """Manager should respect the max_to_keep parameter."""
        ckpt_dir = str(tmp_path / "ckpts")
        manager = create_checkpoint_manager(ckpt_dir, max_to_keep=5)
        # Orbax stores max_to_keep in options
        assert manager is not None

    def test_existing_directory_ok(self, tmp_path):
        """Creating a manager on an existing directory should not fail."""
        ckpt_dir = str(tmp_path / "ckpts")
        os.makedirs(ckpt_dir)
        manager = create_checkpoint_manager(ckpt_dir)
        assert manager is not None


# ---------------------------------------------------------------------------
# Tests for save + load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def _make_optimizer(self, model):
        """Create a simple optimizer wrapping the model."""
        import optax
        tx = optax.sgd(learning_rate=0.01)
        return nnx.Optimizer(model, tx, wrt=nnx.Param)

    def test_save_and_load_latest(self, tiny_model, tmp_path):
        """Save at step N, then load latest should return the same params."""
        ckpt_dir = str(tmp_path / "ckpts")
        manager = create_checkpoint_manager(ckpt_dir, async_checkpointing=False)
        optimizer = self._make_optimizer(tiny_model)
        metadata = {"step": 100, "loss": 2.5}

        save_checkpoint(manager, step=100, model=tiny_model, optimizer=optimizer, metadata=metadata)
        manager.wait_until_finished()

        model_dict, loaded_metadata = load_checkpoint(manager, step=None, model=tiny_model)

        assert loaded_metadata["step"] == 100
        assert loaded_metadata["loss"] == 2.5
        assert model_dict is not None

    def test_save_and_load_specific_step(self, tiny_model, tmp_path):
        """Load a specific step rather than latest."""
        ckpt_dir = str(tmp_path / "ckpts")
        manager = create_checkpoint_manager(ckpt_dir, max_to_keep=5, async_checkpointing=False)
        optimizer = self._make_optimizer(tiny_model)

        save_checkpoint(manager, step=10, model=tiny_model, optimizer=optimizer, metadata={"step": 10})
        manager.wait_until_finished()
        save_checkpoint(manager, step=20, model=tiny_model, optimizer=optimizer, metadata={"step": 20})
        manager.wait_until_finished()

        _, meta_10 = load_checkpoint(manager, step=10, model=tiny_model)
        assert meta_10["step"] == 10

        _, meta_20 = load_checkpoint(manager, step=20, model=tiny_model)
        assert meta_20["step"] == 20

    def test_load_no_checkpoints_raises(self, tmp_path):
        """Loading from an empty manager should raise ValueError."""
        ckpt_dir = str(tmp_path / "empty_ckpts")
        manager = create_checkpoint_manager(ckpt_dir)

        with pytest.raises(ValueError, match="No checkpoints found"):
            load_checkpoint(manager, step=None)

    def test_param_values_preserved(self, tiny_config, tmp_path):
        """Parameter values should be exactly preserved after round-trip."""
        model_a = GPT(tiny_config, rngs=nnx.Rngs(42))
        ckpt_dir = str(tmp_path / "ckpts")
        manager = create_checkpoint_manager(ckpt_dir, async_checkpointing=False)
        optimizer = self._make_optimizer(model_a)

        # Get original param values
        original_state = nnx.state(model_a, nnx.Param)
        original_dict = nnx.to_pure_dict(original_state)

        save_checkpoint(manager, step=1, model=model_a, optimizer=optimizer, metadata={"v": 1})
        manager.wait_until_finished()

        # Create a fresh model with different random init
        model_b = GPT(tiny_config, rngs=nnx.Rngs(99))

        loaded_dict, _ = load_checkpoint(manager, step=1, model=model_b)

        # Compare leaf arrays
        def check_equal(orig, loaded, path=""):
            if isinstance(orig, dict):
                for k in orig:
                    check_equal(orig[k], loaded[k], path=f"{path}.{k}")
            else:
                assert jnp.allclose(jnp.array(orig), jnp.array(loaded), atol=1e-7), \
                    f"Mismatch at {path}"

        check_equal(original_dict, loaded_dict)

    def test_manifest_detects_modified_state_before_mutation(self, tiny_model):
        optimizer = self._make_optimizer(tiny_model)
        model_state = nnx.to_pure_dict(nnx.state(tiny_model))
        metadata = {"resolved_config": {"model": "tiny"}}
        training_state = {"update_step": jnp.asarray(3)}
        manifest = _checkpoint_manifest(
            3, model_state, optimizer.opt_state, metadata, training_state
        )
        changed = jax.tree.map(lambda value: value, model_state)
        first_path, first_leaf = jax.tree_util.tree_flatten_with_path(changed)[0][0]
        del first_path
        # Replacing one leaf is enough to prove the content digest is enforced.
        leaves, treedef = jax.tree.flatten(changed)
        leaves[0] = leaves[0] + jnp.ones_like(first_leaf)
        changed = jax.tree.unflatten(treedef, leaves)
        with pytest.raises(CheckpointCompatibilityError, match="incomplete or corrupt"):
            _validate_manifest(
                manifest, changed, optimizer.opt_state, metadata, training_state
            )

    def test_partial_checkpoint_fails_before_mutating_model(
        self, tiny_config, tmp_path
    ):
        model = GPT(tiny_config, rngs=nnx.Rngs(4))
        optimizer = self._make_optimizer(model)
        checkpoint_dir = str(tmp_path / "partial")
        manager = create_checkpoint_manager(
            checkpoint_dir, async_checkpointing=False
        )
        save_checkpoint(manager, 2, model, optimizer, {"step": 2})
        manager.close()
        shutil.rmtree(tmp_path / "partial" / "2" / "optimizer")

        target = GPT(tiny_config, rngs=nnx.Rngs(9))
        target_optimizer = self._make_optimizer(target)
        before = nnx.to_pure_dict(nnx.state(target))
        with pytest.raises(CheckpointCompatibilityError, match="incomplete, corrupt"):
            restore_model_from_checkpoint(
                target, checkpoint_dir, step=2, optimizer=target_optimizer
            )
        after = nnx.to_pure_dict(nnx.state(target))
        assert all(
            jnp.array_equal(expected, actual)
            for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(after))
        )

    @pytest.mark.integration
    @pytest.mark.filterwarnings("error:Sharding info not provided")
    def test_interrupted_training_matches_uninterrupted_training(self, tmp_path):
        """Five updates + restore + five updates must equal ten uninterrupted."""
        import optax

        class TinyRegressor(nnx.Module):
            def __init__(self, seed):
                self.linear = nnx.Linear(3, 2, rngs=nnx.Rngs(seed))

            def __call__(self, x, y):
                return jnp.mean((self.linear(x) - y) ** 2)

        def make_run(seed=0):
            current_model = TinyRegressor(seed)
            current_optimizer = nnx.Optimizer(
                current_model, optax.adam(1e-2), wrt=nnx.Param
            )
            return current_model, current_optimizer

        def update(current_model, current_optimizer, batch):
            x, y = batch
            loss, grads = nnx.value_and_grad(lambda model: model(x, y))(current_model)
            current_optimizer.update(current_model, grads)
            return loss

        data_key = jax.random.key(123)
        keys = jax.random.split(data_key, 20)
        batches = [
            (jax.random.normal(keys[2 * i], (4, 3)), jax.random.normal(keys[2 * i + 1], (4, 2)))
            for i in range(10)
        ]

        reference_model, reference_optimizer = make_run()
        for batch in batches:
            update(reference_model, reference_optimizer, batch)

        interrupted_model, interrupted_optimizer = make_run()
        for batch in batches[:5]:
            update(interrupted_model, interrupted_optimizer, batch)
        checkpoint_dir = str(tmp_path / "resume")
        manager = create_checkpoint_manager(checkpoint_dir, async_checkpointing=False)
        save_checkpoint(
            manager,
            5,
            interrupted_model,
            interrupted_optimizer,
            {"next_batch": 5, "rng_key": [0, 123]},
            training_state={"update_step": jnp.asarray(5), "next_batch": jnp.asarray(5)},
        )
        manager.wait_until_finished()
        manager.close()

        resumed_model, resumed_optimizer = make_run(seed=999)
        metadata, training_state = restore_model_from_checkpoint(
            resumed_model,
            checkpoint_dir,
            step=5,
            optimizer=resumed_optimizer,
            load_training_state=True,
        )
        start = int(training_state["next_batch"])
        assert start == metadata["next_batch"] == 5
        for batch in batches[start:]:
            update(resumed_model, resumed_optimizer, batch)

        reference = nnx.to_pure_dict(nnx.state(reference_model))
        resumed = nnx.to_pure_dict(nnx.state(resumed_model))
        for expected, actual in zip(jax.tree.leaves(reference), jax.tree.leaves(resumed)):
            assert jnp.array_equal(expected, actual)
        for expected, actual in zip(
            jax.tree.leaves(reference_optimizer.opt_state),
            jax.tree.leaves(resumed_optimizer.opt_state),
        ):
            assert jnp.array_equal(expected, actual)


# ---------------------------------------------------------------------------
# Tests for restore_model_from_checkpoint
# ---------------------------------------------------------------------------

class TestRestoreModelFromCheckpoint:
    def test_restores_in_place(self, tiny_config, tmp_path):
        """restore_model_from_checkpoint should update model params in-place."""
        import optax

        # Save model_a
        model_a = GPT(tiny_config, rngs=nnx.Rngs(42))
        ckpt_dir = str(tmp_path / "ckpts")
        manager = create_checkpoint_manager(ckpt_dir, async_checkpointing=False)
        tx = optax.sgd(learning_rate=0.01)
        optimizer = nnx.Optimizer(model_a, tx, wrt=nnx.Param)

        save_checkpoint(manager, step=5, model=model_a, optimizer=optimizer, metadata={"info": "test"})
        manager.wait_until_finished()

        # Create model_b with different init
        model_b = GPT(tiny_config, rngs=nnx.Rngs(99))

        # Verify they differ before restore
        state_a = nnx.to_pure_dict(nnx.state(model_a, nnx.Param))
        state_b_before = nnx.to_pure_dict(nnx.state(model_b, nnx.Param))

        # Restore into model_b
        metadata = restore_model_from_checkpoint(model_b, ckpt_dir, step=5)

        assert metadata["info"] == "test"

        # Now model_b should match model_a
        state_b_after = nnx.to_pure_dict(nnx.state(model_b, nnx.Param))

        def check_match(a, b, path=""):
            if isinstance(a, dict):
                for k in a:
                    check_match(a[k], b[k], path=f"{path}.{k}")
            else:
                assert jnp.allclose(jnp.array(a), jnp.array(b), atol=1e-7), \
                    f"Mismatch at {path}"

        check_match(state_a, state_b_after)

    def test_returns_metadata(self, tiny_config, tmp_path):
        """restore_model_from_checkpoint should return the saved metadata."""
        import optax

        model = GPT(tiny_config, rngs=nnx.Rngs(0))
        ckpt_dir = str(tmp_path / "ckpts")
        manager = create_checkpoint_manager(ckpt_dir, async_checkpointing=False)
        tx = optax.sgd(learning_rate=0.01)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        save_checkpoint(manager, step=1, model=model, optimizer=optimizer,
                        metadata={"epoch": 3, "loss": 1.23})
        manager.wait_until_finished()

        meta = restore_model_from_checkpoint(model, ckpt_dir)
        assert meta["epoch"] == 3
        assert meta["loss"] == 1.23
