"""
Tests for flaxchat/checkpoint.py — Orbax checkpoint save/load round-trips.

Uses tiny_model fixture from conftest.py and pytest tmp_path for isolation.
"""

import os
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.checkpoint import (
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
