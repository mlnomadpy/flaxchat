"""Manager lifecycle regression; execute factory code without JAX/model imports."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / 'flaxchat/checkpoint.py'


def actual_function(name, environment):
    tree = ast.parse(SOURCE.read_text())
    # The restore function has overload signatures; select the implementation.
    node = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name][-1]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE), 'exec'), environment)
    return environment[name]


@pytest.mark.parametrize('read_only', [False, True])
def test_gcs_factory_preserves_nested_committed_best_namespace(read_only):
    objects = {'best/0/commit_success.txt', 'best/0/model/data', '500/commit_success.txt'}
    captured = {}

    def manager(directory, options):
        captured.update(directory=directory, options=options)
        # Reproduce Orbax0.12.4 GCS cleanup: best/ lacks its own commit marker,
        # so generic cleanup would recursively remove its committed children.
        if options.cleanup_tmp_directories:
            for child in {name.split('/')[0] for name in objects.copy()}:
                if child + '/commit_success.txt' not in objects:
                    objects.difference_update(name for name in objects.copy() if name.startswith(child + '/'))
        return captured

    factory = actual_function('create_checkpoint_manager', {'os': os, 'ocp': SimpleNamespace(
        CheckpointManagerOptions=lambda **kwargs: SimpleNamespace(**kwargs), CheckpointManager=manager)})
    factory('gs://bucket/checkpoints', read_only=read_only, async_checkpointing=False)
    assert 'best/0/model/data' in objects
    assert captured['options'].cleanup_tmp_directories is False
    assert captured['options'].read_only is read_only
    assert captured['options'].create is (not read_only)


def test_read_only_local_factory_does_not_create_missing_directory(tmp_path):
    factory = actual_function('create_checkpoint_manager', {'os': os, 'ocp': SimpleNamespace(
        CheckpointManagerOptions=lambda **kwargs: SimpleNamespace(**kwargs), CheckpointManager=lambda **kwargs: kwargs)})
    missing = tmp_path / 'missing'
    result = factory(str(missing), read_only=True)
    assert not missing.exists()
    assert result['options'].create is False


@pytest.mark.parametrize('name', ['load_checkpoint_metadata', 'restore_model_from_checkpoint'])
def test_checkpoint_read_paths_request_read_only_manager(name):
    class StopBeforeModel(Exception):
        pass

    captured = {}

    def manager(*args, **kwargs):
        captured.update(kwargs)
        raise StopBeforeModel

    function = actual_function(name, {'create_checkpoint_manager': manager})
    with pytest.raises(StopBeforeModel):
        if name == 'load_checkpoint_metadata':
            function('gs://bucket/checkpoints', 500)
        else:
            function(None, 'gs://bucket/checkpoints', step=500)
    assert captured['read_only'] is True
    assert captured['async_checkpointing'] is False
