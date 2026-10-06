"""Cloud campaign wiring only: never initialize JAX or call providers."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import qualify_representation_v2_tpu as qualification
from scripts.representation_v2_cloud_campaign import stage_args
from flaxchat.embedding_stage import add_stage_arguments


def test_stage_arguments_match_real_parser_and_portable_locations():
    parser = argparse.ArgumentParser()
    add_stage_arguments(parser)
    args = parser.parse_args(stage_args('/worker/run', 'gs://bucket/stage/checkpoints'))
    assert args.parent_public == '/worker/run/parent'
    assert args.parent_manifest == '/worker/run/training-data/development/parent-files.json'
    assert args.steps == 20000
    assert args.encoder_chunk_size == 0
    assert args.weight_quantization == 'none'
    assert len(args.data) == 11
    assert len(args.retrieval_dev) == 3


def test_parent_import_leaves_fresh_directory_for_suite_and_uploads_evidence(tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({'output_prefix': 'gs://bucket/run/stage'}))
    calls = []
    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if 'scripts.validate_production_parent_tpu' in argv:
            Path(argv[argv.index('--output') + 1]).write_text(json.dumps({
                'status': 'passed', 'weight_import_qualified': True}))
        if 'scripts.validate_test_suite' in argv:
            target = Path(argv[argv.index('--output') + 1])
            assert not target.exists()
            target.mkdir(exist_ok=False)
    with patch.object(qualification, 'run_logged', side_effect=fake_run):
        qualification.main(['--root', str(tmp_path)])
    assert len(calls) == 3
    assert calls[1][0][:3] == ['gcloud', 'storage', 'cp']
    assert calls[1][0][-1] == 'gs://bucket/run/stage/qualification/parent-qualification.json'
    assert calls[0][1]['timeout'] + calls[1][1]['timeout'] + calls[2][1]['timeout'] <= 1440


def test_failed_actual_parent_import_blocks_suite(tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({'output_prefix': 'gs://bucket/run/stage'}))
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        if '--output' in argv:
            Path(argv[argv.index('--output') + 1]).write_text(json.dumps({
                'status': 'failed', 'weight_import_qualified': False}))
    with patch.object(qualification, 'run_logged', side_effect=fake_run):
        with pytest.raises(ValueError, match='parent restoration'):
            qualification.main(['--root', str(tmp_path)])
    assert len(calls) == 2
    assert calls[1][:3] == ['gcloud', 'storage', 'cp']


def test_real_subprocess_failure_keeps_traceback(tmp_path):
    import subprocess
    import sys
    from scripts.representation_v2_cloud_campaign import run_logged
    log = tmp_path / 'failure.log'
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_logged([sys.executable, '-c', 'raise ValueError("raw source broken")'], 5, log)
    assert 'ValueError: raw source broken' in log.read_text()
    assert 'ValueError: raw source broken' in failure.value.output


def test_capture_keeps_json_stdout_separate_from_warning(tmp_path):
    import sys
    from scripts.representation_v2_cloud_campaign import run_logged
    result = run_logged([sys.executable, '-c',
        'import sys; print("warning", file=sys.stderr); print(123)'],
        5, tmp_path / 'capture.log', capture=True)
    assert json.loads(result.stdout) == 123
    assert 'warning' in (tmp_path / 'capture.log').read_text()


def test_timeout_kills_descendants_before_they_can_write(tmp_path):
    import subprocess
    import sys
    import time
    from scripts.representation_v2_cloud_campaign import run_logged
    sentinel = tmp_path / 'orphan-wrote'
    child = f'import time; from pathlib import Path; time.sleep(1); Path({str(sentinel)!r}).touch()'
    parent = f'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", {child!r}]); print("spawned", flush=True); time.sleep(20)'
    log = tmp_path / 'timeout.log'
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged([sys.executable, '-c', parent], .4, log)
    time.sleep(1.1)
    assert 'spawned' in log.read_text()
    assert not sentinel.exists()


def test_bundle_is_self_contained_for_linked_mixture(tmp_path):
    import tarfile
    from scripts.representation_v2_cloud_campaign import bundle_training_data
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'development').mkdir()
    (data / 'prepared').mkdir()
    (data / 'prepared' / 'tokens').write_bytes(b'1234')
    (data / 'mixture').mkdir()
    (data / 'mixture' / 'tokens').symlink_to('../prepared/tokens')
    for name in ('weights.json', 'registry.json', 'heldout-inventory.json',
                 'final-quarantine.json', 'terminal-data.json'):
        (data / name).write_text('{}')
    bundle = tmp_path / 'data.tar.gz'
    bundle_training_data(data, bundle)
    with tarfile.open(bundle) as archive:
        assert all(not member.issym() for member in archive)
        assert archive.extractfile('mixture/tokens').read() == b'1234'


def test_failed_parent_process_uploads_log_without_masking_root_error(tmp_path):
    import subprocess
    (tmp_path / 'run.json').write_text(json.dumps({'output_prefix': 'gs://bucket/stage'}))
    calls = []
    def fail(argv, **kwargs):
        calls.append(argv)
        Path(kwargs['log_path']).write_text('actual failure')
        if 'scripts.validate_production_parent_tpu' in argv:
            raise subprocess.CalledProcessError(23, argv)
        raise subprocess.CalledProcessError(1, argv)
    with patch.object(qualification, 'run_logged', side_effect=fail):
        with pytest.raises(subprocess.CalledProcessError) as error:
            qualification.main(['--root', str(tmp_path)])
    assert error.value.returncode == 23
    assert len(calls) == 2
    assert calls[1][-1].endswith('/parent-qualification.log')
    assert (tmp_path / 'parent-qualification-upload-errors.json').exists()


def test_prepared_data_reuse_authenticates_recipe_and_all_source_manifests(tmp_path):
    from scripts.prepare_representation_v2_training import WEIGHTS
    from scripts.representation_v2_cloud_campaign import sha, validate_prepared_data
    source = tmp_path / 'source'
    (source / 'scripts').mkdir(parents=True)
    recipe = source / 'scripts/prepare_representation_v2_training.py'
    recipe.write_text('frozen recipe')
    data = tmp_path / 'data'
    data.mkdir()
    quarantine = {'status': 'passed', 'exact_overlaps': 0, 'sources': {}}
    for name in WEIGHTS:
        folder = data / 'mixture' / name
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text('{}')
        quarantine['sources'][name] = {'train_rows': 3, 'manifest_sha256': sha(folder / 'manifest.json')}
    (data / 'final-quarantine.json').write_text(json.dumps(quarantine))
    (data / 'weights.json').write_text(json.dumps(WEIGHTS))
    spec = {'portable': {'sha256': 'portable'}, 'heldout': {'sha256': 'heldout'}}
    terminal = {'status': 'passed', 'model_execution': False, 'portable_sha256': 'portable',
        'original_heldout_sha256': 'heldout', 'source_sha256': sha(recipe),
        'final_quarantine_sha256': sha(data / 'final-quarantine.json')}
    (data / 'terminal-data.json').write_text(json.dumps(terminal))
    assert validate_prepared_data(data, spec, source) == terminal
    recipe.write_text('new recipe')
    with pytest.raises(ValueError, match='this recipe'):
        validate_prepared_data(data, spec, source)
    recipe.write_text('frozen recipe')
    (data / 'mixture/marco_replay/manifest.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='manifest changed'):
        validate_prepared_data(data, spec, source)
