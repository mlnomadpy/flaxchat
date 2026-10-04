"""Deployment admission contracts; no CPU model or cloud execution."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from scripts import representation_run as runner


def fixture():
    return dict(schema_version=1, run_id='campaign', stage_id='stage', parent_identity='parent-sha',
        cleanup_owner='workflow', expected_topology='v5litepod-16', expected_device_count=16, expected_process_count=4,
        runtime_lock=dict(path='lock.txt', sha256='b'*64, wheelhouse_target='wheels'),
        source=dict(uri='gs://bucket/source', sha256='a'*64),
        artifacts=[dict(uri='gs://bucket/wheels', sha256='c'*64, target='wheels', archive=True)],
        output_prefix='gs://bucket/campaign/stage', setup_checks=['{python}', '-m', 'scripts.validate_test_suite'],
        workload=['{python}', '-m', 'scripts.train_yat_embedding_finetune', '--output', '{checkpoint_output}'],
        qualification=dict(receipt='qualification.json', required_tests=['loss', 'recovery']))


class AdmissionTests(unittest.TestCase):
    def test_region_change_requires_matching_new_pricing_evidence(self):
        value = fixture()
        value['deployment'] = dict(project='project', zone='us-east5-a',
            accelerator_type='v5litepod-16', runtime_version='v2-alpha-tpuv5-lite',
            hourly_usd=8, budget_usd=20, provisioning_model='spot',
            pricing=dict(sku='sku', region='us-east5', provisioning_model='spot', source='official-pricing',
                observed_at='2026-10-01T10:00:00Z', accelerator_type='v5litepod-16',
                currency='USD', unit='whole_slice_hour', hourly_usd=8))
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'run.json'
            path.write_text(json.dumps(value))
            runner.load_manifest(path)
            for zone, region in [('us-west4-a', 'us-east5'), ('us-east5', 'us-east5'), ('us-east5-a', 'us-west4')]:
                value['deployment']['zone'] = zone
                value['deployment']['pricing']['region'] = region
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'region in pricing'):
                    runner.load_manifest(path)

    def test_startup_budget_is_validated_and_passed_under_same_lease(self):
        value = fixture()
        value['deployment'] = dict(project='project', zone='us-east5-a',
            accelerator_type='v5litepod-16', runtime_version='v2-alpha-tpuv5-lite',
            hourly_usd=8, budget_usd=20, provisioning_model='spot', attempt_seconds=1800,
            capacity_wait_seconds=300, startup_wait_seconds=600,
            pricing=dict(sku='sku', region='us-east5', provisioning_model='spot', source='official-pricing',
                observed_at='2026-10-01T10:00:00Z', accelerator_type='v5litepod-16',
                currency='USD', unit='whole_slice_hour', hourly_usd=8))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.json'
            path.write_text(json.dumps(value))
            runner.load_manifest(path)
            argv = runner.supervisor_argv(value, 'gs://bucket/run.json', 'f'*64, temporary)
            self.assertEqual(argv[argv.index('--startup-wait-seconds')+1], '600')
            self.assertEqual(argv[argv.index('--attempt-seconds')+1], '1800')
            for invalid in (True, 0, -1, 1801, 1.5, '600'):
                value['deployment']['startup_wait_seconds'] = invalid
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'startup_wait_seconds'):
                    runner.load_manifest(path)

    def test_training_namespace_binds_both_argv_forms_and_rejects_duplicates(self):
        manifest = fixture()
        expected = manifest['output_prefix'] + '/checkpoints'
        self.assertEqual(runner.training_output(manifest), expected)
        manifest['workload'] = manifest['workload'][:-2] + ['--output=' + expected]
        self.assertEqual(runner.training_output(manifest), expected)
        for argv in (manifest['workload'][:-1] + ['--output=gs://bucket/old'],
            manifest['workload'] + ['--output', expected], manifest['workload'][:-1],
            manifest['workload'][:-1] + ['--output']):
            with self.assertRaises(ValueError):
                runner.training_output({**manifest, 'workload': argv})

    def test_setup_receipt_is_not_qualification_and_trivial_probes_fail(self):
        for argv in (['true'], ['echo', 'done'], ['python', '--help'], ['python', '-c', 'pass']):
            with self.assertRaises(ValueError):
                runner.meaningful_setup(argv)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.json'
            manifest = fixture()
            manifest.pop('qualification')
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'explicit qualification'):
                runner.load_manifest(path)

    def test_physical_receipt_is_bound_and_cannot_qualify_skipped_missing_or_stale_tests(self):
        manifest = fixture()
        source_tree = {'scripts/train.py': 'd'*64}
        identity = 'e'*64
        value = dict(schema_version=1, passed=True, backend='tpu', manifest_sha256=identity,
            runtime_lock_sha256=manifest['runtime_lock']['sha256'], source_tree_sha256=runner.source_tree_digest(source_tree),
            devices=16, processes=4, tests=[dict(id='loss', status='passed'), dict(id='recovery', status='passed')])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'qualification.json'
            path.write_text(json.dumps(value))
            result = runner.verify_qualification(manifest, root, identity, source_tree)
            self.assertEqual(result['sha256'], runner.digest(path))
            for bad in ({**value, 'backend': 'cpu'}, {**value, 'manifest_sha256': 'f'*64},
                {**value, 'tests': [dict(id='loss', status='skipped'), dict(id='recovery', status='passed')]},
                {**value, 'tests': [dict(id='loss', status='passed')]}, {**value, 'devices': 8}):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    runner.verify_qualification(manifest, root, identity, source_tree)

    def test_existing_physical_test_suite_receipts_are_verified_without_inventing_acceptance(self):
        from scripts.validate_tpu import source_digest
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            versions = {name: '1.0' for name in ('jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint', 'libtpu')}
            (source / 'lock.txt').write_text('\n'.join(name + '==' + version for name, version in versions.items()) + '\n')
            manifest = fixture()
            manifest['runtime_lock']['sha256'] = runner.digest(source / 'lock.txt')
            manifest['qualification']['format'] = 'test_suite'
            summary = dict(passed=True, hardware_passed=True, not_run=[], source_python_sha256=source_digest(source),
                results=[dict(passed=True, returncode=0, tests=[dict(test='loss', status='passed'), dict(test='recovery', status='passed')])])
            hardware = dict(backend='tpu', devices=16, processes=4, runtime=dict(python='3.12.12', packages=versions))
            (root / 'qualification.json').write_text(json.dumps(summary))
            (root / 'hardware.json').write_text(json.dumps(hardware))
            with patch.object(runner.subprocess, 'check_output', return_value='Python 3.12.12'):
                result = runner.verify_qualification(manifest, root, 'e'*64, runner.source_snapshot(source))
                self.assertEqual(result['hardware_sha256'], runner.digest(root / 'hardware.json'))
                hardware['runtime']['packages']['jax'] = '2.0'
                (root / 'hardware.json').write_text(json.dumps(hardware))
                with self.assertRaisesRegex(ValueError, 'frozen package lock'):
                    runner.verify_qualification(manifest, root, 'e'*64, runner.source_snapshot(source))

    def test_worker_refuses_changed_packages_or_interpreter_before_model_start(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'source').mkdir()
            (root / 'source/lock.txt').write_text('example==1.0\n')
            (root / 'venv/bin').mkdir(parents=True)
            python = root / 'venv/bin/python'
            python.write_text('binary')
            manifest = fixture()
            manifest['workload'] = ['metadata-workload']
            manifest.pop('qualification')
            manifest['runtime_lock']['sha256'] = runner.digest(root / 'source/lock.txt')
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            receipt = dict(schema_version=2, setup_checks_executed=True, physical_acceptance=False, qualification=None,
                manifest_sha256=runner.digest(path), source_tree=runner.source_snapshot(root / 'source'),
                packages=[dict(name='example', version='1.0')], python='Python 3.12.12', interpreter_sha256=runner.digest(python))
            (root / 'runtime-receipt.json').write_text(json.dumps(receipt))
            with patch.object(runner.subprocess, 'check_output', return_value='[{"name":"example","version":"2.0"}]'), patch.object(runner.subprocess, 'Popen') as start:
                with self.assertRaisesRegex(ValueError, 'package inventory changed'):
                    runner.run_worker(path, root)
                start.assert_not_called()
            python.write_text('changed binary')
            with patch.object(runner.subprocess, 'Popen') as start:
                with self.assertRaisesRegex(ValueError, 'interpreter binary changed'):
                    runner.run_worker(path, root)
                start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
