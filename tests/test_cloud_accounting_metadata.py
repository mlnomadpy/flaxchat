"""Metadata only: no JAX import, model execution or cloud calls."""
import io
import json
from pathlib import Path
import tarfile
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from flaxchat.cost_accounting import estimate_slice_cost, posted_summary
from scripts.cloud_accounting import billing_sql, cleanup_plan, normalize_storage
from scripts.representation_run import load_manifest, extract_archive, validate_lock


class AccountingTests(unittest.TestCase):
    def test_slice_rate_ignores_device_count_and_requires_evidence(self):
        rate = dict(sku='sku', region='us-east5', provisioning_model='spot', source='billing-export-row',
                    observed_at='2026-09-30T10:00:00Z', accelerator_type='v5litepod-16',
                    currency='USD', unit='whole_slice_hour', hourly_usd=8)
        self.assertEqual(estimate_slice_cost(1800, rate), 4)
        for bad in ({**rate, 'unit': 'chip_hour'}, {**rate, 'hourly_usd': float('nan')}, {**rate, 'source': ''}):
            with self.assertRaises(ValueError):
                estimate_slice_cost(1800, bad)

    def test_billing_preserves_credit_sign_and_unknown_balance(self):
        self.assertIsNone(posted_summary([])['posted_gross'])
        row = dict(gross='10', credits='-7', promotional_credits='-5', currency='USD',
                   latest_export_time='2026-09-30', latest_usage_end_time='2026-09-29')
        summary = posted_summary([row])
        self.assertEqual(summary['posted_net'], 3)
        self.assertIsNone(summary['remaining_credits'])
        with self.assertRaises(ValueError):
            posted_summary([row, {**row, 'currency': 'EUR'}])
        with self.assertRaises(ValueError):
            billing_sql('table` where true --')
        self.assertIn('UNNEST(credits)', billing_sql('project.dataset.table'))

    def test_storage_all_generations_and_soft_delete_are_separate(self):
        live = dict(name='model/weights', generation='2', size='200')
        old = dict(name='model/weights', generation='1', size='100')
        deleted = dict(name='cache/old', generation='3', size='300')
        inventory = normalize_storage(dict(live=[dict(type='object', metadata=live)], all_versions=[dict(metadata=live), dict(metadata=old)], soft_deleted=[dict(metadata=deleted)]), 'gs://bucket')
        self.assertEqual(inventory['bytes'], dict(live=200, noncurrent=100, soft_deleted=300))
        plan = cleanup_plan(inventory, ['model/', 'cache/'], ['model/'])
        self.assertTrue(all(row['action'] == 'retain' for row in plan['objects']))
        self.assertFalse(plan['destructive_execution'])

    def test_nonatomic_inventory_retains_observed_live_objects(self):
        live = dict(metadata=dict(name='model/weights', generation='1', size='123'))
        result = normalize_storage(dict(live=[live], all_versions=[], soft_deleted=[]), 'gs://bucket')
        self.assertEqual(result['bytes']['live'], 123)
        self.assertFalse(result['consistent'])
        self.assertEqual(len(result['objects']), 1)

    def test_lock_and_archive_reject_mutable_or_escaping_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / 'lock.txt'
            lock.write_text('jax>=0.11\n')
            with self.assertRaises(ValueError):
                validate_lock(lock)
            lock.write_text('jax==0.11.1\njaxlib==0.11.1\n')
            self.assertEqual(validate_lock(lock)['jax'], '0.11.1')
            archive = root / 'bad.tar'
            with tarfile.open(archive, 'w') as handle:
                item = tarfile.TarInfo('../escape')
                item.size = 1
                handle.addfile(item, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                extract_archive(archive, root / 'out')
            self.assertFalse((root / 'escape').exists())

    def test_legacy_adapter_never_calls_cloud(self):
        from scripts import train_tpu
        args = train_tpu.build_parser().parse_args(['--name', 'test'])
        with self.assertRaisesRegex(RuntimeError, 'Legacy tpuz paid execution is disabled'):
            train_tpu.run_adapter(args, None, object(), None)

    def test_stage_prefix_cannot_reuse_historical_output(self):
        value = dict(schema_version=1, run_id='campaign', stage_id='stage', parent_identity='sha', cleanup_owner='guard',
                     expected_topology='v5litepod-16', expected_device_count=16, expected_process_count=4, artifacts=[dict(uri='gs://bucket/wheels', target='wheels', archive=True, sha256='c'*64)], runtime_lock=dict(path='lock.txt', sha256='b'*64, wheelhouse_target='wheels'), source=dict(uri='gs://bucket/source', sha256='a'*64),
                     output_prefix='gs://bucket/old-stage', workload=['train'], setup_checks=['check'])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.json'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Output prefix'):
                load_manifest(path)
            value['output_prefix'] = 'gs://bucket/campaign/stage'
            path.write_text(json.dumps(value))
            self.assertEqual(load_manifest(path)['stage_id'], 'stage')

    def test_guarded_replacement_generates_and_dispatches_same_workload(self):
        from scripts import representation_run as runner
        from scripts import gcp_spot_supervisor as supervisor
        value = dict(schema_version=1, run_id='campaign', stage_id='stage', parent_identity='parent-sha', cleanup_owner='cloud-workflow',
                     expected_topology='v5litepod-16', expected_device_count=16, expected_process_count=4, artifacts=[dict(uri='gs://bucket/wheels', target='wheels', archive=True, sha256='c'*64)], runtime_lock=dict(path='lock.txt', sha256='b'*64, wheelhouse_target='wheels'),
                     source=dict(uri='gs://bucket/source', sha256='a'*64), output_prefix='gs://bucket/campaign/stage',
                     workload=['{python}', '-m', 'scripts.train_yat_embedding_finetune', '--steps', '15000', '--output', '{checkpoint_output}'], qualification=dict(receipt='acceptance.json', required_tests=['objective']), setup_checks=['{python}', 'tpu-check.py'],
                     deployment=dict(project='project', zone='us-east5-a', accelerator_type='v5litepod-16', runtime_version='v2-alpha-tpuv5-lite',
                                     hourly_usd=8, budget_usd=20, attempt_seconds=1800, capacity_wait_seconds=180,
                                     pricing=dict(sku='sku', region='us-east5', provisioning_model='spot', source='pricing-export',
                                                  observed_at='2026-09-30T10:00:00Z', accelerator_type='v5litepod-16',
                                                  currency='USD', unit='whole_slice_hour', hourly_usd=8)))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'run.json'
            path.write_text(json.dumps(value))
            observed = []
            with patch.object(supervisor, 'main', side_effect=lambda argv: observed.append(argv) or 0):
                self.assertEqual(runner.main(['launch', '--manifest', str(path), '--manifest-uri', 'gs://bucket/run.json', '--output', str(root / 'receipts')]), 0)
            argv = observed[0]
            self.assertEqual(argv[argv.index('--accelerator-type') + 1], 'v5litepod-16')
            self.assertEqual(argv[argv.index('--capacity-wait-seconds') + 1], '180')
            setup = json.loads(argv[argv.index('--setup') + 1])
            self.assertIn(runner.digest(path), setup[2])
            self.assertIn('sha256sum -c -', setup[2])
            self.assertIn('scripts.representation_run setup', setup[2])
            workload = json.loads(argv[argv.index('--workload') + 1])
            self.assertIn('run', workload)
            self.assertIn('scripts.representation_run', workload)
            # A long manifest is explicit, bounded, and carried to the supervisor.
            for checks_timeout in (29, 1801, True, 1200.0):
                value['setup_checks_timeout_seconds'] = checks_timeout
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'Setup checks timeout'):
                    load_manifest(path)
            for checks_timeout in (30, 1200, 1800):
                value['setup_checks_timeout_seconds'] = checks_timeout
                path.write_text(json.dumps(value))
                self.assertEqual(load_manifest(path)['setup_checks_timeout_seconds'], checks_timeout)
            value['deployment']['attempt_seconds'] = 43200
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Attempt lease'):
                load_manifest(path)
            value['deployment']['allow_long_lease'] = True
            path.write_text(json.dumps(value))
            loaded = load_manifest(path)
            self.assertIn('--allow-long-lease', runner.supervisor_argv(
                loaded, 'gs://bucket/run.json', runner.digest(path), root / 'long-receipts'))
            for duration, opt_in in ((43201, True), (43200, 'true'), (True, True)):
                value['deployment'].update(attempt_seconds=duration, allow_long_lease=opt_in)
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_manifest(path)
            value['deployment'].update(attempt_seconds=1800, allow_long_lease=False)
            value['deployment']['capacity_wait_seconds'] = 1801
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Capacity wait'):
                load_manifest(path)
            value['deployment']['capacity_wait_seconds'] = 180
            value['deployment']['hourly_usd'] = 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'pricing evidence'):
                load_manifest(path)

    def test_setup_checks_hashed_inputs_installs_offline_and_commits_only_after_checks(self):
        from scripts import representation_run as runner
        import shutil
        import subprocess
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / 'lock.txt'
            lock.write_text('example==1.0\n')
            source = root / 'source.tar'
            with tarfile.open(source, 'w') as archive:
                archive.add(lock, arcname='lock.txt')
            wheel = root / 'example.whl'
            wheel.write_text('fixture bytes')
            wheels = root / 'wheels.tar'
            with tarfile.open(wheels, 'w') as archive:
                archive.add(wheel, arcname='example.whl')
            manifest = dict(schema_version=1, run_id='campaign', stage_id='stage', parent_identity='parent-sha', cleanup_owner='cloud',
                     expected_topology='v5litepod-16', expected_device_count=16, expected_process_count=4,
                     runtime_lock=dict(path='lock.txt', sha256=runner.digest(lock), wheelhouse_target='wheels'),
                     source=dict(uri='gs://bucket/source', sha256=runner.digest(source)),
                     artifacts=[dict(uri='gs://bucket/wheels', target='wheels', archive=True, sha256=runner.digest(wheels))],
                     output_prefix='gs://bucket/campaign/stage', workload=['train'], setup_checks=['qualify'])
            path = root / 'run.json'
            path.write_text(json.dumps(manifest))
            calls = []
            qualification_timeouts = []
            def run(argv, **kwargs):
                calls.append(argv)
                if argv == ['qualify']:
                    qualification_timeouts.append(kwargs['timeout'])
                if '-m' in argv and 'venv' in argv:
                    binary = Path(argv[-1]) / 'bin/python'
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_text('interpreter fixture')
                if argv[:3] == ['gcloud', 'storage', 'cp']:
                    shutil.copy(source if argv[3].endswith('/source') else wheels, argv[4])
                return subprocess.CompletedProcess(argv, 0)
            def output(argv, **kwargs):
                if 'list' in argv:
                    return json.dumps([dict(name='example', version='1.0')])
                if '--version' in argv:
                    return 'Python 3.12.12'
                return json.dumps(dict(backend='tpu', devices=16, local_devices=4, processes=4, device_kind='TPU v5 lite'))
            destination = root / 'stage-root'
            with patch.object(runner.subprocess, 'run', side_effect=run), patch.object(runner.subprocess, 'check_output', side_effect=output), patch.object(runner.platform, 'platform', return_value='Linux'):
                self.assertEqual(runner.setup(path, destination), 0)
            install = next(argv for argv in calls if 'install' in argv)
            self.assertIn('--no-index', install)
            self.assertIn('--no-deps', install)
            self.assertIn('--find-links', install)
            self.assertTrue(json.loads((destination / 'runtime-receipt.json').read_text())['setup_checks_executed'])
            self.assertEqual(qualification_timeouts, [300])
            manifest['setup_checks_timeout_seconds'] = 1200
            path.write_text(json.dumps(manifest))
            with patch.object(runner.subprocess, 'run', side_effect=run), patch.object(runner.subprocess, 'check_output', side_effect=output), patch.object(runner.platform, 'platform', return_value='Linux'):
                self.assertEqual(runner.setup(path, root / 'extended-stage-root'), 0)
            self.assertEqual(qualification_timeouts, [300, 1200])
            del manifest['setup_checks_timeout_seconds']
            path.write_text(json.dumps(manifest))
            # A new setup which fails qualification must not leave a passing receipt.
            def reject(argv, **kwargs):
                if argv == ['qualify']:
                    raise subprocess.CalledProcessError(1, argv)
                return run(argv, **kwargs)
            failed = root / 'failed-stage'
            with patch.object(runner.subprocess, 'run', side_effect=reject), patch.object(runner.subprocess, 'check_output', side_effect=output), patch.object(runner.platform, 'platform', return_value='Linux'):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.setup(path, failed)
            self.assertFalse((failed / 'runtime-receipt.json').exists())
            with patch.object(runner.subprocess, 'run', side_effect=reject), patch.object(runner.subprocess, 'check_output', side_effect=output), patch.object(runner.platform, 'platform', return_value='Linux'):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.setup(path, destination)
            self.assertFalse((destination / 'runtime-receipt.json').exists())
            with self.assertRaises(FileNotFoundError):
                runner.run_worker(path, destination)


    def test_campaign_admissions_share_cap_and_unique_attempt_ids(self):
        from flaxchat.operations import RunLedger
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'campaign.json'
            def admission(identity):
                ledger = RunLedger(path, 10)
                try:
                    ledger.reserve(identity, 'resource-' + identity, 6, 3600)
                    return True
                except ValueError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sum(pool.map(admission, ['run-a:0', 'run-b:0'])), 1)
            self.assertEqual(RunLedger(path, 10).summary()['reserved_usd'], 6)

    def test_hung_telemetry_preserves_workload_error(self):
        from scripts import representation_run as runner
        value = dict(schema_version=1, run_id='campaign', stage_id='stage', parent_identity='sha', cleanup_owner='guard',
                     expected_topology='v5litepod-16', expected_device_count=16, expected_process_count=4, artifacts=[dict(uri='gs://bucket/wheels', target='wheels', archive=True, sha256='c'*64)], runtime_lock=dict(path='lock.txt', sha256='b'*64, wheelhouse_target='wheels'), source=dict(uri='gs://bucket/source', sha256='a'*64),
                     output_prefix='gs://bucket/campaign/stage', workload=['{python}', 'train'], setup_checks=['check'])
        class Process:
            pid = 1
            def wait(self, timeout):
                return 7
        import subprocess
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'run.json'
            path.write_text(json.dumps(value))
            (root / 'venv/bin').mkdir(parents=True)
            (root / 'venv/bin/python').write_text('interpreter fixture')
            (root / 'source').mkdir()
            lock_path = root / 'source/lock.txt'
            lock_path.write_text('fixture')
            value['runtime_lock']['sha256'] = runner.digest(lock_path)
            path.write_text(json.dumps(value))
            (root / 'runtime-receipt.json').write_text(json.dumps(dict(schema_version=2, manifest_sha256=runner.digest(path), setup_checks_executed=True, physical_acceptance=False, interpreter_sha256=runner.digest(root / 'venv/bin/python'), qualification=None, packages=[], python='Python 3.12.12', source_tree=runner.source_snapshot(root / 'source'))))
            import signal
            def stalled_upload(*args, **kwargs):
                # The child is gone; the local result must already exist before telemetry.
                self.assertEqual(json.loads(next(root.glob('*-status.json')).read_text())['workload_returncode'], 7)
                handler = signal.getsignal(signal.SIGTERM)
                if callable(handler):
                    handler(signal.SIGTERM, None)
                raise subprocess.TimeoutExpired('upload', 30)
            with patch.object(runner.os, 'killpg', side_effect=ProcessLookupError()), patch.object(runner.subprocess, 'Popen', return_value=Process()), patch.object(runner.subprocess, 'run', side_effect=stalled_upload), patch.object(runner.subprocess, 'check_output', side_effect=['[]', 'Python 3.12.12']):
                self.assertEqual(runner.run_worker(path, root), 7)
            receipt = next(root.glob('*-status.json'))
            self.assertEqual(json.loads(receipt.read_text())['workload_returncode'], 7)
            self.assertEqual(len(json.loads(receipt.read_text())['telemetry_events']), 1)


if __name__ == '__main__':
    unittest.main()
