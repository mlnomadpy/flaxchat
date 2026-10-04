"""Controller terminal outcomes with fake provider I/O; no cloud allocation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from scripts import gcp_spot_supervisor as supervisor


def argv(root):
    return ['--project', 'project', '--zone', 'us-west4-a', '--accelerator-type', 'v5litepod-8',
        '--runtime-version', 'v2-alpha-tpuv5-lite', '--name', 'flaxchat-validation-test', '--output', str(root),
        '--hourly-usd', '9.6', '--budget-usd', '40', '--attempt-seconds', '1200', '--capacity-wait-seconds', '180',
        '--setup', '["setup"]', '--workload', '["workload"]']


class CampaignReceiptTests(unittest.TestCase):
    def test_transport_discovery_survives_terminal_campaign_failure(self):
        error = TimeoutError('Cleanup scope discovery deadline exhausted')
        discovery = {'reads': [{'operation': ['projects', 'get-iam-policy'],
                                'attempt': 2, 'status': 'transient_failure'}]}
        error.cleanup_scope_discovery = discovery
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / 'controller'
            with patch.object(supervisor.gcp_cleanup_guard, 'preflight_cleanup_scope', side_effect=error):
                with self.assertRaises(TimeoutError):
                    supervisor.main(argv(root))
            receipt = json.loads((root / 'campaign-receipt.json').read_text())
            self.assertEqual(receipt['cleanup_scope_discovery'], discovery)
            self.assertFalse((root / 'ledger.json').exists())
            self.assertTrue(receipt['no_remote_execution'])

    def setUp(self):
        scope = patch.object(supervisor.gcp_cleanup_guard, 'preflight_cleanup_scope',
                             return_value={'server_permission_probes_required': True})
        scope.start()
        self.addCleanup(scope.stop)

    def test_scope_failure_precedes_reservation_and_all_provider_mutations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / 'controller'
            with patch.object(supervisor.gcp_cleanup_guard, 'preflight_cleanup_scope',
                              side_effect=ValueError('Cleanup IAM scope does not cover region')), \
                 patch.object(supervisor, 'cloud') as cloud, \
                 patch.object(supervisor.gcp_cleanup_guard, 'main') as arm, \
                 patch.object(supervisor.gcp_tpu_run, 'main') as worker:
                with self.assertRaisesRegex(ValueError, 'IAM scope'):
                    supervisor.main(argv(root))
            cloud.assert_not_called()
            arm.assert_not_called()
            worker.assert_not_called()
            self.assertFalse((root / 'ledger.json').exists())
            receipt = json.loads((root / 'campaign-receipt.json').read_text())
            self.assertEqual(receipt['phase'], 'cleanup_scope_admission')
            self.assertTrue(receipt['no_remote_execution'])
            self.assertFalse(receipt['cleanup_required'])

    def test_startup_timeout_retains_allocated_state_and_distinct_phase(self):
        clock = {'elapsed': 0., 'created': False}
        def cloud(arguments, **kwargs):
            if arguments[3] == 'create':
                clock['created'] = True
                return {'name': 'operation'}
            if not clock['created']:
                return None
            return {'state': {'state': 'PROVISIONING'}} if arguments[2] == 'queued-resources' else {'state': 'CREATING'}
        def arm(arguments):
            Path(arguments[arguments.index('--receipt') + 1]).write_text('{"verified": true}')
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / 'controller'
            with patch.object(supervisor, 'cloud', side_effect=cloud), \
                 patch.object(supervisor.gcp_cleanup_guard, 'main', side_effect=arm), \
                 patch.object(supervisor, 'cleanup_resource', return_value=True), \
                 patch.object(supervisor.gcp_tpu_run, 'main') as worker, \
                 patch.object(supervisor.time, 'monotonic', side_effect=lambda: clock['elapsed']), \
                 patch.object(supervisor.time, 'time', side_effect=lambda: 1000 + clock['elapsed']), \
                 patch.object(supervisor.time, 'sleep', side_effect=lambda seconds: clock.update(elapsed=clock['elapsed'] + seconds)):
                with self.assertRaisesRegex(TimeoutError, 'startup wait budget'):
                    supervisor.main(argv(root) + ['--startup-wait-seconds', '15'])
            worker.assert_not_called()
            receipt = json.loads((root / 'campaign-receipt.json').read_text())
            attempt = receipt['attempts'][0]
            self.assertEqual(attempt['failure']['phase'], 'startup_wait')
            self.assertTrue(receipt['cleanup_verified'])
            self.assertTrue(receipt['no_remote_execution'])
            self.assertTrue(any(item['queue_state'] == 'PROVISIONING'
                                for item in attempt['resource_state_observations']))
            self.assertEqual(receipt['startup_wait_seconds'], 15)

    def test_denied_scope_preserves_prior_shared_reservations(self):
        from flaxchat.operations import RunLedger
        with TemporaryDirectory() as temporary:
            ledger_path = Path(temporary) / 'campaign-ledger.json'
            RunLedger(ledger_path, 40).reserve('previous-attempt', 'owned/previous', 9.6, 1200, 2)
            original = ledger_path.read_bytes()
            root = Path(temporary) / 'controller'
            with patch.object(supervisor.gcp_cleanup_guard, 'preflight_cleanup_scope',
                              side_effect=ValueError('Cleanup IAM scope does not cover name')):
                with self.assertRaisesRegex(ValueError, 'IAM scope'):
                    supervisor.main(argv(root) + ['--campaign-ledger', str(ledger_path)])
            self.assertEqual(ledger_path.read_bytes(), original)
            receipt = json.loads((root / 'campaign-receipt.json').read_text())
            self.assertEqual(receipt['reserved_usd'], 5.2)

    def test_capacity_failure_preserves_original_cause_absence_and_no_remote_stage(self):
        def cloud(args, **kwargs):
            return {'name': 'operation'} if args[3] == 'create' else None
        def arm(args):
            Path(args[args.index('--receipt')+1]).write_text(json.dumps({'verified': True}))
        def cleanup(name, flags, **kwargs):
            kwargs['observe']({'queue_absent': True, 'node_absent': True})
            return True
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'controller'
            with patch.object(supervisor, 'cloud', side_effect=cloud), patch.object(supervisor.gcp_cleanup_guard, 'main', side_effect=arm), patch.object(supervisor, 'wait_for_ready', side_effect=TimeoutError('Spot capacity wait budget exhausted')), patch.object(supervisor, 'cleanup_resource', side_effect=cleanup), patch.object(supervisor.gcp_tpu_run, 'main') as worker:
                with self.assertRaises(TimeoutError):
                    supervisor.main(argv(root))
            worker.assert_not_called()
            receipt = json.loads((root/'campaign-receipt.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertFalse(receipt['passed'])
            self.assertTrue(receipt['no_remote_execution'])
            self.assertEqual(receipt['model_execution_state'], 'not_started')
            self.assertTrue(receipt['cleanup_verified'])
            self.assertEqual(receipt['attempts'][0]['failure']['phase'], 'capacity_wait')
            self.assertIn('capacity', receipt['attempts'][0]['failure']['error'])
            self.assertEqual(receipt['reserved_usd'], 10)
            self.assertIsNone(receipt['posted_usage_usd'])
            ledger = json.loads((root/'ledger.json').read_text())
            self.assertEqual(next(iter(ledger['attempts'].values()))['status'], 'resource_absent')

    def test_cleanup_unknown_does_not_erase_capacity_error(self):
        def arm(args):
            Path(args[args.index('--receipt')+1]).write_text('{"verified": true}')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'controller'
            with patch.object(supervisor, 'cloud', side_effect=lambda args, **kwargs: {'name': 'operation'} if args[3] == 'create' else None), patch.object(supervisor.gcp_cleanup_guard, 'main', side_effect=arm), patch.object(supervisor, 'wait_for_ready', side_effect=TimeoutError('capacity exhausted')), patch.object(supervisor, 'cleanup_resource', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'absence'):
                    supervisor.main(argv(root))
            receipt = json.loads((root/'campaign-receipt.json').read_text())
            self.assertFalse(receipt['cleanup_verified'])
            self.assertEqual(receipt['attempts'][0]['cleanup']['state'], 'unverified')
            self.assertIn('capacity exhausted', receipt['attempts'][0]['failure']['error'])

    def test_success_is_terminal_only_after_absence_and_remote_dispatch_recorded(self):
        created = False
        def cloud(args, **kwargs):
            nonlocal created
            if args[3] == 'create':
                created = True
                return {'name': 'operation'}
            if created:
                return dict(name='node', labels={'flaxchat-run': 'flaxchat-validation-test', 'flaxchat-stage': 'qualification'})
            return None
        def arm(args):
            Path(args[args.index('--receipt')+1]).write_text('{"verified": true}')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'controller'
            with patch.object(supervisor, 'cloud', side_effect=cloud), patch.object(supervisor.gcp_cleanup_guard, 'main', side_effect=arm), patch.object(supervisor, 'wait_for_ready'), patch.object(supervisor, 'cleanup_resource', return_value=True), patch.object(supervisor.gcp_tpu_run, 'main', return_value=0) as worker:
                self.assertEqual(supervisor.main(argv(root)), 0)
            self.assertEqual(worker.call_count, 2)
            receipt = json.loads((root/'campaign-receipt.json').read_text())
            self.assertTrue(receipt['passed'])
            self.assertTrue(receipt['cleanup_verified'])
            self.assertFalse(receipt['no_remote_execution'])
            self.assertEqual(receipt['attempts'][0]['remote_stages_started'], ['setup', 'workload'])

    def test_guard_failure_has_no_provision_or_cleanup_and_old_receipt_survives(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'controller'
            with patch.object(supervisor, 'cloud', return_value=None) as cloud, patch.object(supervisor.gcp_cleanup_guard, 'main', side_effect=ValueError('guard identity mismatch')), patch.object(supervisor, 'cleanup_resource') as cleanup:
                with self.assertRaises(ValueError):
                    supervisor.main(argv(root))
            cleanup.assert_not_called()
            self.assertTrue(all(call.args[0][3] == 'describe' for call in cloud.call_args_list))
            original = (root/'campaign-receipt.json').read_bytes()
            receipt = json.loads(original)
            self.assertFalse(receipt['cleanup_required'])
            self.assertFalse(receipt['cleanup_verified'])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                supervisor.main(argv(root))
            self.assertEqual((root/'campaign-receipt.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
