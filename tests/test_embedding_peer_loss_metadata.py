"""Fault campaign admission and receipt contracts; no model or cloud execution."""
import copy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from flaxchat import embedding_contract, runtime as runtime_module  # Preload metadata helpers before sys.modules isolation.
import types
from unittest.mock import Mock, patch
from scripts import validate_embedding_peer_loss as campaign
from scripts.validate_embedding_peer_loss import validate_plan, verify_drain, verify_fault, compare_final, verify_durable_status, remote_status_command


def plan():
    return dict(schema_version=1, fault='peer-sigkill', project='project', zone='zone', node='node',
        source_sha256='a'*64, runtime_sha256='b'*64, parent_identity='d'*64, processes=2, target_rank=1,
        fault_step=2, final_step=4, timeout_seconds=1800, remote_python='/frozen/python', directory='/frozen',
        guard_receipt='/tmp/guard.json', baseline_output='gs://bucket/new/baseline', recovery_output='gs://bucket/new/recovery',
        training_argv=['--distributed', '--steps', '4', '--save-every', '1'])


class PeerLossTests(unittest.TestCase):
    def test_plan_refuses_graceful_or_single_host_or_overwriting(self):
        self.assertEqual(validate_plan(plan())['fault'], 'peer-sigkill')
        for change in ({'fault': 'graceful'}, {'processes': 1}, {'target_rank': 0}, {'fault_step': 4},
                       {'baseline_output': plan()['recovery_output']}, {'baseline_output': plan()['recovery_output'] + '/nested'},
                       {'training_argv': plan()['training_argv'] + ['--steps=8']}, {'training_argv': plan()['training_argv'] + ['--out', 'other']}, {'timeout_seconds': 7200},
                       {'training_argv': plan()['training_argv'] + ['--resume']}):
            with self.assertRaises(ValueError):
                validate_plan({**plan(), **change})

    def test_resume_requires_every_remote_terminal_not_transport_failure(self):
        summary = dict(finished_epoch=1, workers=[dict(rank=0, transport_returncode=0, returncode=124),
                                                dict(rank=1, transport_returncode=0, returncode=137)])
        verify_drain(summary, 2)
        for change in ({'workers': summary['workers'][:1]}, {'finished_epoch': None}):
            with self.assertRaises(ValueError):
                verify_drain({**summary, **change}, 2)
        failed = copy.deepcopy(summary)
        failed['workers'][0]['transport_returncode'] = 255
        with self.assertRaises(ValueError):
            verify_drain(failed, 2)

    def test_fault_requires_committed_identity_and_all_rank_owners(self):
        value = plan()
        events = [dict(event='embedding_peer_fault_ready', launcher_rank=rank, rank=rank, token='token',
            backend='tpu', devices=8, committed_step=2, processes=2, source_sha256='a'*64, runtime_sha256='b'*64, pid=100+rank,
            committed_manifest_sha256='c'*64) for rank in range(2)]
        events.append({**events[1], 'event': 'embedding_peer_sigkill'})
        self.assertEqual(verify_fault(events, value, 'token')['launcher_rank'], 1)
        for rows in (events[:-1], events[1:], events + [events[-1]]):
            with self.assertRaises(ValueError):
                verify_fault(rows, value, 'token')
        with self.assertRaises(ValueError):
            verify_fault(events, value, 'wrong-token')

    def test_durable_status_not_attachment_timeout_marker_is_resume_evidence(self):
        identity = '1'*32
        self.assertEqual(verify_durable_status('137\n', 137), 137)
        self.assertEqual(remote_status_command(identity, 1),
                         'test -f /tmp/flaxchat-command-' + identity + '-1/status && cat /tmp/flaxchat-command-' + identity + '-1/status')
        for payload in ('', '124', '137\n137', 'running', '-9'):
            with self.assertRaises(ValueError):
                verify_durable_status(payload, 137)
        with self.assertRaises(ValueError):
            remote_status_command('../unrelated', 1)

    def test_comparison_requires_model_optimizer_and_training_state(self):
        manifest = dict(model_state={'hash': 'model'}, optimizer_state={'hash': 'optimizer'}, training_state={'hash': 'cursor'})
        self.assertTrue(all(compare_final(manifest, manifest).values()))
        for key in manifest:
            with self.assertRaises(ValueError):
                compare_final(manifest, {**manifest, key: {}})

    def test_worker_reaches_trainer_run_with_parsed_recipe_without_model(self):
        class ReachedTrainer(Exception):
            pass
        trainer = types.ModuleType('scripts.train_yat_embedding_finetune')
        trainer.save_checkpoint = Mock()
        original = trainer.save_checkpoint
        def add_arguments(parser):
            parser.add_argument('--steps', type=int, required=True)
        trainer.add_stage_arguments = add_arguments
        trainer.run = Mock(side_effect=ReachedTrainer())
        # There deliberately is no main(argv) entry point on this fake trainer.
        jax = types.ModuleType('jax')
        jax.distributed = types.SimpleNamespace(is_initialized=lambda: True)
        jax.default_backend = lambda: 'tpu'
        jax.process_count = lambda: 2
        experimental = types.ModuleType('jax.experimental')
        experimental.multihost_utils = types.SimpleNamespace(sync_global_devices=Mock())
        from flaxchat.embedding_contract import canonical_hash
        runtime = {'jax': 'fixture'}
        argv = ['--fault-step', '2', '--target-rank', '1', '--source-sha256', 'a'*64,
                '--runtime-sha256', canonical_hash(runtime), '--token', 'token', '--', '--steps', '4']
        modules = {'scripts.train_yat_embedding_finetune': trainer, 'jax': jax, 'jax.experimental': experimental}
        with patch.dict('sys.modules', modules), patch.object(embedding_contract, 'source_identity', return_value={'sha256': 'a'*64}), patch.object(runtime_module, 'runtime_identity', return_value=runtime):
            with self.assertRaises(ReachedTrainer):
                campaign.worker(argv)
        trainer.run.assert_called_once()
        self.assertEqual(trainer.run.call_args.args[0].steps, 4)
        self.assertIs(trainer.save_checkpoint, original)

    def test_worker_injects_only_after_commit_and_only_its_own_pid(self):
        class Killed(Exception):
            pass
        trainer = types.ModuleType('scripts.train_yat_embedding_finetune')
        original = Mock(return_value=True)
        trainer.save_checkpoint = original
        def add_arguments(parser):
            parser.add_argument('--output', required=True)
        trainer.add_stage_arguments = add_arguments
        manager = types.SimpleNamespace(directory='gs://bucket/recovery/best', wait_until_finished=Mock())
        trainer.run = lambda args: trainer.save_checkpoint(manager, 2)
        jax = types.ModuleType('jax')
        jax.distributed = types.SimpleNamespace(is_initialized=lambda: True)
        jax.default_backend = lambda: 'tpu'
        jax.process_count = lambda: 2
        jax.process_index = lambda: 1
        jax.device_count = lambda: 8
        experimental = types.ModuleType('jax.experimental')
        experimental.multihost_utils = types.SimpleNamespace(sync_global_devices=Mock())
        runtime_value = {'jax': 'fixture'}
        argv = ['--fault-step', '2', '--target-rank', '1', '--source-sha256', 'a'*64,
                '--runtime-sha256', embedding_contract.canonical_hash(runtime_value), '--token', 'token',
                '--', '--output', str(manager.directory)]
        modules = {'scripts.train_yat_embedding_finetune': trainer, 'jax': jax, 'jax.experimental': experimental}
        committed = {'committed_receipt': {'manifest_sha256': 'c'*64}}
        with patch.dict('sys.modules', modules), patch.dict(campaign.os.environ, {'JAX_PROCESS_INDEX': '1'}), patch.object(embedding_contract, 'source_identity', return_value={'sha256': 'a'*64}), patch.object(runtime_module, 'runtime_identity', return_value=runtime_value), patch('flaxchat.checkpoint_metadata.read_committed_metadata', return_value=committed) as read, patch.object(campaign.os, 'getpid', return_value=500), patch.object(campaign.os, 'kill', side_effect=Killed()) as kill, patch('builtins.print'):
            with self.assertRaises(Killed):
                campaign.worker(argv)
        manager.wait_until_finished.assert_called_once()
        read.assert_called_once_with(str(manager.directory), 2, include_receipt=True)
        kill.assert_called_once_with(500, campaign.signal.SIGKILL)
        self.assertIs(trainer.save_checkpoint, original)

    def test_missing_durable_peer_status_prevents_resume_even_with_terminal_marker(self):
        launched = []
        def run(argv, **kwargs):
            if argv[:3] == ['gcloud', 'storage', 'ls']:
                return subprocess.CompletedProcess(argv, 0, '[]', '')
            stage = Path(argv[argv.index('--output') + 1])
            stage.mkdir()
            launched.append(stage.name)
            fault = stage.name == 'interruption'
            rows = [dict(rank=rank, transport_returncode=0, returncode=(124 if rank == 0 else 137) if fault else 0) for rank in range(2)]
            (stage/'summary.json').write_text(json.dumps(dict(workers=rows, finished_epoch=1, passed=not fault, execution_id='1'*32)))
            return subprocess.CompletedProcess(argv, 1 if fault else 0)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            guard = root/'guard.json'
            guard.write_text('{}')
            value = {**plan(), 'guard_receipt': str(guard)}
            with patch('scripts.gcp_cleanup_guard.verify', return_value=4000), patch.object(campaign.subprocess, 'run', side_effect=run), patch.object(campaign.subprocess, 'check_output', side_effect=['0\n', '0\n', '']):
                self.assertEqual(campaign.execute(value, root/'output'), 1)
            self.assertEqual(launched, ['baseline', 'interruption'])
            receipt = json.loads((root/'output/receipt.json').read_text())
            self.assertFalse(receipt['passed'])
            self.assertIn('Durable remote terminal status', receipt['error'])


if __name__ == '__main__':
    unittest.main()
