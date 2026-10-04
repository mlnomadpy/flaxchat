"""Same-handle attachment orchestration without cloud or model execution."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from flaxchat import embedding_contract, runtime as runtime_module  # Preload metadata helpers before sys.modules isolation.
import types
from unittest.mock import Mock, patch
from scripts import validate_embedding_attachment_loss as campaign


def plan():
    runtime, parent = {'jax': 'fixture'}, {'weights': 'fixture'}
    return dict(schema_version=1, fault='local-attachment-sigkill', project='project', zone='zone', node='node',
        guard_receipt='/tmp/guard', directory='/frozen', remote_python='/frozen/python', processes=2,
        timeout_seconds=1800, worker_timeout_seconds=600, source_sha256='a'*64,
        runtime_sha256=campaign.canonical_hash(runtime), parent_identity=campaign.canonical_hash(parent),
        checkpoint_output='gs://bucket/new', final_step=4, training_argv=['--distributed', '--steps', '4'])


class AttachmentTests(unittest.TestCase):
    def test_plan_rejects_wrong_scope_and_unbounded_recipe(self):
        self.assertEqual(campaign.validate_plan(plan())['processes'], 2)
        for delta in ({'fault': 'network-loss'}, {'processes': 1}, {'worker_timeout_seconds': 1800},
                      {'training_argv': ['--distributed', '--steps', '4', '--resume']},
                      {'training_argv': ['--distributed', '--steps', '4', '--steps=8']},
                      {'training_argv': ['--distributed', '--steps', '4', '--out', 'other']}):
            with self.assertRaises(ValueError):
                campaign.validate_plan({**plan(), **delta})

    def test_terminal_requires_one_actual_application_event_not_attachment_markers(self):
        identity = '1'*32
        stamp = {'event': 'embedding_application_start', 'execution_id': identity}
        marker = f'FLAXCHAT_REMOTE_EXIT_{identity}_0=0\n'
        campaign.verify_terminal(json.dumps(stamp)+'\n'+marker, stamp, identity, 0, 0)
        for log, transport in ((marker, 0), (json.dumps(stamp)+'\n'+json.dumps(stamp)+'\n'+marker, 0),
                               (json.dumps(stamp)+'\n'+marker, 255)):
            with self.assertRaises(ValueError):
                campaign.verify_terminal(log, stamp, identity, 0, transport)

    def test_execute_kills_only_local_attachments_and_reuses_exact_handle(self):
        value, identity = plan(), '1'*32
        stamps = [dict(event='embedding_application_start', execution_id=identity, application_starts=1,
            launcher_rank=rank, runtime_rank=rank, pid=100+rank, backend='tpu', processes=2, devices=8,
            source_sha256=value['source_sha256'], runtime_sha256=value['runtime_sha256']) for rank in range(2)]
        class Process:
            def __init__(self, pid):
                self.pid, self.returncode = pid, None
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                return self.returncode
        processes = [Process(1000), Process(1001)]
        initial, reattached, killed = [], [], []
        def popen(argv, **kwargs):
            initial.append(argv)
            return processes[len(initial)-1]
        def killpg(pid, signum):
            killed.append(pid)
            next(p for p in processes if p.pid == pid).returncode = -signum
        def run(argv, **kwargs):
            if argv[:3] == ['gcloud', 'storage', 'ls']:
                return subprocess.CompletedProcess(argv, 0, '[]', '')
            rank = int(argv[argv.index('--worker')+1])
            self.assertTrue(argv[-1].startswith('cat /tmp/flaxchat-attachment-start-'))
            return subprocess.CompletedProcess(argv, 0, json.dumps(stamps[rank]), '')
        def transport(argv, log, timeout, cancelled):
            reattached.append(argv)
            rank = int(argv[argv.index('--worker')+1])
            log.write(json.dumps(stamps[rank])+'\n'+f'FLAXCHAT_REMOTE_EXIT_{identity}_{rank}=0\n')
            return 0
        manifest = dict(model_state={'hash': 'model'}, optimizer_state={'hash': 'optimizer'}, training_state={'hash': 'cursor'})
        committed = dict(source_python_sha256='a'*64, resolved_config=dict(runtime={'jax': 'fixture'}, parent={'weights': 'fixture'}), committed_receipt={'step': 4, 'manifest_sha256': campaign.canonical_hash(manifest)})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            guard = root/'guard.json'
            guard.write_text('{}')
            value['guard_receipt'] = str(guard)
            with patch('scripts.gcp_cleanup_guard.verify', return_value=4000), patch('flaxchat.checkpoint_metadata.read_committed_metadata', return_value=committed), patch.object(campaign.uuid, 'uuid4', return_value=Mock(hex=identity)), patch.object(campaign.subprocess, 'Popen', side_effect=popen), patch.object(campaign.subprocess, 'run', side_effect=run), patch.object(campaign.subprocess, 'check_output', side_effect=[json.dumps(dict(state='READY', networkEndpoints=[{'ipAddress': '10.0.0.1'}, {'ipAddress': '10.0.0.2'}])).encode(), json.dumps(manifest).encode()]), patch.object(campaign.os, 'killpg', side_effect=killpg), patch.object(campaign.gcp_tpu_run, 'run_transport', side_effect=transport):
                self.assertEqual(campaign.execute(value, root/'output'), 0)
            self.assertEqual(initial, reattached)
            self.assertEqual(killed, [1000, 1001])
            result = json.loads((root/'output/receipt.json').read_text())
            self.assertTrue(result['passed'])
            self.assertFalse(result['controller_sigkill_qualified'])
            self.assertFalse(result['network_outage_qualified'])
            self.assertFalse(result['lease_cleanup_qualified'])

    def test_worker_exclusive_stamp_prevents_second_application_invocation(self):
        trainer = types.ModuleType('scripts.train_yat_embedding_finetune')
        trainer.add_stage_arguments = lambda parser: parser.add_argument('--steps', type=int)
        trainer.run = Mock(return_value=0)
        jax = types.ModuleType('jax')
        jax.distributed = types.SimpleNamespace(is_initialized=lambda: True)
        jax.default_backend = lambda: 'tpu'
        jax.process_count = lambda: 2
        jax.device_count = lambda: 8
        jax.process_index = lambda: 0
        runtime = {'jax': 'fixture'}
        argv = ['--execution-id', '1'*32, '--source-sha256', 'a'*64,
                '--runtime-sha256', campaign.canonical_hash(runtime), '--', '--steps', '4']
        with TemporaryDirectory() as temporary:
            stamp = str(Path(temporary)/'stamp.json')
            link = campaign.os.link
            published = []
            def publish(source, target):
                # No reader can see an incomplete JSON stamp before atomic linking.
                published.append(json.loads(Path(source).read_text()))
                return link(source, target)
            with patch.dict('sys.modules', {'scripts.train_yat_embedding_finetune': trainer, 'jax': jax}), patch.dict(campaign.os.environ, {'JAX_PROCESS_INDEX': '0'}), patch.object(embedding_contract, 'source_identity', return_value={'sha256': 'a'*64}), patch.object(runtime_module, 'runtime_identity', return_value=runtime), patch.object(campaign, 'stamp_path', return_value=stamp), patch.object(campaign.os, 'link', side_effect=publish), patch('builtins.print'):
                self.assertEqual(campaign.worker(argv), 0)
                with self.assertRaises(FileExistsError):
                    campaign.worker(argv)
        trainer.run.assert_called_once()
        self.assertEqual(trainer.run.call_args.args[0].steps, 4)
        self.assertEqual(published[0]['application_starts'], 1)


if __name__ == '__main__':
    unittest.main()
