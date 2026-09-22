import json
import subprocess
from unittest.mock import patch

from infra.tpu.spot_watchdog import main


def test_delete_timeout_is_retried(tmp_path):
    output = tmp_path / 'watchdog.json'
    argv = ['watchdog', '--project', 'test', '--zone', 'zone', '--queue', 'flaxchat-validation-test',
            '--seconds', '60', '--output', str(output)]
    results = [subprocess.TimeoutExpired('gcloud', 180), subprocess.CompletedProcess('gcloud', 0, '', '')]
    with patch('sys.argv', argv), patch('infra.tpu.spot_watchdog.time.time', side_effect=[0, 61]), patch('infra.tpu.spot_watchdog.subprocess.run', side_effect=results) as run:
        main()
    assert run.call_count == 2
    assert json.loads(output.read_text())['status'] == 'deleted'


def test_queue_state_rejection_deletes_owned_node_then_retries(tmp_path):
    output = tmp_path / 'watchdog.json'
    argv = ['watchdog', '--project', 'test', '--zone', 'zone', '--queue', 'flaxchat-validation-test',
            '--seconds', '60', '--output', str(output)]
    results = [subprocess.CompletedProcess('gcloud', 1, '', 'PROVISIONING'),
               subprocess.CompletedProcess('gcloud', 0, '', ''),
               subprocess.CompletedProcess('gcloud', 0, '', '')]
    with patch('sys.argv', argv), patch('infra.tpu.spot_watchdog.time.time', side_effect=[0, 61]), patch('infra.tpu.spot_watchdog.time.sleep'), patch('infra.tpu.spot_watchdog.subprocess.run', side_effect=results) as run:
        main()
    assert run.call_count == 3
    assert run.call_args_list[1].args[0][3:6] == ['tpu-vm', 'delete', 'flaxchat-validation-test']
    assert json.loads(output.read_text())['status'] == 'deleted'
