import shlex

import pytest

from scripts.gcp_tpu_run import worker_command


def test_command_quotes_literal_shell_characters_and_sets_rank():
    result = worker_command(['python', '-c', 'print("$(do-not-run)")'], rank=1, count=4,
                            coordinator='10.0.0.1:1234', directory='/tmp/space dir',
                            timeout=60, distributed=True)
    assert "cd '/tmp/space dir'" in result
    tokens = shlex.split(result.split(' && ')[1].split('); command_status')[0])
    assert tokens[-1] == 'print("$(do-not-run)")'
    assert 'JAX_PROCESS_INDEX=1' in tokens
    assert tokens[:3] == ['timeout', '--signal=KILL', '60']


def test_setup_does_not_initialize_distributed_runtime():
    result = worker_command(['true'], rank=0, count=4, coordinator='host:123', directory='/tmp',
                            timeout=60, distributed=False)
    assert 'JAX_' not in result


@pytest.mark.parametrize('rank,count,timeout', [(4, 4, 60), (0, 0, 60), (0, 4, 0)])
def test_invalid_worker_coordinates(rank, count, timeout):
    with pytest.raises(ValueError):
        worker_command(['true'], rank=rank, count=count, coordinator='host:123', directory='/tmp',
                       timeout=timeout, distributed=True)


def test_remote_failure_is_reported_without_ssh_retry_and_command_is_once(tmp_path):
    import subprocess
    import sys
    import uuid
    from scripts.gcp_tpu_run import remote_returncode
    execution_id = uuid.uuid4().hex
    counter = tmp_path / 'counter'
    command = worker_command([sys.executable, '-c',
        'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text(p.read_text()+"x" if p.exists() else "x"); sys.exit(255)',
        str(counter)], rank=0, count=1, coordinator='', directory=str(tmp_path), timeout=5,
        distributed=False, execution_id=execution_id)
    # Test the shell protocol on macOS too, without depending on GNU timeout.
    command = command.replace('timeout --signal=KILL 5 ', '')
    first = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    replay = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    assert first.returncode == replay.returncode == 0
    assert remote_returncode(first.stdout, execution_id, 0, 0) == 255
    assert remote_returncode(replay.stdout, execution_id, 0, 0) == 125
    assert counter.read_text() == 'x'


def test_remote_success_requires_one_matching_completion_marker():
    from scripts.gcp_tpu_run import remote_returncode
    identity = 'a' * 32
    marker = f'FLAXCHAT_REMOTE_EXIT_{identity}_2=0\n'
    assert remote_returncode(marker, identity, 2, 0) == 0
    assert remote_returncode('', identity, 2, 0) == 125
    assert remote_returncode(marker * 2, identity, 2, 0) == 125
    assert remote_returncode(marker, identity, 1, 0) == 125
    assert remote_returncode(marker, identity, 2, 255) == 255


def test_fault_injection_targets_coordination_rank_despite_tpu_reordering():
    from scripts.validate_multihost_interruption import coordinator_process
    assert coordinator_process({'JAX_PROCESS_INDEX': '0'}, runtime_rank=3)
    assert not coordinator_process({'JAX_PROCESS_INDEX': '2'}, runtime_rank=0)
    assert coordinator_process({}, runtime_rank=0)
