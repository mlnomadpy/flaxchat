"""Bound a dedicated Spot queued resource lifetime, independent of test success."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--node', help='Owned node ID; defaults to the queue ID')
    parser.add_argument('--seconds', type=int, default=7200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    node = args.node or args.queue
    if not 60 <= args.seconds <= 7200 or not all(name.startswith('flaxchat-validation-') for name in (args.queue, node)):
        parser.error('Only dedicated validation queues with deadlines up to two hours are allowed')
    deadline = time.time() + args.seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'deadline_epoch': deadline, 'queue': args.queue, 'status': 'armed'}))
    while time.time() < deadline:
        time.sleep(min(30, deadline - time.time()))
    command = ['gcloud', 'compute', 'tpus', 'queued-resources', 'delete', args.queue,
               '--project', args.project, '--zone', args.zone, '--force', '--quiet']
    for _ in range(5):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            args.output.write_text(json.dumps({'deadline_epoch': deadline, 'queue': args.queue,
                                              'status': 'delete_timeout_retrying'}))
            continue
        args.output.write_text(json.dumps({'deadline_epoch': deadline, 'queue': args.queue,
                                          'status': 'deleted' if result.returncode == 0 or 'NOT_FOUND' in result.stderr else 'delete_failed',
                                          'returncode': result.returncode, 'output': result.stdout + result.stderr}))
        if result.returncode == 0 or 'NOT_FOUND' in result.stderr:
            return
        # Some control-plane states reject force-deleting the queue. Delete
        # only its explicitly owned node, then retry the queue after suspension.
        node_command = ['gcloud', 'compute', 'tpus', 'tpu-vm', 'delete', node,
                        '--project', args.project, '--zone', args.zone, '--quiet']
        try:
            node_result = subprocess.run(node_command, capture_output=True, text=True, timeout=180)
            args.output.write_text(json.dumps({'deadline_epoch': deadline, 'queue': args.queue,
                'status': 'node_delete_attempted', 'returncode': node_result.returncode,
                'output': node_result.stdout + node_result.stderr}))
        except subprocess.TimeoutExpired:
            args.output.write_text(json.dumps({'deadline_epoch': deadline, 'queue': args.queue,
                                              'status': 'node_delete_timeout_retrying'}))
        time.sleep(30)
    raise SystemExit('Resource cleanup failed; manual deletion required')


if __name__ == '__main__':
    main()
