"""Bounded model-free GCS stream diagnostic and authenticated download proposal.

Range mode reports transport evidence, never whole-artifact authentication.
Full mode writes a separate file only after exact byte/SHA authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import subprocess
import time

FORMAT = 'flaxchat-artifact-transfer-v1'
MAX_ARTIFACT_BYTES = 2 * 1024 ** 3
MAX_RANGE_BYTES = 64 * 1024 ** 2
MAX_STDERR_BYTES = 64 * 1024


def validate_manifest(manifest):
    if (not isinstance(manifest, dict) or manifest.get('format') != FORMAT
            or not isinstance(manifest.get('uri'), str)
            or re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+/[^\s#]+#[1-9][0-9]*', manifest['uri']) is None
            or type(manifest.get('bytes')) is not int
            or not 1 <= manifest['bytes'] <= MAX_ARTIFACT_BYTES
            or not isinstance(manifest.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', manifest['sha256']) is None
            or not isinstance(manifest.get('independent_identity_sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', manifest['independent_identity_sha256']) is None):
        raise ValueError('Exact generation, finite artifact size and independent SHA identity required')
    return manifest


def stream_transfer(manifest, *, output, mode='range', range_bytes=MAX_RANGE_BYTES,
                    phase_seconds=180, total_seconds=600, finalization_margin_seconds=240,
                    progress=None, command_factory=None):
    """Observe one bounded stream; no retries or model imports. Test factory is literal-byte only."""
    validate_manifest(manifest)
    if (mode not in ('range', 'full') or type(range_bytes) is not int
            or not 1 <= range_bytes <= min(MAX_RANGE_BYTES, manifest['bytes'])
            or any(type(value) is not int for value in
                   (phase_seconds, total_seconds, finalization_margin_seconds))
            or not 1 <= phase_seconds <= 1800 or not 1 <= total_seconds <= 2100
            or not 30 <= finalization_margin_seconds < total_seconds
            or phase_seconds > total_seconds - finalization_margin_seconds):
        raise ValueError('Explicit bounded phase/global deadlines and finalization margin required')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh separate output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    if temporary.exists() or temporary.is_symlink():
        raise ValueError('Fresh partial output required; inspect original execution before retry')
    expected = range_bytes if mode == 'range' else manifest['bytes']
    command = ['gcloud', 'storage', 'cat', manifest['uri']]
    if mode == 'range':
        command.append('--range=0-' + str(range_bytes - 1))
    command.append('--quiet')
    if command_factory is not None:
        command = command_factory(command)
    started = time.monotonic()
    deadline = started + min(phase_seconds, total_seconds - finalization_margin_seconds)
    report = {'format': FORMAT, 'status': 'started', 'mode': mode, 'uri': manifest['uri'],
              'expected_artifact_bytes': manifest['bytes'], 'expected_transfer_bytes': expected,
              'expected_artifact_sha256': manifest['sha256'],
              'independent_identity_sha256': manifest['independent_identity_sha256'],
              'phase_seconds': phase_seconds, 'total_seconds': total_seconds,
              'finalization_margin_seconds': finalization_margin_seconds,
              'whole_artifact_authenticated': False, 'model_execution': False,
              'python_version': platform.python_version(), 'platform': platform.platform(),
              'client': 'gcloud-storage-cat', 'client_checksum': 'not computed by cat',
              'bytes_received': 0, 'stderr_bytes': 0, 'stderr_truncated': False}
    process = None
    byte_hash, stderr_hash = hashlib.sha256(), hashlib.sha256()
    stderr_retained = bytearray()
    samples = []
    next_progress = started
    try:
        with temporary.open('xb') as target:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            report['owned_pid'] = process.pid
            selector = selectors.DefaultSelector()
            try:
                for name, pipe in (('stdout', process.stdout), ('stderr', process.stderr)):
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, name)
                while selector.get_map():
                    now = time.monotonic()
                    if now >= deadline:
                        raise TimeoutError('Artifact stream phase deadline exceeded')
                    if now >= next_progress:
                        sample = {'elapsed_seconds': now - started,
                                  'bytes_received': report['bytes_received'],
                                  'stderr_bytes': report['stderr_bytes']}
                        samples.append(sample)
                        if progress is not None:
                            progress(sample)
                        next_progress = now + 5
                    for key, _ in selector.select(timeout=min(1, deadline - now)):
                        data = os.read(key.fileobj.fileno(), 1024 * 1024)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        if key.data == 'stderr':
                            report['stderr_bytes'] += len(data)
                            stderr_hash.update(data)
                            stderr_retained.extend(data[:max(0, MAX_STDERR_BYTES - len(stderr_retained))])
                            report['stderr_truncated'] = report['stderr_bytes'] > MAX_STDERR_BYTES
                            continue
                        report['bytes_received'] += len(data)
                        if report['bytes_received'] > expected:
                            raise ValueError('Stream exceeded exact byte budget; range semantics unverified')
                        byte_hash.update(data)
                        target.write(data)
                code = process.wait(timeout=max(.001, deadline - time.monotonic()))
                report['client_returncode'] = code
                if code != 0:
                    raise ValueError('Artifact transfer client failed')
            finally:
                selector.close()
        if report['bytes_received'] != expected:
            raise ValueError('Stream byte count differs from exact requested size')
        if mode == 'full' and byte_hash.hexdigest() != manifest['sha256']:
            raise ValueError('Actual streamed artifact SHA differs from independent identity')
        report['status'] = 'completed'
        if mode == 'full':
            # Do not overwrite an output created by another process during transfer.
            os.link(temporary, output)
            temporary.unlink()
            report['whole_artifact_authenticated'] = True
        else:
            temporary.unlink()
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            # Ensure descendants do not survive a parent that already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        if temporary.exists():
            temporary.unlink()
        elapsed = time.monotonic() - started
        report.update(elapsed_seconds=elapsed,
                      bytes_per_second=report['bytes_received'] / elapsed if elapsed else 0,
                      observed_stream_sha256=byte_hash.hexdigest(),
                      stderr_sha256=stderr_hash.hexdigest(),
                      stderr_retained_bytes=len(stderr_retained),
                      partial_removed=not temporary.exists(), progress=samples,
                      range_whole_artifact_identity_proven=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--mode', choices=('range', 'full'), default='range')
    parser.add_argument('--range-bytes', type=int, default=MAX_RANGE_BYTES)
    parser.add_argument('--phase-seconds', type=int, default=180)
    parser.add_argument('--total-seconds', type=int, default=600)
    parser.add_argument('--finalization-margin-seconds', type=int, default=240)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        # Let stream_transfer's finally kill its owned process group on TERM.
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.receipt.exists() or args.receipt.is_symlink() or args.receipt.resolve() == args.output.resolve():
        raise ValueError('Fresh distinct receipt required')
    if args.manifest.is_symlink() or args.manifest.stat().st_size > 64 * 1024:
        raise ValueError('Bounded regular transfer manifest required')
    manifest = json.loads(args.manifest.read_text())
    report = stream_transfer(manifest, output=args.output, mode=args.mode,
                             range_bytes=args.range_bytes, phase_seconds=args.phase_seconds,
                             total_seconds=args.total_seconds,
                             finalization_margin_seconds=args.finalization_margin_seconds,
                             progress=lambda item: print(json.dumps({'event': 'artifact_stream_progress', **item}), flush=True))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'bytes_received': report['bytes_received'],
                      'elapsed_seconds': report['elapsed_seconds'],
                      'whole_artifact_authenticated': report['whole_artifact_authenticated']}), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
