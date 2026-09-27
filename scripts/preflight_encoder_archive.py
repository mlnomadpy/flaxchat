"""Validate a trusted, checksum-pinned encoder source/fixture archive on CPU.

Runs the trainer from the extracted archive, preventing a newer local checkout
or fixture from concealing stale files in the payload that will reach the TPU.
"""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile


def preflight(archive, *, sha256, data_path, snapshot, backend, batch_size=16, steps=6):
    archive, snapshot = Path(archive).resolve(), Path(snapshot).resolve()
    relative = PurePosixPath(data_path)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Data path must stay inside the frozen archive')
    if backend not in ('xla', 'xla_full', 'xla_local', 'pallas') or batch_size <= 0 or steps <= 0:
        raise ValueError('Invalid backend, batch size, or training horizon')
    if not (snapshot / 'config.json').is_file():
        raise ValueError('Local pretrained snapshot config is required')
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError('Frozen archive checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='flaxchat-encoder-preflight-') as directory:
        root = Path(directory)
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            names = set()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Archive must contain only regular files and directories inside its root')
                canonical_name = str(path)
                if canonical_name in names:
                    raise ValueError('Duplicate archive member')
                names.add(canonical_name)
            required = {'scripts/__init__.py', 'flaxchat/__init__.py', 'scripts/train_encoder.py', f'{relative}/manifest.json', f'{relative}/tokens.npy'}
            if not required <= names:
                raise ValueError('Archive is missing its trainer or prepared fixture')
            bundle.extractall(root, members=members, filter='data')
        environment = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=str(root))
        for key in ('JAX_COORDINATOR_ADDRESS', 'JAX_PROCESS_COUNT', 'JAX_PROCESS_INDEX',
                    'TPU_WORKER_HOSTNAMES', 'TPU_WORKER_ID', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'):
            environment.pop(key, None)
        subprocess.run([sys.executable, '-m', 'scripts.train_encoder',
            '--config', str(snapshot / 'config.json'), '--pretrained', str(snapshot),
            '--data', str(root / str(relative)), '--output', str(root / 'unused-output'),
            '--steps', str(steps), '--batch-size', str(batch_size), '--dtype', 'bfloat16',
            '--residual-dtype', 'float32', '--mlm-projection', 'masked',
            '--mlm-loss-backend', backend, '--preflight-only'], cwd=root, env=environment,
            timeout=90, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--data-path', required=True, help='Prepared data directory relative to archive root')
    parser.add_argument('--snapshot', required=True, help='Existing local pretrained snapshot')
    parser.add_argument('--backend', choices=['xla', 'xla_full', 'xla_local', 'pallas'], required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=6)
    args = parser.parse_args()
    preflight(args.archive, sha256=args.sha256, data_path=args.data_path, snapshot=args.snapshot,
              backend=args.backend, batch_size=args.batch_size, steps=args.steps)


if __name__ == '__main__':
    main()
