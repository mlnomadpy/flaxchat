"""Capture complete Linux pins and wheel bytes without importing/executing models.

Run on an existing Linux qualification worker, not a local macOS environment.
A generated bundle is reproducible evidence, not proof of numerical quality.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from scripts.representation_run import digest, validate_lock


def host_platforms(python, identity, *, python_version=None, abi=None):
    """Use pip's complete supported tag set for this exact Linux interpreter."""
    if identity['platform'] != 'linux':
        raise ValueError('Automatic host platforms require the selected Linux interpreter')
    major, minor = identity['python'].split('.')[:2]
    target_version = major + '.' + minor
    target_abi = 'cp' + major + minor
    if python_version and python_version not in (target_version, major + minor):
        raise ValueError('Automatic host platforms cannot target a different Python version')
    if abi and abi != target_abi:
        raise ValueError('Automatic host platforms cannot target a different Python ABI')
    tags = json.loads(subprocess.check_output([python, '-c',
        'import json; from pip._vendor.packaging.tags import sys_tags; print(json.dumps([str(tag) for tag in sys_tags()]))'],
        text=True, timeout=30))
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and len(tag.split('-')) == 3 for tag in tags):
        raise ValueError('Invalid host pip compatibility tags')
    platforms = list(dict.fromkeys(tag.rsplit('-', 1)[1] for tag in tags if not tag.endswith('-any')))
    if not platforms:
        raise ValueError('Host exposes no Linux wheel platforms')
    return {'platforms': platforms, 'python': target_version, 'abi': target_abi, 'host_tags': tags}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, help='Selected Linux environment interpreter')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, help='Cross-download a complete exact lock instead of capturing local runtime')
    parser.add_argument('--target-platform', action='append', default=[], help='Explicit compatible pip Linux wheel platform; repeat for allowed tags')
    parser.add_argument('--auto-host-platforms', action='store_true', help='For Linux --lock downloads, derive the complete compatible tag set from the selected interpreter')
    parser.add_argument('--python-version', help='Target Python major.minor for cross-download')
    parser.add_argument('--abi', help='Target CPython ABI, e.g. cp312')
    parser.add_argument('--index-url', default='https://pypi.org/simple')
    parser.add_argument('--extra-index-url', action='append', default=[])
    args = parser.parse_args(argv)
    identity = json.loads(subprocess.check_output([args.python, '-c',
        'import json,platform,sys; print(json.dumps(dict(platform=sys.platform, machine=platform.machine(), python=platform.python_version())))'], text=True, timeout=30))
    cross_download = args.lock is not None
    target = None
    if args.auto_host_platforms:
        if not cross_download or args.target_platform:
            parser.error('--auto-host-platforms requires --lock and cannot be combined with --target-platform')
        try:
            target = host_platforms(args.python, identity, python_version=args.python_version, abi=args.abi)
        except ValueError as error:
            parser.error(str(error))
        args.target_platform, args.python_version, args.abi = target['platforms'], target['python'], target['abi']
    if cross_download:
        if not args.target_platform or not args.python_version or not args.abi or (not args.auto_host_platforms and any(not tag.startswith('manylinux') or not tag.endswith('_x86_64') for tag in args.target_platform)):
            parser.error('Cross-download requires explicit manylinux x86_64 platforms, Python version and ABI')
    elif identity['platform'] != 'linux':
        parser.error('Runtime capture must run on Linux; use --lock and explicit target tags for download-only cross preparation')
    args.output.mkdir(parents=True, exist_ok=False)
    packages = []
    lock = args.output / 'runtime-lock.txt'
    if cross_download:
        lock.write_text(args.lock.read_text())
    else:
        packages = json.loads(subprocess.check_output([args.python, '-m', 'pip', 'list', '--format=json'], text=True, timeout=60))
        pins = sorted(item['name'] + '==' + item['version'] for item in packages if item['name'].lower() != 'flaxchat')
        lock.write_text('\n'.join(pins) + '\n')
    validate_lock(lock)
    wheels = args.output / 'wheels'
    wheels.mkdir()
    command = [args.python, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
        '--requirement', str(lock), '--dest', str(wheels), '--index-url', args.index_url]
    if cross_download:
        for tag in args.target_platform:
            command.extend(['--platform', tag])
        command.extend(['--python-version', args.python_version, '--implementation', 'cp', '--abi', args.abi])
    for index in args.extra_index_url:
        command.extend(['--extra-index-url', index])
    subprocess.run(command, check=True, timeout=900)
    archive = args.output / 'wheelhouse.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for path in sorted(wheels.glob('*.whl')):
            handle.add(path, arcname=path.name, recursive=False)
    receipt = {'schema_version': 1, 'interpreter': identity, 'packages': packages, 'cross_download_only': cross_download,
        'target': (target or {'platforms': args.target_platform, 'python': args.python_version, 'abi': args.abi}) if cross_download else identity,
        'auto_host_platforms': args.auto_host_platforms,
        'indexes': [args.index_url, *args.extra_index_url], 'runtime_lock_sha256': digest(lock),
        'wheelhouse_sha256': digest(archive), 'wheel_sha256': {path.name: digest(path) for path in wheels.glob('*.whl')},
        'physical_numerical_qualification': False}
    (args.output / 'runtime-bundle.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
