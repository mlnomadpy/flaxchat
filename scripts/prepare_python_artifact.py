"""Prepare a pinned official standalone Linux CPython archive; no interpreter execution.

Select release/version and expected SHA256 from official release evidence before
invoking this helper. It never guesses a newest version or installs a floating tool.
"""
from __future__ import annotations
import argparse
import json
import posixpath
from pathlib import Path
import re
import tarfile
import time
import urllib.request
from urllib.parse import unquote

from scripts.representation_run import checked_hash, digest


def headless_archive(source, output, expected_sha256):
    """Remove only unused terminal databases; retain all interpreter/runtime bytes.

    Hash the upstream archive before transformation. Do not relax extraction
    filters to accommodate optional terminfo aliases on older bootstrap Python.
    """
    checked_hash(source, expected_sha256)
    if Path(output).exists():
        raise ValueError('Refusing to overwrite an interpreter artifact')
    removed = []
    kept = []
    with tarfile.open(source) as original:
        for entry in original.getmembers():
            name = Path(entry.name)
            if name.is_absolute() or '..' in name.parts or not (entry.isfile() or entry.isdir() or entry.issym() or entry.islnk()):
                raise ValueError('Unsafe upstream interpreter member: ' + entry.name)
            if entry.issym() or entry.islnk():
                parent = posixpath.dirname(entry.name) if entry.issym() else ''
                target = posixpath.normpath(posixpath.join(parent, entry.linkname))
                if entry.linkname.startswith('/') or target == '..' or target.startswith('../'):
                    raise ValueError('Unsafe upstream interpreter link: ' + entry.name)
            if entry.name == 'python/share/terminfo' or entry.name.startswith('python/share/terminfo/'):
                removed.append(entry.name)
                continue
            kept.append(entry)
        with tarfile.open(output, 'w:gz') as result:
            for entry in kept:
                result.addfile(entry, original.extractfile(entry) if entry.isfile() else None)
    return {'policy': 'headless-exclude-terminfo-v1', 'upstream_sha256': expected_sha256,
            'artifact_sha256': digest(output), 'excluded_members': removed,
            'interpreter_runtime_bytes_changed': False}


def validate_python_url(url, release, version):
    if not re.fullmatch(r'\d{8}', release) or not re.fullmatch(r'3\.\d+\.\d+', version):
        raise ValueError('Exact official release date and CPython version required')
    prefix = 'https://github.com/astral-sh/python-build-standalone/releases/download/' + release + '/'
    if not url.startswith(prefix):
        raise ValueError('Require the pinned official python-build-standalone release URL')
    asset = unquote(url[len(prefix):])
    if not re.fullmatch(r'cpython-' + re.escape(version) + r'\+' + re.escape(release) + r'-x86_64-unknown-linux-gnu-install_only(?:_stripped)?\.tar\.gz', asset):
        raise ValueError('Expected exact CPython Linux x86_64 install-only asset')
    return 'python/bin/python' + '.'.join(version.split('.')[:2])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--python-version', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, help='Reuse an already downloaded archive with the same official hash')
    parser.add_argument('--headless', action='store_true', help='Exclude unused terminfo aliases while preserving interpreter/runtime bytes')
    args = parser.parse_args(argv)
    try:
        executable = validate_python_url(args.url, args.release, args.python_version)
        if not re.fullmatch('[0-9a-f]{64}', args.sha256):
            raise ValueError('Expected official SHA256 required')
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('Refusing to overwrite an existing interpreter artifact')
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    deadline = time.monotonic() + 900
    transformation = None
    try:
        response_source = args.source_archive.open('rb') if args.source_archive else urllib.request.urlopen(args.url, timeout=120)
        with response_source as response, temporary.open('xb') as output:
            total = 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Interpreter artifact download exceeded 900-second deadline')
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 500_000_000:
                    raise ValueError('Interpreter artifact exceeds 500 MB download bound')
                output.write(chunk)
        checked_hash(temporary, args.sha256)
        with tarfile.open(temporary) as archive:
            if executable not in archive.getnames():
                raise ValueError('Pinned archive lacks the declared interpreter executable')
        if args.headless:
            try:
                transformation = headless_archive(temporary, args.output, args.sha256)
            except BaseException:
                args.output.unlink(missing_ok=True)
                raise
        else:
            temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {'schema_version': 1, 'url': args.url, 'release': args.release,
        'python_version': args.python_version, 'sha256': digest(args.output), 'upstream_sha256': args.sha256,
        'transformation': transformation, 'executed': False,
        'manifest_artifact': {'local_path': str(args.output), 'uri': 'gs://REPLACE/immutable/python.tar.gz',
            'target': 'cpython', 'archive': True, 'allow_internal_links': True, 'sha256': digest(args.output)},
        'python_executable': '{root}/cpython/' + executable}
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
