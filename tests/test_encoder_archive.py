import hashlib
import io
import json
import tarfile

import pytest

from scripts.preflight_encoder_archive import preflight


def bundle(tmp_path, extra=None, trainer=None):
    archive = tmp_path / 'source.tar.gz'
    files = {'scripts/__init__.py': '', 'flaxchat/__init__.py': '', 'scripts/train_encoder.py': trainer or 'raise RuntimeError("not invoked by mocked test")',
             'data/manifest.json': '{}', 'data/tokens.npy': 'fixture'}
    with tarfile.open(archive, 'w:gz') as tar:
        for name, text in files.items():
            content = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
        if extra:
            tar.addfile(extra)
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'config.json').write_text('{}')
    return archive, dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                        data_path='data', snapshot=snapshot, backend='xla_full')


@pytest.mark.parametrize('kind', ['checksum', 'traversal', 'symlink', 'duplicate', 'alias-duplicate', 'data-path'])
def test_invalid_archive_never_executes(tmp_path, monkeypatch, kind):
    extra = None
    if kind == 'traversal':
        extra = tarfile.TarInfo('../escape')
    if kind == 'symlink':
        extra = tarfile.TarInfo('link')
        extra.type, extra.linkname = tarfile.SYMTYPE, '/tmp'
    if kind in ('duplicate', 'alias-duplicate'):
        extra = tarfile.TarInfo(('data/' if kind == 'duplicate' else './data/') + 'manifest.json')
    archive, args = bundle(tmp_path, extra=extra)
    if kind == 'checksum':
        args['sha256'] = '0' * 64
    if kind == 'data-path':
        args['data_path'] = '../escape'
    monkeypatch.setattr('scripts.preflight_encoder_archive.subprocess.run',
                        lambda *a, **k: pytest.fail('Must not execute invalid archive'))
    with pytest.raises(ValueError):
        preflight(archive, **args)


def test_executes_frozen_trainer_and_isolates_distributed_environment(tmp_path, monkeypatch, capfd):
    # The real project trainer would reject the deliberately minimal fixture.
    # Executing this fixture proves we imported the archived code instead.
    trainer = '''import json, os, pathlib, sys
args = sys.argv
assert '--preflight-only' in args
assert args[args.index('--mlm-loss-backend') + 1] == 'xla_full'
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not any(key in os.environ for key in ('JAX_PROCESS_COUNT', 'SLURM_NTASKS'))
assert pathlib.Path(args[args.index('--data') + 1], 'tokens.npy').read_text() == 'fixture'
print(json.dumps({'frozen_trainer': True}))
'''
    archive, args = bundle(tmp_path, trainer=trainer)
    monkeypatch.setenv('JAX_PROCESS_COUNT', '2')
    monkeypatch.setenv('SLURM_NTASKS', '2')
    preflight(archive, **args)
    assert json.loads(capfd.readouterr().out)['frozen_trainer'] is True


def test_frozen_trainer_failure_is_not_qualification(tmp_path):
    import subprocess
    archive, args = bundle(tmp_path, trainer='raise SystemExit(7)')
    with pytest.raises(subprocess.CalledProcessError) as failure:
        preflight(archive, **args)
    assert failure.value.returncode == 7


def test_missing_package_cannot_fall_back_to_installed_checkout(tmp_path, monkeypatch):
    archive, args = bundle(tmp_path)
    with tarfile.open(archive) as old:
        members = [(member, old.extractfile(member).read()) for member in old.getmembers()
                   if member.name != 'scripts/__init__.py']
    with tarfile.open(archive, 'w:gz') as rebuilt:
        for member, content in members:
            rebuilt.addfile(member, io.BytesIO(content))
    args['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr('scripts.preflight_encoder_archive.subprocess.run',
                        lambda *a, **k: pytest.fail('Must reject incomplete package before import'))
    with pytest.raises(ValueError, match='missing'):
        preflight(archive, **args)
