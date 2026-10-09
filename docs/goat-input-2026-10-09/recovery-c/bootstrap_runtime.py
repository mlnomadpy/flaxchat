"""Install only the manifest's authenticated offline runtime; stdlib bootstrap."""
import json
from pathlib import Path
import subprocess
import sys
from scripts.representation_run import checked_hash, extract_archive, package_inventory, validate_lock

root = Path('/opt/flaxchat-goat-input-1009c')
manifest = json.loads((root / 'run.json').read_text())
lock_config = manifest['runtime_lock']
items = [item for item in manifest['artifacts'] if item['target'] in ('wheels', 'cpython')]
if {item['target'] for item in items} != {'wheels', 'cpython'}:
    raise ValueError('Complete pinned offline CPython/wheelhouse required')
for index, item in enumerate(items):
    archive = root / ('runtime-input-' + str(index))
    subprocess.run(['gcloud', 'storage', 'cp', item['uri'], str(archive)], check=True, timeout=900)
    checked_hash(archive, item['sha256'])
    if not item.get('archive'):
        raise ValueError('Runtime inputs must be authenticated archives')
    extract_archive(archive, root / item['target'], allow_internal_links=item.get('allow_internal_links', False))
    archive.unlink()
lock = root / 'source' / lock_config['path']
checked_hash(lock, lock_config['sha256'])
pins = validate_lock(lock)
interpreter = lock_config['python_executable'].replace('{root}', str(root))
subprocess.run([interpreter, '-m', 'venv', str(root / 'venv')], check=True, timeout=120)
python = str(root / 'venv/bin/python')
subprocess.run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--find-links',
                str(root / lock_config['wheelhouse_target']), '-r', str(lock)], check=True, timeout=900)
subprocess.run([python, '-m', 'pip', 'check'], check=True, timeout=120)
installed = package_inventory(subprocess.check_output([python, '-m', 'pip', 'list', '--format=json'], text=True, timeout=60))
if any(installed.get(name) != version for name, version in pins.items()) or set(installed) - set(pins) - {'pip', 'setuptools'}:
    raise ValueError('Controller package inventory differs from exact runtime lock')
subprocess.run([python, '-c', "import sys,filelock; import scripts.gcp_spot_supervisor; assert 'jax' not in sys.modules; print('controller imports verified; no model backend imported')"], check=True, timeout=60)
(root / 'runtime-bootstrap.json').write_text(json.dumps({'passed': True, 'runtime_lock_sha256': lock_config['sha256'], 'packages': installed, 'model_execution': False}, indent=2) + '\n')
