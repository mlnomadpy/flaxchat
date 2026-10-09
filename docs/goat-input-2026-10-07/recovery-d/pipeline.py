"""Single cloud-owned GOAT MLM attempt; qualify first, never retry blindly."""
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.representation_run import checked_hash, digest, extract_archive, load_manifest, supervisor_argv

ROOT = Path('/opt/flaxchat-goat-input-1007d')
PREFIX = 'gs://azettaai-yat-eval-0929/goat-input-1007d'
PYTHON = str(ROOT / 'venv/bin/python')
os.environ.update(PYTHONPATH=str(ROOT / 'source'), CLOUDSDK_STORAGE_PROCESS_COUNT='1', CLOUDSDK_STORAGE_THREAD_COUNT='1')
os.chdir(ROOT / 'source')
with (ROOT / 'pipeline-claim').open('x') as claim:
    claim.write('GOAT MLM only; one bounded campaign\n')
status = dict(objective='masked_language_modeling', architecture='goat_input', initialization='exact_resume_checkpoint3000_same_optimizer',
              started_unix=time.time(), status='authenticating_retained_inputs')
def call(argv, seconds):
    subprocess.run(argv, check=True, timeout=seconds)
def publish():
    (ROOT / 'pipeline-status.json').write_text(json.dumps(status, indent=2) + '\n')
    call(['gcloud', 'storage', 'cp', str(ROOT / 'pipeline-status.json'), PREFIX + '/pipeline-status.json'], 90)
try:
    ledger = json.loads((ROOT / 'campaign-ledger.json').read_text())
    template = load_manifest(ROOT / 'run.json')
    required = template['deployment']['hourly_usd'] * (template['deployment']['attempt_seconds'] + 1800) / 3600 + 10
    assert sum(a['reservation_usd'] for a in ledger['attempts'].values()) + required <= ledger['budget_usd'], 'Budget admission including cleanup reserve failed'
    publish()
    manifest = load_manifest(ROOT / 'run.json')
    for index, item in enumerate(manifest['artifacts']):
        if item['target'] in ('wheels', 'cpython'):
            continue
        target = ROOT / item['target']
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = ROOT / f'input-{index}' if item.get('archive') else target
        call(['gcloud', 'storage', 'cp', item['uri'], str(downloaded)], 900)
        checked_hash(downloaded, item['sha256'])
        if item.get('archive'):
            extract_archive(downloaded, target)
            downloaded.unlink()
    preflight = [PYTHON, '-m', 'scripts.run_yat_mlm_continuation', '--root', str(ROOT),
                 '--output', manifest['checkpoint_output'], '--random-init', '--goat-score-source', 'input',
                 '--parent-metadata-sha256', '35cff96b97c0a9d5ba401c39545dfeff62e1c73e527d5f5661ba0960042bf4f4',
                 '--parent-manifest-sha256', 'bf891c59974ec83f69349e08e704a99388fd8bdad6fd356ee2be9dfa1fb1b4d3', '--preflight-only']
    subprocess.run(preflight, check=True, timeout=1800, env=os.environ | {'JAX_PLATFORMS': 'cpu'})
    status.update(status='requesting_TPU_for_physical_qualification', corpus=json.loads((ROOT / 'corpus/COMPLETE.json').read_text()),
                  manifest_sha256=digest(ROOT / 'run.json'))
    publish()
    argv = supervisor_argv(manifest, PREFIX + '/run.json', digest(ROOT / 'run.json'), ROOT / 'tpu-controller')
    argv[argv.index('--setup-timeout-seconds') + 1] = '1800'
    argv += ['--ancillary-reserve-usd', '10']
    from scripts.gcp_spot_supervisor import main
    result = main(argv)
    status.update(status='controller_completed', controller_exit=result)
    publish()
    if result:
        raise RuntimeError(f'TPU controller exited {result}')
except BaseException as error:
    status.update(status='failed', error=f'{type(error).__name__}: {error}')
    try:
        publish()
    except Exception:
        pass
    raise
finally:
    for path in (ROOT / 'pipeline.log', ROOT / 'pipeline-status.json', ROOT / 'campaign-ledger.json'):
        if path.exists():
            subprocess.run(['gcloud', 'storage', 'cp', str(path), PREFIX + '/controller/'], timeout=90, check=False)
    if (ROOT / 'tpu-controller').exists():
        subprocess.run(['gcloud', 'storage', 'rsync', str(ROOT / 'tpu-controller'), PREFIX + '/controller/tpu-controller', '--recursive'], timeout=300, check=False)
    subprocess.run(['gcloud', 'compute', 'instances', 'delete', 'yat-goat-input-1007d', '--project=azettaai', '--zone=us-west4-a', '--quiet'], timeout=120, check=False)
