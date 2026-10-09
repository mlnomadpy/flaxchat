"""Single cloud-owned GOAT MLM attempt; qualify first, never retry blindly."""
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.representation_run import checked_hash, digest, extract_archive, load_manifest, supervisor_argv

ROOT = Path('/opt/flaxchat-goat-input-1009b')
PREFIX = 'gs://azettaai-yat-eval-0929/goat-input-1009b'
PYTHON = str(ROOT / 'venv/bin/python')
os.environ.update(PYTHONPATH=str(ROOT / 'source'), CLOUDSDK_STORAGE_PROCESS_COUNT='1', CLOUDSDK_STORAGE_THREAD_COUNT='1')
os.chdir(ROOT / 'source')
with (ROOT / 'pipeline-claim').open('x') as claim:
    claim.write('GOAT MLM only; one bounded campaign\n')
status = dict(objective='masked_language_modeling', architecture='goat_input', initialization='exact_resume_checkpoint35500_same_optimizer',
              started_unix=time.time(), status='authenticating_retained_inputs')
def authenticate_committed_parent(checkpoint_root):
    # Pure metadata admission before any paid TPU resource is created.
    import importlib.util
    import re
    markers = subprocess.check_output(['gcloud', 'storage', 'ls',
        checkpoint_root + '/*/commit_success.txt'], text=True, timeout=90).splitlines()
    steps = [int(m.group(1)) for line in markers
        if (m := re.fullmatch(re.escape(checkpoint_root) + r'/(\d+)/commit_success.txt', line))]
    if not steps or max(steps) != 35500:
        raise ValueError('Latest committed checkpoint must be35500 before allocation')
    spec = importlib.util.spec_from_file_location('frozen_checkpoint_metadata', ROOT / 'source/flaxchat/checkpoint_metadata.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.read_committed_metadata(checkpoint_root, 35500, include_receipt=True)
    receipt = metadata['committed_receipt']
    if (receipt['manifest_sha256'] != 'fb288f99fc572e70e9a59ca15e33b4619022ac12dbf2e1e64e6bedd087530b94'
            or receipt['metadata_sha256'] != 'df811f2835b0365b9be60c814c7797e0b2f7cec787f063a0dd885d8b5d72470e'
            or metadata['resolved_config']['encoder']['attention_score'] != 'goat_input'):
        raise ValueError('Committed checkpoint identity differs from frozen parent')
    (ROOT / 'committed-parent-admission.json').write_text(json.dumps(dict(
        passed=True, observed_unix=time.time(), latest_committed_step=35500,
        commit_markers=markers, metadata=metadata), indent=2) + '\n')

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
    authenticate_committed_parent(manifest['checkpoint_output'])
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
    subprocess.run(['gcloud', 'compute', 'instances', 'delete', 'yat-goat-input-1009b', '--project=azettaai', '--zone=us-west4-a', '--quiet'], timeout=120, check=False)
